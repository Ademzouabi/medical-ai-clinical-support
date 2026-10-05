from pathlib import Path
import sys
import unittest

from fastapi.testclient import TestClient


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import api
from terminology_matcher import TERMINOLOGY


def complete_positive_text():
    return " ".join(
        f"Currently, I have {concept['preferred_term']}."
        for concept in TERMINOLOGY
        if concept.get("ml_feature_mapping")
    )


class ApiTests(unittest.TestCase):
    def setUp(self):
        api.sessions.clear()
        self.client = TestClient(api.app)

    def test_create_and_retrieve_session(self):
        created = self.client.post("/sessions", json={"text": "I have a cough."})

        self.assertEqual(created.status_code, 201)
        payload = created.json()
        self.assertIsInstance(payload["session_id"], str)
        self.assertEqual(payload["original_text"], "I have a cough.")
        self.assertEqual(payload["feature_states"]["cough"], 1)
        self.assertFalse(payload["prediction_ready"])
        self.assertIsNotNone(payload["next_question"])
        self.assertIn("concept_id", payload["next_question"])

        retrieved = self.client.get(f"/sessions/{payload['session_id']}")
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(retrieved.json(), payload)

    def test_submit_clarification_answer_updates_existing_session(self):
        created = self.client.post("/sessions", json={"text": ""}).json()
        session_id = created["session_id"]
        asked_concept = created["next_question"]["concept_id"]

        updated = self.client.post(
            f"/sessions/{session_id}/answers",
            json={"answer": "Yes, I currently have it."},
        )

        self.assertEqual(updated.status_code, 200)
        state = updated.json()
        statuses = {
            item["concept_id"]: item["status"] for item in state["concept_matches"]
        }
        self.assertEqual(statuses[asked_concept], "YES")
        self.assertEqual(
            state["clarification_history"][-1]["interpreted_status"], "YES"
        )
        self.assertNotEqual(state["next_question"]["concept_id"], asked_concept)

    def test_incomplete_prediction_is_rejected_without_result(self):
        created = self.client.post("/sessions", json={"text": "I have fever."}).json()

        response = self.client.post(f"/sessions/{created['session_id']}/predict")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "prediction_not_ready")
        state = self.client.get(f"/sessions/{created['session_id']}").json()
        self.assertIsNone(state["prediction_result"])
        self.assertFalse(state["prediction_ready"])

    def test_complete_session_runs_real_prediction(self):
        created = self.client.post(
            "/sessions",
            json={"text": complete_positive_text()},
        )

        self.assertEqual(created.status_code, 201)
        state = created.json()
        self.assertTrue(state["prediction_ready"])
        self.assertEqual(len(state["feature_states"]), 26)
        self.assertTrue(all(value == 1 for value in state["feature_states"].values()))

        prediction = self.client.post(f"/sessions/{state['session_id']}/predict")
        self.assertEqual(prediction.status_code, 200)
        payload = prediction.json()
        self.assertEqual(payload["session_id"], state["session_id"])
        self.assertTrue(payload["predictions"])
        self.assertTrue(
            all(set(item) == {"disease", "score"} for item in payload["predictions"])
        )
        self.assertTrue(
            all(isinstance(item["disease"], str) for item in payload["predictions"])
        )

    def test_unknown_session_returns_not_found(self):
        response = self.client.get("/sessions/not-a-session")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "session_not_found")

    def test_invalid_request_bodies_are_rejected(self):
        missing_text = self.client.post("/sessions", json={})
        wrong_text_type = self.client.post("/sessions", json={"text": 123})
        created = self.client.post("/sessions", json={"text": ""}).json()
        missing_answer = self.client.post(
            f"/sessions/{created['session_id']}/answers",
            json={},
        )

        self.assertEqual(missing_text.status_code, 422)
        self.assertEqual(wrong_text_type.status_code, 422)
        self.assertEqual(missing_answer.status_code, 422)


if __name__ == "__main__":
    unittest.main()
