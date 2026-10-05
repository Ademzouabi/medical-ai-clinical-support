import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import predict
from pipeline import (
    PredictionNotReadyError,
    can_predict,
    get_current_state,
    predict as run_pipeline_prediction,
    start_pipeline,
    submit_clarification,
)
from terminology_matcher import ML_FEATURES, TERMINOLOGY


MAPPED_CONCEPTS = [
    concept for concept in TERMINOLOGY if concept.get("ml_feature_mapping")
]
MAPPED_IDS = {concept["concept_id"] for concept in MAPPED_CONCEPTS}


def evidence_text(status):
    if status == "YES":
        prefix = "Currently, I have "
    elif status == "NO":
        prefix = "I do not have "
    else:
        raise ValueError("Evidence text only supports explicit YES or NO")
    return " ".join(
        f"{prefix}{concept['preferred_term']}." for concept in MAPPED_CONCEPTS
    )


def prediction_stub():
    return Mock(return_value=[{"disease": "stub", "score": 0.5}])


class EndToEndTests(unittest.TestCase):
    def test_complete_positive_text_runs_real_prediction(self):
        session = start_pipeline(evidence_text("YES"))
        state = get_current_state(session)
        matches = {
            match["concept_id"]: match["status"] for match in state["concept_matches"]
        }

        self.assertTrue(MAPPED_IDS.issubset(matches))
        self.assertTrue(all(matches[concept_id] == "YES" for concept_id in MAPPED_IDS))
        self.assertEqual(state["unresolved_features"], [])
        self.assertIsNone(state["next_question"])
        self.assertTrue(state["prediction_ready"])
        self.assertTrue(can_predict(session))
        self.assertEqual(list(state["feature_states"]), list(predict.FEATURES))
        self.assertEqual(len(state["feature_states"]), 26)
        self.assertTrue(
            all(
                type(value) is int and value == 1
                for value in state["feature_states"].values()
            )
        )

        result = run_pipeline_prediction(session)
        self.assertEqual(len(result), len(predict.model.classes_))
        self.assertTrue(all(set(row) == {"disease", "score"} for row in result))
        self.assertTrue(all(isinstance(row["disease"], str) for row in result))
        self.assertTrue(all(math.isfinite(row["score"]) for row in result))
        self.assertTrue(all(0.0 <= row["score"] <= 1.0 for row in result))
        self.assertAlmostEqual(sum(row["score"] for row in result), 1.0, places=6)
        self.assertEqual(
            [row["score"] for row in result],
            sorted((row["score"] for row in result), reverse=True),
        )
        self.assertEqual(get_current_state(session)["prediction_result"], result)

    def test_complete_negative_text_resolves_every_feature_to_zero(self):
        session = start_pipeline(evidence_text("NO"))
        state = get_current_state(session)
        matches = {
            match["concept_id"]: match["status"] for match in state["concept_matches"]
        }

        self.assertTrue(MAPPED_IDS.issubset(matches))
        self.assertTrue(all(matches[concept_id] == "NO" for concept_id in MAPPED_IDS))
        self.assertEqual(state["unresolved_features"], [])
        self.assertTrue(state["prediction_ready"])
        self.assertEqual(set(state["feature_states"].values()), {0})

        stub = prediction_stub()
        with patch.object(predict, "predict_differential", stub):
            self.assertEqual(run_pipeline_prediction(session), stub.return_value)
        stub.assert_called_once()
        self.assertTrue(all(value == 0 for value in stub.call_args.args[0].values()))

    def test_partial_information_blocks_and_never_calls_model(self):
        session = start_pipeline("I have a cough.")
        state = get_current_state(session)
        matches = {
            match["concept_id"]: match["status"] for match in state["concept_matches"]
        }
        self.assertEqual(matches["cough"], "YES")
        self.assertEqual(state["feature_states"]["cough"], 1)
        self.assertTrue(state["unresolved_features"])
        self.assertFalse(state["prediction_ready"])

        stub = prediction_stub()
        with patch.object(predict, "predict_differential", stub):
            with self.assertRaises(PredictionNotReadyError):
                run_pipeline_prediction(session)
        stub.assert_not_called()

    def test_ambiguous_language_remains_unknown(self):
        cases = (
            ("I might have a fever.", "fever"),
            ("I think I'm short of breath.", "shortness_of_breath"),
            ("Maybe I have chest pain.", "chest_pain"),
            ("I don't know whether I'm wheezing.", "wheezing"),
            ("I think I used to have chest pain.", "chest_pain"),
            ("Maybe, but I'm not certain about fever.", "fever"),
        )
        for text, concept_id in cases:
            with self.subTest(text=text):
                session = start_pipeline(text)
                state = get_current_state(session)
                statuses = {
                    match["concept_id"]: match["status"]
                    for match in state["concept_matches"]
                }
                self.assertEqual(statuses.get(concept_id), "UNKNOWN")
                self.assertFalse(state["prediction_ready"])
                self.assertTrue(state["unresolved_features"])
                self.assertFalse(
                    any(
                        value == 0
                        for key, value in state["feature_states"].items()
                        if key
                        in {
                            "fever",
                            "shortness_of_breath",
                            "sharp_chest_pain",
                            "wheezing",
                        }
                    )
                )

    def test_negation_and_shared_feature_semantics(self):
        cough_session = start_pipeline("I don't have a cough.")
        cough_state = get_current_state(cough_session)
        cough_matches = {
            item["concept_id"]: item["status"]
            for item in cough_state["concept_matches"]
        }
        self.assertEqual(cough_matches["cough"], "NO")
        self.assertEqual(cough_state["feature_states"]["cough"], 0)

        mixed_session = start_pipeline("I have fever but no shortness of breath.")
        mixed_state = get_current_state(mixed_session)
        mixed_matches = {
            item["concept_id"]: item["status"]
            for item in mixed_state["concept_matches"]
        }
        self.assertEqual(mixed_matches["fever"], "YES")
        self.assertEqual(mixed_matches["shortness_of_breath"], "NO")
        self.assertIsNone(mixed_state["feature_states"]["shortness_of_breath"])
        self.assertFalse(mixed_state["prediction_ready"])

        chest_session = start_pipeline("No chest pain.")
        chest_state = get_current_state(chest_session)
        chest_matches = {
            item["concept_id"]: item["status"]
            for item in chest_state["concept_matches"]
        }
        self.assertEqual(chest_matches["chest_pain"], "NO")
        self.assertIsNone(chest_state["feature_states"]["sharp_chest_pain"])

    def test_temporal_text_never_infers_present_yes_from_history_only(self):
        past_fever = start_pipeline("I had a fever yesterday but don't have one now.")
        past_state = get_current_state(past_fever)
        fever_matches = [
            item["status"]
            for item in past_state["concept_matches"]
            if item["concept_id"] == "fever"
        ]
        self.assertNotIn("YES", fever_matches)
        self.assertIsNone(past_state["feature_states"]["fever"])
        self.assertFalse(past_state["prediction_ready"])

        historical_chest_pain = start_pipeline("I used to have chest pain.")
        historical_state = get_current_state(historical_chest_pain)
        chest_matches = {
            item["concept_id"]: item["status"]
            for item in historical_state["concept_matches"]
        }
        self.assertEqual(chest_matches.get("chest_pain"), "UNKNOWN")
        self.assertFalse(historical_state["prediction_ready"])

        unresolved_pronoun = start_pipeline("I don't have it anymore.")
        pronoun_state = get_current_state(unresolved_pronoun)
        self.assertFalse(pronoun_state["concept_matches"])
        self.assertFalse(pronoun_state["prediction_ready"])
        self.assertTrue(pronoun_state["unresolved_features"])

    def test_third_person_and_report_attribution_are_not_patient_evidence(self):
        cases = (
            ("My brother has a cough.", "cough"),
            ("The doctor said I have fever.", "fever"),
            ("My friend has chest pain.", "chest_pain"),
        )
        for text, concept_id in cases:
            with self.subTest(text=text):
                state = get_current_state(start_pipeline(text))
                self.assertNotIn(
                    concept_id,
                    {item["concept_id"] for item in state["concept_matches"]},
                )
                self.assertFalse(state["prediction_ready"])

    def test_unresolved_contradiction_stays_unknown_and_explicit_correction_applies(
        self,
    ):
        contradiction = start_pipeline("I have fever but I don't have fever.")
        state = get_current_state(contradiction)
        statuses = {
            item["concept_id"]: item["status"] for item in state["concept_matches"]
        }
        self.assertEqual(statuses["fever"], "UNKNOWN")
        self.assertIsNone(state["feature_states"]["fever"])
        self.assertFalse(state["prediction_ready"])

        corrected = start_pipeline("I don't have fever, but actually I do have fever.")
        corrected_state = get_current_state(corrected)
        corrected_statuses = {
            item["concept_id"]: item["status"]
            for item in corrected_state["concept_matches"]
        }
        self.assertEqual(corrected_statuses["fever"], "YES")
        self.assertEqual(corrected_state["feature_states"]["fever"], 1)

    def test_shared_feature_text_cases_preserve_existing_aggregation(self):
        orthopnea = start_pipeline("I have orthopnea.")
        self.assertEqual(
            get_current_state(orthopnea)["feature_states"]["shortness_of_breath"], 1
        )

        exertional_dyspnea = start_pipeline("I have dyspnea on exertion.")
        self.assertEqual(
            get_current_state(exertional_dyspnea)["feature_states"][
                "shortness_of_breath"
            ],
            1,
        )

        breathing_unknown = start_pipeline(
            "I do not have shortness of breath, but I am unsure about orthopnea."
        )
        breathing_state = get_current_state(breathing_unknown)
        self.assertIsNone(breathing_state["feature_states"]["shortness_of_breath"])
        self.assertFalse(breathing_state["prediction_ready"])

        chest_proxy = start_pipeline("I have chest congestion with phlegm.")
        chest_state = get_current_state(chest_proxy)
        self.assertEqual(chest_state["feature_states"]["chest_congestion"], 1)
        self.assertIsNone(chest_state["feature_states"]["productive_cough"])

        direct_productive_cough = start_pipeline("I have productive cough.")
        self.assertEqual(
            get_current_state(direct_productive_cough)["feature_states"][
                "productive_cough"
            ],
            1,
        )

        conflict = start_pipeline(
            "I do not have productive cough, but I have chest congestion with phlegm."
        )
        conflict_state = get_current_state(conflict)
        self.assertIsNone(conflict_state["feature_states"]["productive_cough"])
        self.assertFalse(conflict_state["prediction_ready"])

        unknown_direct = start_pipeline(
            "I am unsure about productive cough, but I have chest congestion with phlegm."
        )
        self.assertIsNone(
            get_current_state(unknown_direct)["feature_states"]["productive_cough"]
        )

        both_negative = start_pipeline(
            "I do not have productive cough and I do not have chest congestion with phlegm."
        )
        self.assertEqual(
            get_current_state(both_negative)["feature_states"]["productive_cough"],
            0,
        )

    def test_clarification_flow_reaches_readiness_only_after_all_required_concepts(
        self,
    ):
        session = start_pipeline("I might have a fever.")
        asked = []
        max_steps = len(TERMINOLOGY) + 1

        for _ in range(max_steps):
            state = get_current_state(session)
            if state["prediction_ready"]:
                break
            question = state["next_question"]
            self.assertIsNotNone(question)
            concept_id = question["concept_id"]
            self.assertNotIn(concept_id, asked)
            asked.append(concept_id)
            answer = "I don't have it anymore" if concept_id == "fever" else "no"
            state = submit_clarification(session, answer)
        else:
            self.fail("Clarification did not terminate within the concept bound")

        final_state = get_current_state(session)
        self.assertTrue(final_state["prediction_ready"])
        self.assertEqual(final_state["unresolved_features"], [])
        self.assertEqual(final_state["feature_states"]["fever"], 0)
        self.assertEqual(len(asked), len(set(asked)))
        self.assertIsNone(final_state["next_question"])

    def test_refusal_or_unknown_during_clarification_keeps_prediction_blocked(self):
        session = start_pipeline("")
        question = get_current_state(session)["next_question"]
        state = submit_clarification(session, "I don't know")

        self.assertEqual(
            state["clarification_history"][-1]["interpreted_status"], "UNKNOWN"
        )
        self.assertNotEqual(
            state["next_question"]["concept_id"], question["concept_id"]
        )
        self.assertFalse(state["prediction_ready"])
        self.assertTrue(state["unresolved_features"])

    def test_feature_contract_is_exact_and_only_binary_values_reach_real_model(self):
        session = start_pipeline(evidence_text("YES"))
        state = get_current_state(session)

        self.assertEqual(len(state["feature_states"]), 26)
        self.assertEqual(list(state["feature_states"]), list(predict.FEATURES))
        self.assertTrue(
            all(
                type(value) is int and value in {0, 1}
                for value in state["feature_states"].values()
            )
        )
        result = run_pipeline_prediction(session)
        self.assertTrue(result)
        self.assertEqual(len(result), len(predict.model.classes_))

    def test_real_end_to_end_prediction_from_non_project_working_directory(self):
        text_literal = repr(evidence_text("YES"))
        source_literal = repr(str(PROJECT_DIR / "src"))
        child_code = (
            "import json, sys; "
            f"sys.path.insert(0, {source_literal}); "
            "from pipeline import start_pipeline, can_predict, predict; "
            f"session = start_pipeline({text_literal}); "
            "assert can_predict(session); "
            "result = predict(session); "
            "print(json.dumps({'ready': can_predict(session), 'features': len(session.clarification.feature_states()), 'predictions': len(result), 'keys': sorted(result[0])}))"
        )
        environment = os.environ.copy()
        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                [sys.executable, "-c", child_code],
                cwd=working_directory,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        child_result = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(child_result["ready"])
        self.assertEqual(child_result["features"], 26)
        self.assertEqual(child_result["predictions"], len(predict.model.classes_))
        self.assertEqual(child_result["keys"], ["disease", "score"])


if __name__ == "__main__":
    unittest.main()
