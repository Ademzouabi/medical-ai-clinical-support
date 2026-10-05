import importlib.util
import os
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from clarification import get_feature_states
from terminology_matcher import ML_FEATURES, TERMINOLOGY
from pipeline import (
    PredictionNotReadyError,
    can_predict,
    get_current_state,
    predict,
    start_pipeline,
    submit_clarification,
)


def complete_evidence_text():
    return " ".join(
        f"I have {concept['preferred_term']}."
        for concept in TERMINOLOGY
        if concept.get("ml_feature_mapping")
    )


def prediction_stub(result=None):
    stub = ModuleType("predict")
    stub.FEATURES = list(ML_FEATURES)
    stub.predict_differential = Mock(return_value=result)
    return stub


class PipelineTests(unittest.TestCase):
    def test_initial_incomplete_text_exposes_unresolved_state_and_blocks_prediction(
        self,
    ):
        session = start_pipeline("I have fever.")
        state = get_current_state(session)

        self.assertEqual(state["original_text"], "I have fever.")
        self.assertEqual(len(state["feature_states"]), 26)
        self.assertGreater(len(state["unresolved_features"]), 0)
        self.assertTrue(state["next_question"])
        self.assertFalse(state["prediction_ready"])
        self.assertFalse(can_predict(session))

    def test_unknown_blocks_prediction_without_loading_prediction_module(self):
        session = start_pipeline("I am unsure about fever.")
        stub = prediction_stub([{"disease": "unused", "score": 0.0}])

        with patch.dict(sys.modules, {"predict": stub}):
            with self.assertRaises(PredictionNotReadyError):
                predict(session)
            stub.predict_differential.assert_not_called()

        self.assertIsNone(session.prediction_result)

    def test_complete_synthetic_evidence_opens_gate(self):
        session = start_pipeline(complete_evidence_text())
        state = get_current_state(session)

        self.assertEqual(len(state["feature_states"]), 26)
        self.assertEqual(state["unresolved_features"], [])
        self.assertTrue(state["prediction_ready"])
        self.assertTrue(can_predict(session))
        self.assertTrue(
            all(
                type(value) is int and value in {0, 1}
                for value in state["feature_states"].values()
            )
        )

    def test_refusal_keeps_unknown_and_blocks_prediction(self):
        session = start_pipeline("")
        question = get_current_state(session)["next_question"]

        state = submit_clarification(session, "I refuse")

        self.assertEqual(
            state["clarification_history"][-1]["concept_id"], question["concept_id"]
        )
        self.assertEqual(
            state["clarification_history"][-1]["interpreted_status"], "UNKNOWN"
        )
        self.assertFalse(state["prediction_ready"])
        self.assertGreater(len(state["unresolved_features"]), 0)

    def test_clarification_updates_only_asked_concept_and_recomputes_features(self):
        session = start_pipeline("I have fever and I am unsure about cough.")
        before_statuses = None

        for _ in range(30):
            state = get_current_state(session)
            question = state["next_question"]
            self.assertIsNotNone(question)
            if question["concept_id"] == "cough":
                before_statuses = {
                    match["concept_id"]: match["status"]
                    for match in state["concept_matches"]
                }
                break
            state = submit_clarification(session, "no")

        self.assertIsNotNone(before_statuses)
        updated = submit_clarification(session, "yes, I do")
        after_statuses = {
            match["concept_id"]: match["status"] for match in updated["concept_matches"]
        }
        changed = {
            concept_id
            for concept_id in set(before_statuses) | set(after_statuses)
            if before_statuses.get(concept_id) != after_statuses.get(concept_id)
        }

        self.assertEqual(changed, {"cough"})
        self.assertEqual(after_statuses["fever"], "YES")
        self.assertEqual(after_statuses["cough"], "YES")
        self.assertEqual(
            updated["feature_states"],
            get_feature_states(updated["concept_matches"]),
        )
        self.assertEqual(updated["feature_states"]["cough"], 1)

    def test_shared_feature_aggregation_and_productive_cough_conflict(self):
        orthopnea = start_pipeline("I have orthopnea.")
        self.assertEqual(
            get_current_state(orthopnea)["feature_states"]["shortness_of_breath"],
            1,
        )

        unresolved_breathing = start_pipeline(
            "I do not have shortness of breath, but I am unsure about orthopnea."
        )
        breathing_state = get_current_state(unresolved_breathing)
        self.assertIsNone(breathing_state["feature_states"]["shortness_of_breath"])
        self.assertFalse(breathing_state["prediction_ready"])

        productive_conflict = start_pipeline(
            "I do not have productive cough, but I have chest congestion with phlegm."
        )
        conflict_state = get_current_state(productive_conflict)
        self.assertIsNone(conflict_state["feature_states"]["productive_cough"])
        self.assertFalse(conflict_state["prediction_ready"])

    def test_prediction_calls_existing_api_with_exact_order_and_caches_result(self):
        session = start_pipeline(complete_evidence_text())
        expected_result = [{"disease": "stub disease", "score": 0.5}]
        stub = prediction_stub(expected_result)

        with patch.dict(sys.modules, {"predict": stub}):
            self.assertEqual(predict(session), expected_result)
            self.assertEqual(predict(session), expected_result)

        stub.predict_differential.assert_called_once()
        feature_vector = stub.predict_differential.call_args.args[0]
        self.assertEqual(list(feature_vector), list(ML_FEATURES))
        self.assertEqual(len(feature_vector), 26)
        self.assertTrue(
            all(
                type(value) is int and value in {0, 1}
                for value in feature_vector.values()
            )
        )
        self.assertEqual(session.prediction_result, expected_result)
        self.assertEqual(
            get_current_state(session)["prediction_result"], expected_result
        )

    def test_feature_contract_is_exactly_26_ordered_values(self):
        session = start_pipeline("I have fever.")
        features = get_current_state(session)["feature_states"]

        self.assertEqual(len(features), 26)
        self.assertEqual(list(features), list(ML_FEATURES))
        self.assertTrue(all(value in {0, 1, None} for value in features.values()))

    def test_predictor_is_not_called_when_any_feature_is_unknown(self):
        session = start_pipeline("I have fever.")
        stub = prediction_stub([])

        with patch.dict(sys.modules, {"predict": stub}):
            with self.assertRaises(PredictionNotReadyError):
                predict(session)

        stub.predict_differential.assert_not_called()

    def test_predict_model_loads_and_runs_outside_project_working_directory(self):
        project_dir = PROJECT_DIR
        predict_path = project_dir / "src" / "predict.py"
        spec = importlib.util.spec_from_file_location(
            "predict_from_external_working_directory",
            predict_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)

        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as external_cwd:
            os.chdir(external_cwd)
            try:
                self.assertNotEqual(Path.cwd().resolve(), project_dir)
                spec.loader.exec_module(module)
            finally:
                os.chdir(original_cwd)

        self.assertEqual(
            module.MODEL_PATH,
            project_dir / "models" / "final_logistic_regression.pkl",
        )
        self.assertEqual(module.model.n_features_in_, 26)
        results = module.predict_differential(
            {feature: 0 for feature in module.FEATURES}
        )
        self.assertEqual(len(results), len(module.model.classes_))


if __name__ == "__main__":
    unittest.main()
