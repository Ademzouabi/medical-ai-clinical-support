from itertools import product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clarification import (
    MAX_ATTEMPTS_PER_CONCEPT,
    QUESTION_DEFINITIONS,
    ClarificationSession,
    get_feature_states,
    get_next_question,
    get_unresolved_concepts,
    interpret_answer,
    start_session,
)
from terminology_matcher import TERMINOLOGY, concepts_to_features, match_concepts


class ClarificationTests(unittest.TestCase):
    def test_every_mapped_concept_has_one_question(self):
        mapped = {
            concept["concept_id"]
            for concept in TERMINOLOGY
            if concept.get("ml_feature_mapping")
        }
        prompted = {question.concept_id for question in QUESTION_DEFINITIONS}

        self.assertEqual(prompted, mapped)
        self.assertEqual(len(prompted), len(QUESTION_DEFINITIONS))

    def test_questions_follow_grouped_deterministic_order(self):
        session = start_session("")

        first = session.next_question()
        self.assertEqual(first["group"], "breathing")
        self.assertEqual(first["concept_id"], "shortness_of_breath")

        session.submit_answer(first["concept_id"], "not sure")
        second = session.next_question()
        self.assertEqual(second["concept_id"], "difficulty_breathing")

    def test_positive_shared_feature_skips_other_mapped_concepts(self):
        matches = match_concepts("I have shortness of breath.")
        unresolved = {item.concept_id for item in get_unresolved_concepts(matches)}

        self.assertNotIn("orthopnea", unresolved)
        self.assertNotIn("dyspnea_on_exertion", unresolved)

    def test_proxy_chest_phlegm_leaves_direct_productive_cough_question(self):
        session = start_session("I have chest congestion with phlegm.")

        self.assertEqual(session.feature_states()["chest_congestion"], 1)
        self.assertIsNone(session.feature_states()["productive_cough"])
        unresolved = {item.concept_id for item in session.unresolved_concepts()}
        self.assertIn("productive_cough", unresolved)
        self.assertNotIn("chest_congestion_with_phlegm", unresolved)

    def test_unknown_answer_is_recorded_and_concept_is_not_reasked(self):
        session = start_session("")
        question = session.next_question()
        status = session.submit_answer(question["concept_id"], "I don't know")

        self.assertEqual(status, "UNKNOWN")
        self.assertEqual(session.history[0]["raw_answer"], "I don't know")
        self.assertEqual(session.history[0]["attempt"], 1)
        self.assertNotEqual(
            session.next_question()["concept_id"],
            question["concept_id"],
        )
        self.assertFalse(session.prediction_ready())

        with self.assertRaises(ValueError):
            session.submit_answer(question["concept_id"], "yes")

    def test_refusal_remains_unknown_and_preserves_original_text(self):
        original = "I have a fever and I am unsure about cough."
        session = start_session(original)
        status = session.submit_answer("cough", "I prefer not to answer")

        self.assertEqual(status, "UNKNOWN")
        self.assertEqual(session.original_text, original)
        self.assertEqual(session.history[0]["interpreted_status"], "UNKNOWN")

    def test_answer_is_constrained_to_asked_concept(self):
        session = start_session("I do not have fever.")
        status = session.submit_answer("cough", "I have a fever, but no cough.")
        matches = {
            item["concept_id"]: item["status"] for item in session.concept_matches()
        }

        self.assertEqual(status, "NO")
        self.assertEqual(matches["fever"], "NO")
        self.assertEqual(matches["cough"], "NO")

    def test_answer_about_unasked_concept_does_not_resolve_target(self):
        self.assertEqual(
            interpret_answer("cough", "I have a fever."),
            "UNKNOWN",
        )

    def test_temporal_answers_are_current_state_conservative(self):
        self.assertEqual(
            interpret_answer("fever", "yes, yesterday but not now"),
            "NO",
        )
        self.assertEqual(
            interpret_answer("fever", "I had it last week"),
            "UNKNOWN",
        )
        self.assertEqual(interpret_answer("fever", "I still have it"), "YES")

    def test_prediction_gate_rejects_unresolved_features(self):
        session = start_session("")

        self.assertEqual(len(session.unresolved_features()), 26)
        self.assertFalse(session.prediction_ready())

    def test_prediction_gate_opens_for_complete_binary_evidence(self):
        text = " ".join(
            f"I have {concept['preferred_term']}."
            for concept in TERMINOLOGY
            if concept.get("ml_feature_mapping")
        )
        session = start_session(text)

        self.assertTrue(session.prediction_ready())
        self.assertEqual(session.unresolved_features(), [])

    def test_unresolved_feature_helper_uses_existing_aggregation(self):
        matches = [
            {"concept_id": "shortness_of_breath", "status": "NO"},
            {"concept_id": "orthopnea", "status": "UNKNOWN"},
            {"concept_id": "dyspnea_on_exertion", "status": "NO"},
        ]
        features, _ = concepts_to_features(matches)

        self.assertIsNone(features["shortness_of_breath"])
        self.assertIn(
            "shortness_of_breath",
            [feature for feature, value in features.items() if value is None],
        )

    def test_next_question_is_concept_level(self):
        question = get_next_question([], history=())

        self.assertEqual(question.concept_id, "shortness_of_breath")
        self.assertEqual(question.target_concept, "shortness_of_breath")
        self.assertIn("currently", question.question)


class ClarificationAdversarialTests(unittest.TestCase):
    def test_answer_interpretation_is_scoped_to_the_asked_concept(self):
        self.assertEqual(
            interpret_answer("fever", "No cough, but yes fever."),
            "YES",
        )
        self.assertEqual(
            interpret_answer("cough", "I don't have cough, but I have fever."),
            "NO",
        )
        self.assertEqual(
            interpret_answer("wheezing", "My brother is wheezing, I'm not."),
            "UNKNOWN",
        )
        self.assertEqual(
            interpret_answer("fever", "My brother has fever."),
            "UNKNOWN",
        )

    def test_answer_updates_only_the_asked_concept(self):
        session = start_session("")
        status = session.submit_answer("fever", "No cough, but yes fever.")
        matches = {
            item["concept_id"]: item["status"] for item in session.concept_matches()
        }

        self.assertEqual(status, "YES")
        self.assertEqual(matches, {"fever": "YES"})

        session = start_session("")
        status = session.submit_answer(
            "cough",
            "I don't have cough, but I have fever.",
        )
        matches = {
            item["concept_id"]: item["status"] for item in session.concept_matches()
        }

        self.assertEqual(status, "NO")
        self.assertEqual(matches, {"cough": "NO"})

    def test_explicit_answer_vocabulary_is_conservative(self):
        answers = {
            "no": "NO",
            "nope": "NO",
            "no, I don't": "NO",
            "not currently": "NO",
            "yes": "YES",
            "yeah": "YES",
            "yes I do": "YES",
            "definitely": "YES",
            "currently": "YES",
            "I don't think so": "UNKNOWN",
            "I think not": "UNKNOWN",
            "maybe": "UNKNOWN",
            "probably": "UNKNOWN",
            "not sure": "UNKNOWN",
            "I don't know": "UNKNOWN",
            "I'm unsure": "UNKNOWN",
        }

        for answer, expected in answers.items():
            with self.subTest(answer=answer):
                self.assertEqual(interpret_answer("fever", answer), expected)

    def test_current_state_temporal_answers(self):
        answers = {
            "I had it last week": "UNKNOWN",
            "I had it yesterday": "UNKNOWN",
            "I had it yesterday but not now": "NO",
            "I had it yesterday and I still have it": "YES",
            "I used to have it": "UNKNOWN",
            "I don't have it anymore": "NO",
            "I currently have it": "YES",
            "I still have it": "YES",
            "I don't have it currently": "NO",
            "currently, I don't have it": "NO",
        }

        for answer, expected in answers.items():
            with self.subTest(answer=answer):
                self.assertEqual(interpret_answer("fever", answer), expected)

    def test_contradiction_and_correction_answers(self):
        answers = {
            "yes, but actually no": "NO",
            "no, actually yes": "YES",
            "I have it but I don't have it": "UNKNOWN",
            "I don't have it, but actually I do": "YES",
            "I had it, but I still have it now": "YES",
            "No... wait, yes.": "YES",
        }

        for answer, expected in answers.items():
            with self.subTest(answer=answer):
                self.assertEqual(interpret_answer("fever", answer), expected)

    def test_long_free_text_is_limited_to_the_asked_concept(self):
        cases = (
            (
                "fever",
                "I have fever and cough, but no wheezing.",
                "YES",
            ),
            (
                "cough",
                "I don't have fever. I do have cough.",
                "YES",
            ),
            (
                "wheezing",
                "My chest hurts and I'm coughing, but I'm not sure about wheezing.",
                "UNKNOWN",
            ),
        )

        for concept_id, answer, expected in cases:
            with self.subTest(concept_id=concept_id, answer=answer):
                self.assertEqual(interpret_answer(concept_id, answer), expected)

    def test_breathing_shared_feature_aggregation_and_question_selection(self):
        concepts = (
            "shortness_of_breath",
            "orthopnea",
            "dyspnea_on_exertion",
        )
        for statuses in product(("YES", "NO", "UNKNOWN"), repeat=3):
            matches = [
                {"concept_id": concept_id, "status": status}
                for concept_id, status in zip(concepts, statuses)
            ]
            feature = get_feature_states(matches)["shortness_of_breath"]
            expected = (
                1
                if "YES" in statuses
                else 0
                if all(status == "NO" for status in statuses)
                else None
            )
            with self.subTest(statuses=statuses):
                self.assertEqual(feature, expected)

        yes_matches = [{"concept_id": "shortness_of_breath", "status": "YES"}]
        unresolved = {item.concept_id for item in get_unresolved_concepts(yes_matches)}
        self.assertNotIn("orthopnea", unresolved)
        self.assertNotIn("dyspnea_on_exertion", unresolved)

        unknown_matches = [{"concept_id": "shortness_of_breath", "status": "UNKNOWN"}]
        unresolved = {
            item.concept_id for item in get_unresolved_concepts(unknown_matches)
        }
        self.assertIn("orthopnea", unresolved)
        self.assertIn("dyspnea_on_exertion", unresolved)

    def test_chest_congestion_shared_feature_aggregation(self):
        concepts = ("chest_congestion", "chest_congestion_with_phlegm")
        for statuses in product(("YES", "NO", "UNKNOWN"), repeat=2):
            matches = [
                {"concept_id": concept_id, "status": status}
                for concept_id, status in zip(concepts, statuses)
            ]
            feature = get_feature_states(matches)["chest_congestion"]
            expected = (
                1
                if "YES" in statuses
                else 0
                if all(status == "NO" for status in statuses)
                else None
            )
            with self.subTest(statuses=statuses):
                self.assertEqual(feature, expected)

    def test_productive_cough_direct_and_proxy_aggregation(self):
        concepts = ("productive_cough", "chest_congestion_with_phlegm")
        for direct_status, proxy_status in product(("YES", "NO", "UNKNOWN"), repeat=2):
            matches = [
                {"concept_id": concept_id, "status": status}
                for concept_id, status in zip(
                    concepts,
                    (direct_status, proxy_status),
                )
            ]
            feature = get_feature_states(matches)["productive_cough"]
            expected = (
                1
                if direct_status == "YES"
                else 0
                if direct_status == "NO" and proxy_status == "NO"
                else None
            )
            with self.subTest(direct=direct_status, proxy=proxy_status):
                self.assertEqual(feature, expected)

    def test_proxy_alone_never_sets_productive_cough_yes(self):
        session = start_session("I have chest congestion with phlegm.")

        self.assertEqual(session.feature_states()["chest_congestion"], 1)
        self.assertIsNone(session.feature_states()["productive_cough"])
        self.assertFalse(session.prediction_ready())

    def test_unresolved_shared_mapping_blocks_prediction(self):
        for matches in (
            [
                {"concept_id": "shortness_of_breath", "status": "NO"},
                {"concept_id": "orthopnea", "status": "UNKNOWN"},
                {"concept_id": "dyspnea_on_exertion", "status": "NO"},
            ],
            [
                {"concept_id": "productive_cough", "status": "NO"},
                {
                    "concept_id": "chest_congestion_with_phlegm",
                    "status": "YES",
                },
            ],
        ):
            with self.subTest(matches=matches):
                features, _ = concepts_to_features(matches)
                session = ClarificationSession(original_text="")
                session.concept_matches = lambda: matches
                session.feature_states = lambda: features
                self.assertFalse(session.prediction_ready())

    def test_attempt_limit_is_finite_even_when_answers_stay_unknown(self):
        session = start_session("")
        seen = set()

        for _ in range(len(QUESTION_DEFINITIONS) + 1):
            question = session.next_question()
            if question is None:
                break
            concept_id = question["concept_id"]
            self.assertNotIn(concept_id, seen)
            seen.add(concept_id)
            self.assertEqual(
                session.submit_answer(concept_id, "probably"),
                "UNKNOWN",
            )

        self.assertLessEqual(len(session.history), len(QUESTION_DEFINITIONS))
        self.assertEqual(MAX_ATTEMPTS_PER_CONCEPT, 1)
        self.assertIsNone(session.next_question())
        self.assertFalse(session.prediction_ready())

    def test_refusal_leaves_prediction_blocked(self):
        session = start_session("")
        question = session.next_question()

        self.assertEqual(
            session.submit_answer(question["concept_id"], "I refuse"),
            "UNKNOWN",
        )
        self.assertFalse(session.prediction_ready())

    def test_provenance_and_original_text_are_preserved(self):
        original = "I have fever and I am unsure about cough."
        session = start_session(original)
        session.submit_answer("cough", "yes, I do")

        self.assertEqual(session.original_text, original)
        self.assertEqual(
            session.history[0],
            {
                "concept_id": "cough",
                "question": next(
                    item.question
                    for item in QUESTION_DEFINITIONS
                    if item.concept_id == "cough"
                ),
                "raw_answer": "yes, I do",
                "interpreted_status": "YES",
                "attempt": 1,
            },
        )

    def test_features_are_recomputed_through_existing_aggregation(self):
        session = start_session("I do not have fever.")
        session.submit_answer("cough", "yes")

        expected, _ = concepts_to_features(session.concept_matches())
        self.assertEqual(session.feature_states(), expected)
        self.assertEqual(session.feature_states()["fever"], 0)
        self.assertEqual(session.feature_states()["cough"], 1)

    def test_question_groups_keep_required_order(self):
        groups = []
        for definition in QUESTION_DEFINITIONS:
            if definition.group not in groups:
                groups.append(definition.group)

        self.assertEqual(
            groups,
            ["breathing", "cough_chest", "ent", "cardiovascular", "systemic"],
        )


if __name__ == "__main__":
    unittest.main()
