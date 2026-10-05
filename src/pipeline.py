"""V1 orchestration from clinical text through gated prediction."""

from dataclasses import dataclass

from clarification import start_session
from terminology_matcher import ML_FEATURES


class PredictionNotReadyError(RuntimeError):
    """Raised when the complete explicit binary feature vector is unavailable."""


class ClarificationUnavailableError(RuntimeError):
    """Raised when unresolved features remain but no question can be asked."""


@dataclass
class PipelineSession:
    clarification: object
    prediction_result: object = None

    @property
    def original_text(self):
        return self.clarification.original_text


def start_pipeline(text):
    """Start a V1 session from the user's original clinical text."""

    return PipelineSession(clarification=start_session(text))


def _binary_feature_vector(features):
    return (
        len(features) == 26
        and tuple(features) == tuple(ML_FEATURES)
        and all(type(value) is int and value in {0, 1} for value in features.values())
    )


def can_predict(session):
    """Return true only for the complete ordered 26-feature integer vector."""

    return _binary_feature_vector(session.clarification.feature_states())


def get_current_state(session):
    """Expose matcher, feature, clarification, and readiness state."""

    concept_matches = session.clarification.concept_matches()
    feature_states = session.clarification.feature_states()
    unresolved_concepts = session.clarification.unresolved_concepts()

    return {
        "original_text": session.original_text,
        "concept_matches": concept_matches,
        "feature_states": feature_states,
        "unresolved_concepts": [item.concept_id for item in unresolved_concepts],
        "unresolved_features": session.clarification.unresolved_features(),
        "clarification_history": [dict(item) for item in session.clarification.history],
        "next_question": session.clarification.next_question(),
        "prediction_ready": can_predict(session),
        "prediction_result": session.prediction_result,
    }


def submit_clarification(session, answer):
    """Submit an answer to the next selected concept question."""

    if session.prediction_result is not None:
        raise RuntimeError("Prediction has already been produced for this session.")

    question = session.clarification.next_question()
    if question is None:
        raise ClarificationUnavailableError(
            "No clarification question is currently available."
        )

    session.clarification.submit_answer(question["concept_id"], answer)
    return get_current_state(session)


def predict(session):
    """Call the existing predictor only after all 26 features are binary."""

    if session.prediction_result is not None:
        return session.prediction_result

    feature_states = session.clarification.feature_states()
    if not _binary_feature_vector(feature_states):
        raise PredictionNotReadyError(
            "Prediction is blocked until all 26 features are explicit integer 0/1 values."
        )

    # Import lazily so text intake and clarification do not load the model.
    from predict import FEATURES, predict_differential

    if tuple(FEATURES) != tuple(ML_FEATURES):
        raise RuntimeError("Matcher and prediction feature orders do not match.")

    ordered_features = {feature: feature_states[feature] for feature in FEATURES}
    session.prediction_result = predict_differential(ordered_features)
    return session.prediction_result
