"""Concept-level clarification before binary ML prediction."""

from dataclasses import dataclass, field
import re

from terminology_matcher import (
    TERMINOLOGY,
    concepts_to_features,
    find_concepts,
    get_feature_mapping,
    is_doctor_report,
    is_other_person_statement,
    match_concepts,
    normalize_fragment,
    normalize_text,
)


MAX_ATTEMPTS_PER_CONCEPT = 1


@dataclass(frozen=True)
class QuestionDefinition:
    concept_id: str
    question: str
    target_concept: str
    group: str


_QUESTION_DATA = (
    (
        "breathing",
        "shortness_of_breath",
        "Do you currently feel short of breath or have difficulty getting enough air?",
    ),
    (
        "breathing",
        "difficulty_breathing",
        "Do you currently have difficulty breathing or need increased effort to breathe?",
    ),
    (
        "breathing",
        "orthopnea",
        "Do you currently become short of breath or have difficulty breathing when lying flat?",
    ),
    (
        "breathing",
        "dyspnea_on_exertion",
        "Do you currently become short of breath with physical activity, such as walking?",
    ),
    (
        "breathing",
        "pain_with_breathing",
        "Do you currently feel pain that occurs or worsens when you breathe?",
    ),
    (
        "breathing",
        "wheezing",
        "Do you currently wheeze or hear a whistling sound when breathing?",
    ),
    (
        "breathing",
        "apnea",
        "Are you currently experiencing pauses or episodes where your breathing stops?",
    ),
    ("cough_chest", "cough", "Are you currently coughing?"),
    (
        "cough_chest",
        "productive_cough",
        "Are you currently coughing up mucus, phlegm, or sputum?",
    ),
    (
        "cough_chest",
        "chest_congestion",
        "Do you currently feel mucus or congestion in your chest?",
    ),
    (
        "cough_chest",
        "chest_congestion_with_phlegm",
        "Do you currently feel mucus or phlegm accumulating in your chest?",
    ),
    (
        "cough_chest",
        "chest_tightness",
        "Do you currently feel tightness, pressure, or squeezing in your chest?",
    ),
    (
        "cough_chest",
        "hemoptysis",
        "Are you currently coughing up blood or seeing blood in your sputum?",
    ),
    (
        "cough_chest",
        "sharp_chest_pain",
        "Do you currently have sharp, stabbing, or piercing chest pain?",
    ),
    (
        "ent",
        "hoarseness",
        "Is your voice currently hoarse, raspy, or unusually changed?",
    ),
    (
        "ent",
        "sore_throat",
        "Do you currently have a sore, painful, scratchy, or irritated throat?",
    ),
    ("ent", "nasal_congestion", "Is your nose currently stuffy or blocked?"),
    (
        "ent",
        "sinus_congestion",
        "Do you currently feel sinus pressure, fullness, or blocked sinuses?",
    ),
    ("ent", "coryza", "Do you currently have a runny nose or nasal discharge?"),
    (
        "cardiovascular",
        "palpitations",
        "Are you currently unusually aware of your heartbeat, such as pounding, fluttering, or racing?",
    ),
    (
        "cardiovascular",
        "irregular_heartbeat",
        "Does your heartbeat currently feel irregular, uneven, or like it skips beats?",
    ),
    (
        "cardiovascular",
        "increased_heart_rate",
        "Have you been told or noticed that your heart rate is currently faster than usual?",
    ),
    (
        "cardiovascular",
        "dizziness",
        "Are you currently dizzy, lightheaded, unsteady, or experiencing a spinning sensation?",
    ),
    (
        "systemic",
        "fever",
        "Do you currently have a fever or an abnormally high temperature?",
    ),
    ("systemic", "chills", "Are you currently experiencing chills or shivering?"),
    ("systemic", "fatigue", "Are you currently unusually tired or lacking energy?"),
    ("systemic", "malaise", "Do you currently feel generally unwell?"),
    (
        "systemic",
        "general_weakness",
        "Do you currently feel generally or physically weak?",
    ),
    ("systemic", "vomiting", "Are you currently vomiting or throwing up?"),
)

_TERMINOLOGY_BY_ID = {concept["concept_id"]: concept for concept in TERMINOLOGY}

QUESTION_DEFINITIONS = tuple(
    QuestionDefinition(
        concept_id=concept_id,
        question=question,
        target_concept=concept_id,
        group=group,
    )
    for group, concept_id, question in _QUESTION_DATA
    if concept_id in _TERMINOLOGY_BY_ID
    and _TERMINOLOGY_BY_ID[concept_id].get("ml_feature_mapping")
)

_QUESTION_BY_ID = {
    definition.concept_id: definition for definition in QUESTION_DEFINITIONS
}

_MAPPED_CONCEPT_IDS = tuple(
    concept["concept_id"]
    for concept in TERMINOLOGY
    if concept.get("ml_feature_mapping")
)

_YES_ANSWERS = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "yes i do",
    "yes i am",
    "i do",
    "definitely",
    "currently",
}

_NO_ANSWERS = {
    "no",
    "nope",
    "no i do not",
    "no i am not",
    "not currently",
    "not now",
}

_UNKNOWN_ANSWERS = {
    "not sure",
    "i am not sure",
    "i do not know",
    "unsure",
    "i am unsure",
    "maybe",
    "i think so",
    "i think not",
    "i do not think so",
    "i am uncertain",
    "uncertain",
    "pass",
    "skip",
    "prefer not to answer",
    "rather not answer",
    "i do not want to answer",
    "i would rather not say",
    "i refuse",
}

_PAST_ONLY_PATTERNS = (
    r"\byesterday\b",
    r"\blast week\b",
    r"\blast month\b",
    r"\bearlier\b",
    r"\bpreviously\b",
    r"\bin the past\b",
    r"\bused to\b",
)


def _normalized_answer(answer):
    return normalize_fragment(answer)


def _corrected_polarity(normalized):
    if not re.search(r"\b(?:actually|wait|correction)\b", normalized):
        return None

    polarity_patterns = (
        (
            "YES",
            r"\b(?:yes|yeah|yep|yup|definitely)\b|\bi do(?! not)\b|\bi am(?! not)\b",
        ),
        ("NO", r"\b(?:no|nope|do not|am not|is not|are not|cannot|no longer)\b"),
    )
    mentions = []
    for status, pattern in polarity_patterns:
        mentions.extend(
            (match.start(), status) for match in re.finditer(pattern, normalized)
        )

    if {status for _, status in mentions} != {"YES", "NO"}:
        return None

    return max(mentions)[1]


def _target_segment_status(concept_id, raw_answer):
    chunks = re.split(
        r"\s+\b(?:but|however|though)\b\s+|[.!?]+",
        normalize_text(raw_answer),
        flags=re.IGNORECASE,
    )
    statuses = []

    for chunk in chunks:
        if concept_id not in find_concepts(chunk):
            continue
        if is_other_person_statement(chunk) or is_doctor_report(chunk):
            continue

        chunk_status = next(
            (
                match["status"]
                for match in match_concepts(chunk)
                if match["concept_id"] == concept_id
            ),
            "UNKNOWN",
        )
        if chunk_status in {"YES", "NO"}:
            statuses.append(chunk_status)
            continue

        if re.match(r"\s*(?:yes|yeah|yep|yup|definitely)\b", chunk):
            statuses.append("YES")

    if not statuses:
        return None
    if len(set(statuses)) == 1:
        return statuses[0]
    return "UNKNOWN"


def interpret_answer(concept_id, raw_answer):
    """Interpret an answer for only the concept named by the question."""

    if concept_id not in _QUESTION_BY_ID:
        raise ValueError(f"No clarification question for concept: {concept_id}")

    normalized = _normalized_answer(raw_answer)

    if not normalized:
        return "UNKNOWN"

    if normalized in _YES_ANSWERS:
        return "YES"

    if normalized in _NO_ANSWERS:
        return "NO"

    if normalized in _UNKNOWN_ANSWERS:
        return "UNKNOWN"

    if re.search(
        r"\b(?:not sure|unsure|uncertain|maybe|do not know|do not think)\b",
        normalized,
    ):
        return "UNKNOWN"

    has_past_reference = any(
        re.search(pattern, normalized) for pattern in _PAST_ONLY_PATTERNS
    )
    has_current_denial = bool(
        re.search(
            r"\b(?:not now|not currently|no longer|do not currently|do not have it now)\b"
            r"|\b(?:do not|am not|is not|are not|cannot)\b.*\b(?:now|currently|today)\b"
            r"|\b(?:do not|am not|is not|are not|cannot)\b.*\banymore\b"
            r"|\b(?:now|currently|today)\b.*\b(?:do not|am not|is not|are not|cannot)\b",
            normalized,
        )
    )
    if has_current_denial:
        return "NO"

    if has_past_reference and not re.search(
        r"\b(?:now|currently|still|today)\b",
        normalized,
    ):
        return "UNKNOWN"

    if re.search(r"\b(?:still have it|still experiencing it)\b", normalized):
        return "YES"

    if re.search(r"\bcurrently\b", normalized) and re.search(
        r"\b(?:have|has|feel|feeling|experience|experiencing|am)\b", normalized
    ):
        return "YES"

    # Reuse the matcher, but only accept evidence for the asked concept.
    answer_matches = match_concepts(raw_answer)
    for match in answer_matches:
        if match["concept_id"] == concept_id and match["status"] in {"YES", "NO"}:
            return match["status"]

    target_status = _target_segment_status(concept_id, raw_answer)
    if target_status is not None:
        return target_status

    corrected_status = _corrected_polarity(normalized)
    if corrected_status is not None:
        return corrected_status

    return "UNKNOWN"


def get_feature_states(concept_matches):
    """Recompute feature states through the existing aggregation layer."""

    features, _ = concepts_to_features(concept_matches)
    return features


def get_unresolved_features(concept_matches):
    """Return feature names that are not yet explicit binary values."""

    features = get_feature_states(concept_matches)
    return [feature for feature, value in features.items() if value is None]


def get_unresolved_concepts(concept_matches):
    """Return unresolved, askable concepts relevant to UNKNOWN features."""

    feature_states = get_feature_states(concept_matches)
    unresolved_features = {
        feature for feature, value in feature_states.items() if value is None
    }
    statuses = {
        match["concept_id"]: match["status"]
        for match in concept_matches
        if match.get("concept_id") in _MAPPED_CONCEPT_IDS
    }

    unresolved = []
    for definition in QUESTION_DEFINITIONS:
        concept_id = definition.concept_id
        mapped_features = get_feature_mapping(concept_id)

        if statuses.get(concept_id, "UNKNOWN") != "UNKNOWN":
            continue
        if not unresolved_features.intersection(mapped_features):
            continue

        unresolved.append(definition)

    return unresolved


def get_next_question(concept_matches, history=()):
    """Select the first eligible question in deterministic group order."""

    attempts = {}
    for entry in history:
        concept_id = entry.get("concept_id")
        attempts[concept_id] = attempts.get(concept_id, 0) + 1

    for definition in get_unresolved_concepts(concept_matches):
        if attempts.get(definition.concept_id, 0) >= MAX_ATTEMPTS_PER_CONCEPT:
            continue
        return definition

    return None


def _answer_evidence(concept_id, status):
    term = _TERMINOLOGY_BY_ID[concept_id]["preferred_term"].lower()
    if status == "YES":
        return f"I have {term} currently."
    if status == "NO":
        return f"I do not have {term} now."
    return None


@dataclass
class ClarificationSession:
    """In-memory state for one clarification flow; original text is retained."""

    original_text: str
    history: list = field(default_factory=list)
    _answer_statements: dict = field(default_factory=dict, repr=False)
    _concept_matches_cache: list = field(default=None, init=False, repr=False)

    def concept_matches(self):
        if self._concept_matches_cache is not None:
            return [dict(match) for match in self._concept_matches_cache]

        original_matches = match_concepts(self.original_text)
        overrides = {}
        if self._answer_statements:
            combined_text = ". ".join(
                [self.original_text]
                + [
                    statement
                    for statements in self._answer_statements.values()
                    for statement in statements
                ]
            ).strip(" .")
            answered_matches = match_concepts(combined_text)
            overrides = {
                match["concept_id"]: match["status"]
                for match in answered_matches
                if match["concept_id"] in self._answer_statements
            }

        results = [
            match for match in original_matches if match["concept_id"] not in overrides
        ]
        results.extend(
            {"concept_id": concept_id, "status": status}
            for concept_id, status in overrides.items()
        )
        self._concept_matches_cache = results
        return [dict(match) for match in results]

    def feature_states(self):
        return get_feature_states(self.concept_matches())

    def unresolved_features(self):
        return get_unresolved_features(self.concept_matches())

    def unresolved_concepts(self):
        return get_unresolved_concepts(self.concept_matches())

    def next_question(self):
        definition = get_next_question(self.concept_matches(), self.history)
        if definition is None:
            return None

        attempts = sum(
            entry["concept_id"] == definition.concept_id for entry in self.history
        )
        return {
            "concept_id": definition.concept_id,
            "question": definition.question,
            "target_concept": definition.target_concept,
            "group": definition.group,
            "attempt": attempts + 1,
        }

    def submit_answer(self, concept_id, raw_answer):
        if concept_id not in _QUESTION_BY_ID:
            raise ValueError(f"No clarification question for concept: {concept_id}")

        attempt = 1 + sum(entry["concept_id"] == concept_id for entry in self.history)
        if attempt > MAX_ATTEMPTS_PER_CONCEPT:
            raise ValueError(f"Clarification attempt limit reached: {concept_id}")

        status = interpret_answer(concept_id, raw_answer)
        definition = _QUESTION_BY_ID[concept_id]
        self.history.append(
            {
                "concept_id": concept_id,
                "question": definition.question,
                "raw_answer": raw_answer,
                "interpreted_status": status,
                "attempt": attempt,
            }
        )

        statement = _answer_evidence(concept_id, status)
        if statement:
            self._answer_statements.setdefault(concept_id, []).append(statement)
        self._concept_matches_cache = None

        return status

    def prediction_ready(self):
        return all(value in (0, 1) for value in self.feature_states().values())


def start_session(original_text):
    """Create a clarification session while retaining the original user text."""

    return ClarificationSession(original_text=original_text or "")
