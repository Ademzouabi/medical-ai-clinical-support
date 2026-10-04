import json
import re
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TERMINOLOGY_FILE = BASE_DIR / "terminology_v1.json"


ML_FEATURES = [
    "shortness_of_breath",
    "difficulty_breathing",
    "pain_with_breathing",
    "cough",
    "productive_cough",
    "wheezing",
    "chest_tightness",
    "chest_congestion",
    "hemoptysis",
    "apnea",
    "hoarseness",
    "sore_throat",
    "nasal_congestion",
    "sinus_congestion",
    "coryza",
    "sharp_chest_pain",
    "palpitations",
    "irregular_heartbeat",
    "increased_heart_rate",
    "dizziness",
    "fever",
    "chills",
    "fatigue",
    "malaise",
    "general_weakness",
    "vomiting",
]


# ============================================================
# Negation / uncertainty patterns
# ============================================================

NEGATION_PATTERNS = [

    # Basic negation
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bnever\b",
    r"\bnone\b",

    # Clinical/documentation negation
    r"\bdeny\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bdenying\b",

    # Explicit absence
    r"\bdoes not have\b",
    r"\bdo not have\b",
    r"\bdid not have\b",
    r"\bdoes not experience\b",
    r"\bdo not experience\b",
    r"\bdoes not feel\b",
    r"\bdo not feel\b",

    # Common normalized contractions
    r"\bdoesnt have\b",
    r"\bdont have\b",
    r"\bdidnt have\b",
    r"\bcant have\b",
    r"\bcannot have\b",
    r"\bwont have\b",
    r"\bwouldnt have\b",

    r"\bdoesnt experience\b",
    r"\bdont experience\b",
    r"\bdoesnt feel\b",
    r"\bdont feel\b",

    r"\bdoesnt\b",
    r"\bdont\b",
    r"\bdidnt\b",
]


UNCERTAINTY_PATTERNS = [
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bnot sure\b",
    r"\bunsure\b",
    r"\buncertain\b",
    r"\bi wonder if\b",
    r"\bwonder if\b",
    r"\bcould be\b",
    r"\bmight have\b",
    r"\bmay have\b",
    r"\bthink i have\b",
    r"\bthink i might have\b",
    r"\bthink i may have\b",
]


# ============================================================
# Load terminology
# ============================================================

def load_terminology():

    if not TERMINOLOGY_FILE.exists():

        raise FileNotFoundError(
            f"Terminology file not found: {TERMINOLOGY_FILE}"
        )

    with open(
        TERMINOLOGY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        terminology = json.load(file)

    if not isinstance(terminology, list):

        raise ValueError(
            "terminology_v1.json must contain a JSON list."
        )

    return terminology


TERMINOLOGY = load_terminology()


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    """
    Normalize text while preserving the meaning needed
    for terminology, negation and uncertainty detection.
    """

    if not isinstance(text, str):

        raise TypeError(
            "Input text must be a string."
        )

    text = text.lower().strip()

    # Normalize apostrophes.
    text = text.replace("’", "'")

    # Expand common contractions BEFORE removing punctuation.
    #
    # This prevents:
    #
    # "don't" -> "don t"
    #
    # and instead produces:
    #
    # "don't" -> "do not"

    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "cannot",
        "couldn't": "could not",
        "won't": "will not",
        "wouldn't": "would not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",

        "i'm": "i am",
        "i've": "i have",
        "i'd": "i would",
        "i'll": "i will",

        "you're": "you are",
        "you've": "you have",
        "you'd": "you would",
        "you'll": "you will",

        "he's": "he is",
        "she's": "she is",
        "it's": "it is",

        "we're": "we are",
        "we've": "we have",
        "we'd": "we would",

        "they're": "they are",
        "they've": "they have",
        "they'd": "they would",
    }

    for contraction, replacement in contractions.items():

        text = text.replace(
            contraction,
            replacement
        )

    # Replace punctuation with spaces.
    text = re.sub(
        r"[^\w\s-]",
        " ",
        text
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Build terminology index
# ============================================================

def build_phrase_index(terminology):
    """
    Create:

        normalized phrase -> concept IDs
    """

    index = {}

    for concept in terminology:

        concept_id = concept["concept_id"]

        phrases = []

        preferred_term = concept.get(
            "preferred_term"
        )

        if preferred_term:

            phrases.append(
                preferred_term
            )

        phrases.extend(
            concept.get(
                "synonyms",
                []
            )
        )

        for phrase in phrases:

            normalized_phrase = normalize_text(
                phrase
            )

            if not normalized_phrase:
                continue

            index.setdefault(
                normalized_phrase,
                []
            )

            if concept_id not in index[
                normalized_phrase
            ]:

                index[
                    normalized_phrase
                ].append(
                    concept_id
                )

    return index


PHRASE_INDEX = build_phrase_index(
    TERMINOLOGY
)


# ============================================================
# Concept lookup
# ============================================================

def get_concept(concept_id):

    for concept in TERMINOLOGY:

        if concept["concept_id"] == concept_id:

            return concept

    return None


# ============================================================
# Negation detection
# ============================================================




# ============================================================
# Uncertainty detection
# ============================================================

def get_local_context(text, phrase_start):
    """
    Get the local text before a matched phrase.

    Stops at strong clause boundaries so negation from one
    clause does not incorrectly affect another concept.
    """

    before_phrase = text[:phrase_start]

    boundaries = [
        " but ",
        " however ",
        " although ",
        " though ",
        " whereas ",
        "and",
    ]

    latest_boundary = -1
    latest_boundary_length = 0

    for boundary in boundaries:
        position = before_phrase.rfind(boundary)

        if position > latest_boundary:
            latest_boundary = position
            latest_boundary_length = len(boundary)

    if latest_boundary >= 0:
        before_phrase = before_phrase[
            latest_boundary + latest_boundary_length:
        ]

    words = before_phrase.split()

    return " ".join(words[-10:])


def contains_negation(text, phrase_start):
    """
    Determine whether the matched phrase is negated.
    """

    context = get_local_context(
        text,
        phrase_start
    )

    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, context):
            return True

    return False



def contains_uncertainty(
    text,
    phrase_start
):
    """
    Determine whether a matched phrase is uncertain.
    """

    context = get_local_context(
        text,
        phrase_start
    )

    for pattern in UNCERTAINTY_PATTERNS:

        if re.search(
            pattern,
            context
        ):

            return True

    return False


# ============================================================
# Match concepts
# ============================================================

def match_concepts(text):
    """
    Find concepts explicitly mentioned in the text.

    Each match receives:
        YES
        NO
        UNKNOWN

    based on surrounding language.
    """

    normalized_text = normalize_text(text)

    matches = {}

    for phrase, concept_ids in PHRASE_INDEX.items():

        pattern = (
            r"(?<!\w)"
            + re.escape(phrase)
            + r"(?!\w)"
        )

        for match in re.finditer(
            pattern,
            normalized_text
        ):

            start = match.start()

            # Check uncertainty first.
            is_uncertain = contains_uncertainty(
                normalized_text,
                start
            )

            # Then check negation.
            is_negated = contains_negation(
                normalized_text,
                start
            )

            if is_uncertain:
                status = "UNKNOWN"

            elif is_negated:
                status = "NO"

            else:
                status = "YES"

            for concept_id in concept_ids:

                concept = get_concept(
                    concept_id
                )

                if concept is None:
                    continue

                if concept_id not in matches:

                    matches[concept_id] = {
                        "concept_id": concept_id,
                        "preferred_term": concept[
                            "preferred_term"
                        ],
                        "matched_phrase": phrase,
                        "definition": concept.get(
                            "definition",
                            ""
                        ),
                        "status": status
                    }

                else:

                    existing = matches[
                        concept_id
                    ]

                    priority = {
                        "NO": 1,
                        "UNKNOWN": 2,
                        "YES": 3
                    }

                    if priority[status] > priority[
                        existing["status"]
                    ]:

                        existing["status"] = status
                        existing["matched_phrase"] = phrase

    return list(matches.values())

# ============================================================
# Convert concepts -> ML feature states
# ============================================================

def concepts_to_features(matches):
    """
    Convert terminology concepts into feature states.

    States:

        1      = YES
        0      = NO
        None   = UNKNOWN

    Missing information stays UNKNOWN.
    """

    features = {
        feature: None
        for feature in ML_FEATURES
    }

    feature_sources = {
        feature: []
        for feature in ML_FEATURES
    }

    for match in matches:

        concept = get_concept(
            match["concept_id"]
        )

        if concept is None:
            continue

        mapping = concept.get(
            "ml_feature_mapping",
            {}
        )

        status = match[
            "status"
        ]

        for feature, value in mapping.items():

            if feature not in features:
                continue

            if value != 1:
                continue

            # -----------------------------------------------
            # Explicitly present.
            # -----------------------------------------------

            if status == "YES":

                features[
                    feature
                ] = 1

                feature_sources[
                    feature
                ].append(
                    {
                        "concept":
                            match[
                                "concept_id"
                            ],

                        "status":
                            "YES"
                    }
                )

            # -----------------------------------------------
            # Explicitly absent.
            # -----------------------------------------------

            elif status == "NO":

                # Never overwrite an explicit YES.
                if features[
                    feature
                ] != 1:

                    features[
                        feature
                    ] = 0

                feature_sources[
                    feature
                ].append(
                    {
                        "concept":
                            match[
                                "concept_id"
                            ],

                        "status":
                            "NO"
                    }
                )

            # -----------------------------------------------
            # Uncertain.
            # -----------------------------------------------

            elif status == "UNKNOWN":

                # UNKNOWN does not overwrite
                # an already known YES or NO.
                if features[
                    feature
                ] is None:

                    features[
                        feature
                    ] = None

                feature_sources[
                    feature
                ].append(
                    {
                        "concept":
                            match[
                                "concept_id"
                            ],

                        "status":
                            "UNKNOWN"
                    }
                )

    return (
        features,
        feature_sources
    )


# ============================================================
# Full normalization pipeline
# ============================================================

def normalize_clinical_text(text):
    """
    Complete pipeline:

        Raw text
            ↓
        Text normalization
            ↓
        Concept matching
            ↓
        YES / NO / UNKNOWN
            ↓
        ML feature representation
    """

    matches = match_concepts(
        text
    )

    features, feature_sources = (
        concepts_to_features(
            matches
        )
    )

    return {

        "input_text":
            text,

        "normalized_text":
            normalize_text(
                text
            ),

        "matched_concepts":
            matches,

        "features":
            features,

        "feature_sources":
            feature_sources
    }


# ============================================================
# Pretty printing
# ============================================================

def print_result(result):

    print(
        "\n" + "=" * 60
    )

    print(
        "INPUT"
    )

    print(
        "=" * 60
    )

    print(
        result["input_text"]
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "NORMALIZED TEXT"
    )

    print(
        "=" * 60
    )

    print(
        result["normalized_text"]
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "MATCHED CONCEPTS"
    )

    print(
        "=" * 60
    )

    if result[
        "matched_concepts"
    ]:

        for match in result[
            "matched_concepts"
        ]:

            print(
                f"- {match['concept_id']}"
                f" | matched: "
                f"\"{match['matched_phrase']}\""
                f" | status: "
                f"{match['status']}"
                f" | term: "
                f"{match['preferred_term']}"
            )

    else:

        print(
            "No concepts matched."
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "ML FEATURES"
    )

    print(
        "=" * 60
    )

    found_feature = False

    for feature, value in result[
        "features"
    ].items():

        if value is not None:

            found_feature = True

            sources = result[
                "feature_sources"
            ][feature]

            source_text = ", ".join(
                f"{source['concept']} "
                f"({source['status']})"
                for source in sources
            )

            print(
                f"- {feature}: {value}"
                f" <- {source_text}"
            )

    if not found_feature:

        print(
            "No ML features explicitly determined."
        )


# ============================================================
# Test cases
# ============================================================

if __name__ == "__main__":

    test_cases = [

        # ----------------------------------------------------
        # YES
        # ----------------------------------------------------

        "I have a fever.",

        # ----------------------------------------------------
        # NO
        # ----------------------------------------------------

        "I don't have a fever.",

        # ----------------------------------------------------
        # NO
        # ----------------------------------------------------

        "I have no chest pain.",

        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        "I'm not sure if I have a fever.",

        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        "Maybe I have a cough.",

        # ----------------------------------------------------
        # Multiple YES
        # ----------------------------------------------------

        "I've been feeling very tired and feverish.",

        # ----------------------------------------------------
        # Specific concept
        # ----------------------------------------------------

        "My chest hurts when I take a deep breath.",

        # ----------------------------------------------------
        # Productive cough
        # ----------------------------------------------------

        "I've been coughing up phlegm.",

        # ----------------------------------------------------
        # Chest pain should NOT become sharp chest pain
        # ----------------------------------------------------

        "I have chest pain.",

        # ----------------------------------------------------
        # Hemoptysis
        # ----------------------------------------------------

        "I've been coughing up blood.",

        # ----------------------------------------------------
        # Multiple concepts
        # ----------------------------------------------------

        "I have a blocked nose and a runny nose.",

        # ----------------------------------------------------
        # Exertional dyspnea
        # ----------------------------------------------------

        "I feel short of breath when walking.",

        # ----------------------------------------------------
        # Negation + positive symptom
        # ----------------------------------------------------

        "I don't have fever but I have a cough.",
    ]

    print(
        "Terminology matcher loaded successfully."
    )

    print(
        f"Loaded concepts: "
        f"{len(TERMINOLOGY)}"
    )

    print(
        f"Indexed phrases: "
        f"{len(PHRASE_INDEX)}"
    )

    for text in test_cases:

        result = normalize_clinical_text(
            text
        )

        print_result(
            result
        )