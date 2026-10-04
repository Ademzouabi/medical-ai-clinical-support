import json
import re
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TERMINOLOGY_FILE = BASE_DIR / "terminology_v1.json"


# Features expected by the ML layer
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
# Extra aliases
# ============================================================

EXTRA_PHRASE_ALIASES = {
    "palpitations": [
        "heart racing",
        "heart is racing",
        "my heart is racing",
        "heart beats fast",
        "heart beating fast",
        "heart pounding",
        "heart is pounding",
    ],
    "shortness_of_breath": [
        "short of breath",
        "out of breath",
        "breathless",
        "dyspnea",
        "sob",
    ],
    "difficulty_breathing": [
        "hard to breathe",
        "hard to breath",
        "hard breathing",
        "trouble breathing",
        "difficulty breathing",
        "difficult breathing",
    ],
    "dyspnea_on_exertion": [
        "shortness of breath while walking",
        "shortness of breath when walking",
        "shortness of breath while exercising",
        "shortness of breath when exercising",
        "shortness of breath on exertion",
        "short of breath while walking",
        "short of breath when walking",
        "short of breath while exercising",
        "short of breath when exercising",
        "short of breath on exertion",
        "breathless while walking",
        "breathless when walking",
    ],
    "fever": [
        "feverish",
        "feeling feverish",
        "feel feverish",
    ],
    "chest_tightness": [
        "chest feels tight",
        "my chest feels tight",
        "chest is tight",
        "tight chest",
    ],
    "vomiting": [
        "throwing up",
        "throw up",
        "threw up",
    ],
}


# ============================================================
# Normalization
# ============================================================


def normalize_text(text):
    """
    Normalize text while preserving punctuation needed for
    sentence/clause reasoning.
    """
    if text is None:
        return ""

    text = str(text).lower().strip()

    # Expand common contractions
    contractions = {
        "i'm": "i am",
        "i've": "i have",
        "i'd": "i would",
        "i'll": "i will",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "can't": "cannot",
        "couldn't": "could not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "that's": "that is",
        "there's": "there is",
    }

    for contraction, replacement in contractions.items():
        text = re.sub(
            rf"\b{re.escape(contraction)}\b",
            replacement,
            text,
        )

    text = re.sub(r"\s+", " ", text)

    return text


def normalize_fragment(text):
    """
    Normalize a fragment for phrase matching.
    """
    text = normalize_text(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Terminology loading
# ============================================================


def load_terminology():
    with open(TERMINOLOGY_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        concepts = data.get("concepts", [])
    elif isinstance(data, list):
        concepts = data
    else:
        raise ValueError("Invalid terminology_v1.json format.")

    return concepts


TERMINOLOGY = load_terminology()


# ============================================================
# Phrase index
# ============================================================


def build_phrase_index():
    """
    phrase -> set(concept_ids)
    """

    index = {}

    for concept in TERMINOLOGY:
        concept_id = concept.get("concept_id")

        if not concept_id:
            continue

        phrases = []

        preferred = concept.get("preferred_term")
        if preferred:
            phrases.append(preferred)

        phrases.extend(concept.get("synonyms", []))

        for phrase in phrases:
            normalized = normalize_fragment(phrase)

            if normalized:
                index.setdefault(normalized, set()).add(concept_id)

    # Add manual aliases
    for concept_id, aliases in EXTRA_PHRASE_ALIASES.items():
        for alias in aliases:
            normalized = normalize_fragment(alias)

            if normalized:
                index.setdefault(normalized, set()).add(concept_id)

    return index


PHRASE_INDEX = build_phrase_index()

# Longest phrases first prevents shorter phrases from taking
# precedence over more specific medical expressions.
SORTED_PHRASES = sorted(
    PHRASE_INDEX.keys(),
    key=lambda phrase: len(phrase.split()),
    reverse=True,
)


# ============================================================
# ML mappings
# ============================================================


def get_feature_mapping(concept_id):
    for concept in TERMINOLOGY:
        if concept.get("concept_id") == concept_id:
            return concept.get("ml_feature_mapping", {})
    return {}


# ============================================================
# Sentence / clause handling
# ============================================================


def split_sentences(text):
    """
    Keep sentence boundaries because contradictions and
    corrections can occur across sentences.
    """
    return [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]


def split_clauses(sentence):
    """
    Split on strong discourse boundaries.

    Examples:
        fever but cough
        unsure fever, but definitely cough
        no fever, however I have cough
    """

    parts = re.split(
        r"\s+\bbut\b\s+|\s+\bhowever\b\s*",
        sentence,
        flags=re.IGNORECASE,
    )

    return [part.strip(" ,") for part in parts if part.strip(" ,")]


def split_assertion_segments(clause):
    """
    Split comma-separated independent assertions.

    Important:
        "fever, chills, and fatigue"
        stays together.

        "cough, no fever, and I am unsure about wheezing"
        becomes separate assertions.

    A trailing uncertainty phrase such as:
        "No fever, I think."
    must remain attached to the preceding assertion.
    """

    parts = re.split(
        r",\s*(?="
        r"no\b"
        r"|without\b"
        r"|denies?\b"
        r"|deny\b"
        r"|i am\b"
        r"|i have\b"
        r"|i do\b"
        r"|i feel\b"
        r"|i get\b"
        r"|and i am\b"
        r"|and i have\b"
        r"|and i do\b"
        r"|and i feel\b"
        r"|and i get\b"
        r"|i think\b"
        r"|i guess\b"
        r"|i might\b"
        r"|i may\b"
        r"|i could\b"
        r"|i am not sure\b"
        r"|i am unsure\b"
        r"|i do not know\b"
        r"|i do not think\b"
        r")",
        clause,
        flags=re.IGNORECASE,
    )

    parts = [part.strip(" ,") for part in parts if part.strip(" ,")]

    # Re-attach trailing context that contains no medical concept.
    #
    # Example:
    #   "No fever, I think"
    #
    # becomes:
    #   "No fever, I think"
    #
    # But:
    #   "cough, I think fever"
    #
    # stays split because "I think fever" contains a concept.
    merged = []

    for part in parts:
        if merged and not find_concepts(part) and has_uncertainty(part):
            merged[-1] = f"{merged[-1]}, {part}"
        else:
            merged.append(part)

    return merged


# ============================================================
# Context detection
# ============================================================

UNCERTAINTY_PATTERNS = [
    r"\bnot sure\b",
    r"\bunsure\b",
    r"\buncertain\b",
    r"\bmaybe\b",
    r"\bmight have\b",
    r"\bmay have\b",
    r"\bpossibly\b",
    r"\bi guess\b",
    r"\bi think\b",
    r"\bi do not think\b",
    r"\bi do not know\b",
    r"\bdo not know whether\b",
    r"\bdo not know if\b",
    r"\bnot sure whether\b",
    r"\bnot sure if\b",
    r"\bunsure whether\b",
    r"\bunsure if\b",
    r"\bcould have\b",
]


NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bdeny\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bdo not have\b",
    r"\bdoes not have\b",
    r"\bdid not have\b",
    r"\bhave not\b",
    r"\bhas not\b",
    r"\bhad not\b",
    r"\bis not\b",
    r"\bare not\b",
    r"\bwas not\b",
    r"\bwere not\b",
]


POSITIVE_PATTERNS = [
    r"\bi also have\b",
    r"\bi have\b",
    r"\bi am\b",
    r"\bi get\b",
    r"\bi feel\b",
    r"\bi experience\b",
    r"\bi notice\b",
    r"\bi suffer\b",
    r"\breport\b",
    r"\breports\b",
    r"\breported\b",
    r"\bcomplain\b",
    r"\bcomplains\b",
    r"\bcomplained\b",
    r"\bexperiencing\b",
    r"\bfeels\b",
    r"\bfeel\b",
    r"\bwith\b",
    r"\bdo have\b",
    r"\bdefinitely have\b",
    r"\bactually have\b",
]


OTHER_PERSON_PATTERNS = [
    r"\bmy brother\b",
    r"\bmy sister\b",
    r"\bmy father\b",
    r"\bmy mother\b",
    r"\bmy dad\b",
    r"\bmy mom\b",
    r"\bmy friend\b",
    r"\bmy husband\b",
    r"\bmy wife\b",
    r"\bmy son\b",
    r"\bmy daughter\b",
]


DOCTOR_REPORT_PATTERNS = [
    r"\bthe doctor said\b",
    r"\bdoctor said\b",
    r"\bthe doctor told me\b",
    r"\bdoctor told me\b",
]


def contains_pattern(text, patterns):
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def is_other_person_statement(text):
    return contains_pattern(text, OTHER_PERSON_PATTERNS)


def is_doctor_report(text):
    return contains_pattern(text, DOCTOR_REPORT_PATTERNS)


def has_uncertainty(text):
    return contains_pattern(text, UNCERTAINTY_PATTERNS)


def has_negation(text):
    """
    Avoid treating:
        I do not know...
        I do not think...

    as ordinary negation.
    """

    cleaned = re.sub(
        r"\bi do not know\b|\bi do not think\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if re.search(
        r"\b(?:do|does|did)\s+not\b.*?\b(?:have|has|had)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return True

    return contains_pattern(cleaned, NEGATION_PATTERNS)


def has_positive_statement(text):
    return contains_pattern(text, POSITIVE_PATTERNS)


# ============================================================
# Concept detection
# ============================================================


def find_concepts(text):
    """
    Return concept IDs found in a text fragment.

    Uses normalized substring matching and longest phrases first.
    """

    normalized = normalize_fragment(text)

    found = []

    for phrase in SORTED_PHRASES:
        pattern = rf"\b{re.escape(phrase)}\b"

        if re.search(pattern, normalized):
            found.extend(PHRASE_INDEX[phrase])

    return list(dict.fromkeys(found))


# ============================================================
# Status detection
# ============================================================


def status_for_segment(segment, concept_id):
    """
    Determine YES / NO / UNKNOWN for a concept in one assertion.
    """

    normalized = normalize_fragment(segment)

    if not normalized:
        return None

    # Questions are intentionally conservative.
    if segment.strip().endswith("?"):
        return "UNKNOWN"

    # If this is explicitly somebody else's statement,
    # do not assign it to the user.
    if is_other_person_statement(normalized):
        return None

    # Doctor-reported findings are not treated as the patient's
    # current self-report.
    if is_doctor_report(normalized):
        return None

    # Find the concept phrase position when possible.
    concept_phrases = []

    for phrase, concept_ids in PHRASE_INDEX.items():
        if concept_id in concept_ids:
            concept_phrases.append(phrase)

    concept_position = None

    for phrase in sorted(
        concept_phrases,
        key=lambda value: len(value),
        reverse=True,
    ):
        match = re.search(
            rf"\b{re.escape(phrase)}\b",
            normalized,
        )

        if match:
            concept_position = match.start()
            break

    if concept_position is None:
        return None

    # --------------------------------------------------------
    # Local context before the concept
    # --------------------------------------------------------

    before = normalized[:concept_position].strip()
    # Uncertainty overrides a preceding weak negation.
    # Example:
    # "No fever, I think." -> UNKNOWN
    if has_uncertainty(normalized):
        return "UNKNOWN"
    # Explicit uncertainty immediately governing the concept.
    local_uncertainty_patterns = [
        r"\bnot sure\b",
        r"\bunsure\b",
        r"\buncertain\b",
        r"\bmaybe\b",
        r"\bmight have\b",
        r"\bmay have\b",
        r"\bpossibly\b",
        r"\bi guess\b",
        r"\bi think\b",
        r"\bi do not think\b",
        r"\bi do not know\b",
        r"\bdo not know whether\b",
        r"\bdo not know if\b",
        r"\bnot sure whether\b",
        r"\bnot sure if\b",
        r"\bunsure whether\b",
        r"\bunsure if\b",
        r"\bcould have\b",
    ]

    if contains_pattern(before, local_uncertainty_patterns):
        return "UNKNOWN"

    # Explicit negation governing the concept.
    local_negation_patterns = [
        r"\bno$",
        r"\bno\s*$",
        r"\bwithout$",
        r"\bdeny$",
        r"\bdenies$",
        r"\bdenied$",
        r"\bdo not have$",
        r"\bdoes not have$",
        r"\bdid not have$",
        r"\bhave not$",
        r"\bhas not$",
        r"\bhad not$",
        r"\bis not$",
        r"\bare not$",
        r"\bwas not$",
        r"\bwere not$",
    ]

    if contains_pattern(before, local_negation_patterns):
        return "NO"

    # "no fever" / "without fever"
    if re.search(
        r"\b(no|without|denies?|denied)\s+$",
        before,
    ):
        return "NO"

    # --------------------------------------------------------
    # Whole-segment context
    # --------------------------------------------------------

    uncertainty = has_uncertainty(normalized)
    negation = has_negation(normalized)
    positive = has_positive_statement(normalized)

    # Explicit positive idioms
    positive_idioms = [
        "heart racing",
        "heart is racing",
        "heart beats fast",
        "heart beating fast",
        "heart pounding",
        "heart is pounding",
        "chest feels tight",
        "chest is tight",
        "feverish",
        "throwing up",
        "throw up",
        "threw up",
    ]

    if any(idiom in normalized for idiom in positive_idioms):
        # If the same segment explicitly expresses uncertainty,
        # uncertainty wins.
        if uncertainty:
            return "UNKNOWN"

        return "YES"

    # Uncertainty is safer than positive inference.
    if uncertainty:
        return "UNKNOWN"

    if negation:
        return "NO"

    if positive:
        return "YES"

    # A bare medical term is ambiguous.
    return "UNKNOWN"


# ============================================================
# Matching
# ============================================================


def match_concepts(text):
    """
    Main clinical terminology matcher.

    Returns:
        [
            {
                "concept_id": "...",
                "status": "YES"
            },
            ...
        ]

    Status values:
        YES
        NO
        UNKNOWN
    """

    text = normalize_text(text)

    if not text:
        return []

    # concept_id -> list of observed statuses
    observations = {}

    sentences = split_sentences(text)

    for sentence in sentences:
        clauses = split_clauses(sentence)

        for clause in clauses:
            segments = split_assertion_segments(clause)

            for segment in segments:
                concept_ids = find_concepts(segment)

                if not concept_ids:
                    continue

                for concept_id in concept_ids:
                    status = status_for_segment(
                        segment,
                        concept_id,
                    )

                    if status is None:
                        continue

                    observations.setdefault(
                        concept_id,
                        [],
                    ).append(
                        {
                            "status": status,
                            "segment": segment,
                        }
                    )

    # --------------------------------------------------------
    # Resolve multiple observations
    # --------------------------------------------------------

    results = []

    for concept_id, evidence in observations.items():
        statuses = [item["status"] for item in evidence]

        # Explicit contradiction = UNKNOWN unless later evidence
        # explicitly corrects the earlier statement.
        if "YES" in statuses and "NO" in statuses:
            corrected = False
            positive_evidence = [item for item in evidence if item["status"] == "YES"]

            for item in positive_evidence:
                segment = normalize_fragment(item["segment"])

                correction_markers = [
                    r"\bnow\b",
                    r"\bcurrently\b",
                    r"\bactually\b",
                    r"\bbut i have\b",
                    r"\bbut i do have\b",
                    r"\bbut i am\b",
                    r"\bbut i feel\b",
                    r"\bbut i get\b",
                ]

                if any(
                    re.search(
                        marker,
                        segment,
                        re.IGNORECASE,
                    )
                    for marker in correction_markers
                ):
                    corrected = True
                    break

            if corrected:
                final_status = "YES"
            else:
                # Genuine unresolved contradiction.
                #
                # Example:
                # "I have fever. I don't have fever."
                #
                # We stay conservative.
                final_status = "UNKNOWN"

        # Positive later/current evidence beats uncertainty.
        elif "YES" in statuses:
            final_status = "YES"

        # Negative evidence beats uncertainty.
        elif "NO" in statuses:
            final_status = "NO"

        else:
            final_status = "UNKNOWN"

        results.append(
            {
                "concept_id": concept_id,
                "status": final_status,
            }
        )

    return results


# ============================================================
# Feature conversion
# ============================================================


def concepts_to_features(matches):
    """
    Convert terminology matches into ML features.

    IMPORTANT:
    Returns TWO values because the regression test suite and
    downstream code expect:

        features, feature_sources = concepts_to_features(matches)

    features:
        {
            "fever": 1,
            "cough": 0,
            ...
        }

    feature_sources:
        {
            "shortness_of_breath": [
                "shortness_of_breath",
                "dyspnea_on_exertion"
            ]
        }
    """

    features = {feature: None for feature in ML_FEATURES}

    feature_sources = {feature: [] for feature in ML_FEATURES}

    for match in matches:
        concept_id = match.get("concept_id")
        status = match.get("status")

        if not concept_id or status not in {
            "YES",
            "NO",
            "UNKNOWN",
        }:
            continue

        mapping = get_feature_mapping(concept_id)

        for feature_name in mapping.keys():
            if feature_name not in features:
                continue

            feature_sources[feature_name].append(concept_id)

            current = features[feature_name]

            if status == "YES":
                # Positive evidence dominates.
                features[feature_name] = 1

            elif status == "NO":
                # Do not overwrite an existing positive.
                if current != 1:
                    features[feature_name] = 0

            elif status == "UNKNOWN":
                # UNKNOWN only fills an empty state.
                if current is None:
                    features[feature_name] = None

    # Remove empty source lists for cleaner output.
    feature_sources = {
        feature: sources for feature, sources in feature_sources.items() if sources
    }

    return features, feature_sources


# ============================================================
# Full pipeline
# ============================================================


def normalize_clinical_text(text):
    """
    Full terminology pipeline.

    Returns:
        {
            "matches": [...],
            "features": {...},
            "feature_sources": {...}
        }
    """

    matches = match_concepts(text)

    features, feature_sources = concepts_to_features(matches)

    return {
        "matches": matches,
        "features": features,
        "feature_sources": feature_sources,
    }


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":
    examples = [
        "I have fever and cough.",
        "I do not have fever.",
        "I am not sure whether I have fever and chills.",
        "I don't know if I have fever, but I have fatigue.",
        "I said I had no cough, but actually I do have cough.",
        "No fever, I think.",
        "My brother has no fever, but I have fever.",
        "The doctor said I don't have fever, but I feel feverish now.",
        "I have fever. I also have cough and fatigue.",
        "I get shortness of breath while walking.",
    ]

    for example in examples:
        print("\n" + "=" * 70)
        print(example)

        matches = match_concepts(example)

        print("\nMatches:")
        for match in matches:
            print(f"  {match['concept_id']}: {match['status']}")

        features, sources = concepts_to_features(matches)

        print("\nFeatures:")
        for feature, value in features.items():
            if value is not None:
                print(f"  {feature}: {value}")

        print("\nSources:")
        for feature, source_list in sources.items():
            print(f"  {feature}: {source_list}")
