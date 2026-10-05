from itertools import product
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terminology_matcher import match_concepts, concepts_to_features


TEST_CASES = [
    # ============================================================
    # 1. BASIC POSITIVE DETECTION
    # ============================================================
    {
        "name": "Basic fever",
        "text": "I have a fever.",
        "expected": {
            "fever": "YES",
        },
    },
    {
        "name": "Basic cough",
        "text": "I have a cough.",
        "expected": {
            "cough": "YES",
        },
    },
    {
        "name": "Multiple positive symptoms",
        "text": "I have fever, cough, and fatigue.",
        "expected": {
            "fever": "YES",
            "cough": "YES",
            "fatigue": "YES",
        },
    },
    {
        "name": "Shortness of breath",
        "text": "I have shortness of breath.",
        "expected": {
            "shortness_of_breath": "YES",
        },
    },
    {
        "name": "Wheezing",
        "text": "I am wheezing.",
        "expected": {
            "wheezing": "YES",
        },
    },
    {
        "name": "Dizziness",
        "text": "I feel dizzy.",
        "expected": {
            "dizziness": "YES",
        },
    },
    {
        "name": "Vomiting",
        "text": "I am throwing up.",
        "expected": {
            "vomiting": "YES",
        },
    },
    {
        "name": "Palpitations",
        "text": "My heart is racing.",
        "expected": {
            "palpitations": "YES",
        },
    },
    # ============================================================
    # 2. BASIC NEGATION
    # ============================================================
    {
        "name": "No fever",
        "text": "I have no fever.",
        "expected": {
            "fever": "NO",
        },
    },
    {
        "name": "No cough",
        "text": "I do not have a cough.",
        "expected": {
            "cough": "NO",
        },
    },
    {
        "name": "Does not have fever",
        "text": "The patient does not have fever.",
        "expected": {
            "fever": "NO",
        },
    },
    {
        "name": "Doesn't have fever",
        "text": "I don't have fever.",
        "expected": {
            "fever": "NO",
        },
    },
    {
        "name": "Denies fever",
        "text": "The patient denies fever.",
        "expected": {
            "fever": "NO",
        },
    },
    {
        "name": "No chest pain",
        "text": "I have no chest pain.",
        "expected": {
            "chest_pain": "NO",
        },
    },
    # ============================================================
    # 3. NEGATION WITH CONJUNCTIONS
    # ============================================================
    {
        "name": "No fever or chills",
        "text": "I have no fever or chills.",
        "expected": {
            "fever": "NO",
            "chills": "NO",
        },
    },
    {
        "name": "No fever and cough",
        "text": "I have no fever and no cough.",
        "expected": {
            "fever": "NO",
            "cough": "NO",
        },
    },
    {
        "name": "No fever cough or wheezing",
        "text": "There is no fever, cough, or wheezing.",
        "expected": {
            "fever": "NO",
            "cough": "NO",
            "wheezing": "NO",
        },
    },
    {
        "name": "Denies fever and fatigue",
        "text": "The patient denies fever and fatigue.",
        "expected": {
            "fever": "NO",
            "fatigue": "NO",
        },
    },
    {
        "name": "No chest pain or shortness of breath",
        "text": "I don't have chest pain or shortness of breath.",
        "expected": {
            "chest_pain": "NO",
            "shortness_of_breath": "NO",
        },
    },
    # ============================================================
    # 4. UNCERTAINTY
    # ============================================================
    {
        "name": "Unsure fever",
        "text": "I'm unsure whether I have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Not sure fever",
        "text": "I'm not sure if I have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Uncertain fever",
        "text": "I am uncertain whether I have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Maybe fever",
        "text": "Maybe I have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Might have fever",
        "text": "I might have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Possibly fever",
        "text": "I possibly have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "I guess fever",
        "text": "I guess I have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Not sure about cough",
        "text": "I'm not sure whether I have a cough.",
        "expected": {
            "cough": "UNKNOWN",
        },
    },
    # ============================================================
    # 5. UNCERTAINTY + CONJUNCTION
    # ============================================================
    {
        "name": "Uncertain fever and chills",
        "text": "The patient is not sure whether they have fever and chills.",
        "expected": {
            "fever": "UNKNOWN",
            "chills": "UNKNOWN",
        },
    },
    {
        "name": "Uncertain cough and wheezing",
        "text": "The patient is not sure whether they have cough and wheezing.",
        "expected": {
            "cough": "UNKNOWN",
            "wheezing": "UNKNOWN",
        },
    },
    {
        "name": "Uncertain fever and fatigue",
        "text": "The patient is unsure whether they have fever and fatigue.",
        "expected": {
            "fever": "UNKNOWN",
            "fatigue": "UNKNOWN",
        },
    },
    {
        "name": "Uncertain chest pain and dizziness",
        "text": "The patient is not sure whether they have chest pain and dizziness.",
        "expected": {
            "chest_pain": "UNKNOWN",
            "dizziness": "UNKNOWN",
        },
    },
    {
        "name": "Uncertain three symptoms",
        "text": "The patient is unsure whether they have fever, chills, and fatigue.",
        "expected": {
            "fever": "UNKNOWN",
            "chills": "UNKNOWN",
            "fatigue": "UNKNOWN",
        },
    },
    {
        "name": "Uncertain symptoms with OR",
        "text": "I don't know whether I have fever or chills.",
        "expected": {
            "fever": "UNKNOWN",
            "chills": "UNKNOWN",
        },
    },
    {
        "name": "Maybe fever and cough",
        "text": "Maybe I have fever and cough.",
        "expected": {
            "fever": "UNKNOWN",
            "cough": "UNKNOWN",
        },
    },
    {
        "name": "Might have fever and dizziness",
        "text": "I might have fever and dizziness.",
        "expected": {
            "fever": "UNKNOWN",
            "dizziness": "UNKNOWN",
        },
    },
    # ============================================================
    # 6. MIXED POSITIVE + NEGATIVE
    # ============================================================
    {
        "name": "Fever but no cough",
        "text": "I have fever but no cough.",
        "expected": {
            "fever": "YES",
            "cough": "NO",
        },
    },
    {
        "name": "No fever but cough",
        "text": "I have no fever but I have a cough.",
        "expected": {
            "fever": "NO",
            "cough": "YES",
        },
    },
    {
        "name": "No chest pain but shortness of breath",
        "text": "I don't have chest pain, but I have shortness of breath.",
        "expected": {
            "chest_pain": "NO",
            "shortness_of_breath": "YES",
        },
    },
    {
        "name": "Denied fever but reported cough",
        "text": "I deny fever but report cough.",
        "expected": {
            "fever": "NO",
            "cough": "YES",
        },
    },
    {
        "name": "Fever and chills but no cough",
        "text": "I have fever and chills but no cough.",
        "expected": {
            "fever": "YES",
            "chills": "YES",
            "cough": "NO",
        },
    },
    {
        "name": "Cough and wheezing but no fever",
        "text": "I have cough and wheezing but no fever.",
        "expected": {
            "cough": "YES",
            "wheezing": "YES",
            "fever": "NO",
        },
    },
    # ============================================================
    # 7. MIXED POSITIVE + UNKNOWN
    # ============================================================
    {
        "name": "Unknown fever but chest pain",
        "text": "I'm unsure whether I have fever, but I do have chest pain.",
        "expected": {
            "fever": "UNKNOWN",
            "chest_pain": "YES",
        },
    },
    {
        "name": "Unknown fever but cough",
        "text": "I'm not sure about fever, but I definitely have a cough.",
        "expected": {
            "fever": "UNKNOWN",
            "cough": "YES",
        },
    },
    {
        "name": "Unknown fever but positive fatigue",
        "text": "I don't know if I have fever, but I have fatigue.",
        "expected": {
            "fever": "UNKNOWN",
            "fatigue": "YES",
        },
    },
    {
        "name": "Unknown cough but positive fever",
        "text": "I'm unsure about cough, but I definitely have fever.",
        "expected": {
            "cough": "UNKNOWN",
            "fever": "YES",
        },
    },
    {
        "name": "Unknown fever and positive cough",
        "text": "I may have fever, but I definitely have cough.",
        "expected": {
            "fever": "UNKNOWN",
            "cough": "YES",
        },
    },
    # ============================================================
    # 8. UNCERTAINTY + NEGATION
    # ============================================================
    {
        "name": "Unsure fever but no cough",
        "text": "I'm not sure whether I have fever, but I don't have cough.",
        "expected": {
            "fever": "UNKNOWN",
            "cough": "NO",
        },
    },
    {
        "name": "No fever but unsure cough",
        "text": "I don't have fever, but I'm not sure about cough.",
        "expected": {
            "fever": "NO",
            "cough": "UNKNOWN",
        },
    },
    {
        "name": "Unsure chest pain but no fever",
        "text": "I'm unsure whether I have chest pain, but I don't have fever.",
        "expected": {
            "chest_pain": "UNKNOWN",
            "fever": "NO",
        },
    },
    # ============================================================
    # 9. LATER INFORMATION / OVERRIDING INFORMATION
    # ============================================================
    {
        "name": "Initially negative then positive fever",
        "text": "I initially had no fever but now I have fever.",
        "expected": {
            "fever": "YES",
        },
    },
    {
        "name": "Previously negative then positive cough",
        "text": "I said I had no cough, but actually I do have cough.",
        "expected": {
            "cough": "YES",
        },
    },
    {
        "name": "Unsure then positive fever",
        "text": "At first I wasn't sure about fever, but now I definitely have fever.",
        "expected": {
            "fever": "YES",
        },
    },
    {
        "name": "Unsure then negative fever",
        "text": "I wasn't sure about fever earlier, but I don't have fever now.",
        "expected": {
            "fever": "NO",
        },
    },
    # ============================================================
    # 10. CONTRADICTIONS
    # ============================================================
    {
        "name": "Direct contradiction fever",
        "text": "I have fever but I don't have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Direct contradiction cough",
        "text": "I don't have cough but I have cough.",
        "expected": {
            "cough": "UNKNOWN",
        },
    },
    {
        "name": "Positive then negative fever",
        "text": "I have fever. I don't have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Negative then positive cough",
        "text": "I don't have cough. I have cough.",
        "expected": {
            "cough": "UNKNOWN",
        },
    },
    # ============================================================
    # 11. MESSY HUMAN LANGUAGE
    # ============================================================
    {
        "name": "Question mark fever",
        "text": "Fever?",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Maybe fever short",
        "text": "Maybe fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Not sure fever short",
        "text": "Not sure fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "No fever I think",
        "text": "No fever, I think.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "I guess fever",
        "text": "I guess I have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "I don't think fever",
        "text": "I don't think I have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Pretty sure no fever",
        "text": "I'm pretty sure I don't have fever.",
        "expected": {
            "fever": "NO",
        },
    },
    {
        "name": "Think might have cough",
        "text": "I think I might have cough.",
        "expected": {
            "cough": "UNKNOWN",
        },
    },
    {
        "name": "Could have fever",
        "text": "I could have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    # ============================================================
    # 12. NATURAL / CLINICAL LANGUAGE
    # ============================================================
    {
        "name": "Dyspnea",
        "text": "The patient reports dyspnea.",
        "expected": {
            "shortness_of_breath": "YES",
        },
    },
    {
        "name": "Difficulty breathing",
        "text": "The patient reports difficulty breathing.",
        "expected": {
            "shortness_of_breath": "YES",
            "difficulty_breathing": "YES",
        },
    },
    {
        "name": "Coughing up phlegm",
        "text": "I am coughing up phlegm.",
        "expected": {
            "cough": "YES",
            "productive_cough": "YES",
        },
    },
    {
        "name": "Coughing up blood",
        "text": "I am coughing up blood.",
        "expected": {
            "cough": "YES",
            "hemoptysis": "YES",
        },
    },
    {
        "name": "Blocked and runny nose",
        "text": "I have a blocked nose and a runny nose.",
        "expected": {
            "nasal_congestion": "YES",
            "coryza": "YES",
        },
    },
    {
        "name": "Pain with deep breathing",
        "text": "I have pain when taking a deep breath.",
        "expected": {
            "pain_with_breathing": "YES",
        },
    },
    {
        "name": "Dyspnea on exertion",
        "text": "I get short of breath when exercising.",
        "expected": {
            "shortness_of_breath": "YES",
            "dyspnea_on_exertion": "YES",
        },
    },
    {
        "name": "Shortness of breath walking",
        "text": "I get shortness of breath while walking.",
        "expected": {
            "shortness_of_breath": "YES",
            "dyspnea_on_exertion": "YES",
        },
    },
    {
        "name": "Chest tightness",
        "text": "My chest feels tight.",
        "expected": {
            "chest_tightness": "YES",
        },
    },
    {
        "name": "Sharp chest pain",
        "text": "I have sharp chest pain.",
        "expected": {
            "sharp_chest_pain": "YES",
            "chest_pain": "YES",
        },
    },
    # ============================================================
    # 13. ABBREVIATIONS / SYNONYMS
    # ============================================================
    {
        "name": "SOB abbreviation",
        "text": "I have SOB.",
        "expected": {
            "shortness_of_breath": "YES",
        },
    },
    {
        "name": "Heart racing",
        "text": "My heart is racing.",
        "expected": {
            "palpitations": "YES",
        },
    },
    {
        "name": "Dizzy synonym",
        "text": "I feel dizzy.",
        "expected": {
            "dizziness": "YES",
        },
    },
    {
        "name": "Throwing up synonym",
        "text": "I've been throwing up.",
        "expected": {
            "vomiting": "YES",
        },
    },
    # ============================================================
    # 14. SCOPE / SUBJECT SEPARATION
    # ============================================================
    {
        "name": "Other person has no fever",
        "text": "My brother has no fever, but I have fever.",
        "expected": {
            "fever": "YES",
        },
    },
    {
        "name": "Other person has cough",
        "text": "I don't know if my brother has cough. I have cough.",
        "expected": {
            "cough": "YES",
        },
    },
    {
        "name": "Doctor says no fever, patient reports fever",
        "text": "The doctor said I don't have fever, but I feel feverish now.",
        "expected": {
            "fever": "YES",
        },
    },
    # ============================================================
    # 15. MULTI-SENTENCE CONTEXT
    # ============================================================
    {
        "name": "Positive symptoms across sentences",
        "text": "I have fever. I also have cough and fatigue.",
        "expected": {
            "fever": "YES",
            "cough": "YES",
            "fatigue": "YES",
        },
    },
    {
        "name": "Negative symptoms across sentences",
        "text": "I don't have fever. I don't have cough either.",
        "expected": {
            "fever": "NO",
            "cough": "NO",
        },
    },
    {
        "name": "Unknown then positive",
        "text": "I'm not sure about fever. However, I definitely have cough.",
        "expected": {
            "fever": "UNKNOWN",
            "cough": "YES",
        },
    },
    {
        "name": "Negative then positive",
        "text": "I don't have fever. However, I have chills.",
        "expected": {
            "fever": "NO",
            "chills": "YES",
        },
    },
    # ============================================================
    # 16. OVERLAPPING CONCEPTS
    # ============================================================
    {
        "name": "Difficulty breathing overlap",
        "text": "I have difficulty breathing.",
        "expected": {
            "shortness_of_breath": "YES",
            "difficulty_breathing": "YES",
        },
    },
    {
        "name": "Exertional dyspnea overlap",
        "text": "I have dyspnea on exertion.",
        "expected": {
            "shortness_of_breath": "YES",
            "dyspnea_on_exertion": "YES",
        },
    },
    {
        "name": "Sharp chest pain overlap",
        "text": "I have sharp chest pain.",
        "expected": {
            "sharp_chest_pain": "YES",
            "chest_pain": "YES",
        },
    },
    {
        "name": "Productive cough overlap",
        "text": "I have a productive cough.",
        "expected": {
            "cough": "YES",
            "productive_cough": "YES",
        },
    },
    {
        "name": "Hemoptysis overlap",
        "text": "I am coughing blood.",
        "expected": {
            "cough": "YES",
            "hemoptysis": "YES",
        },
    },
    # ============================================================
    # 17. ML FEATURE CONVERSION
    # ============================================================
    {
        "name": "ML feature positive",
        "text": "I have fever.",
        "expected": {
            "fever": "YES",
        },
        "expected_features": {
            "fever": 1,
        },
    },
    {
        "name": "ML feature negative",
        "text": "I don't have fever.",
        "expected": {
            "fever": "NO",
        },
        "expected_features": {
            "fever": 0,
        },
    },
    {
        "name": "ML feature unknown",
        "text": "I'm not sure whether I have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
        "expected_features": {
            "fever": None,
        },
    },
    {
        "name": "Multiple ML feature states",
        "text": "I have fever and cough, but no wheezing.",
        "expected": {
            "fever": "YES",
            "cough": "YES",
            "wheezing": "NO",
        },
        "expected_features": {
            "fever": 1,
            "cough": 1,
            "wheezing": 0,
        },
    },
    {
        "name": "Mixed ML states with unknown",
        "text": "I have cough, no fever, and I'm unsure about wheezing.",
        "expected": {
            "cough": "YES",
            "fever": "NO",
            "wheezing": "UNKNOWN",
        },
        "expected_features": {
            "cough": 1,
            "fever": 0,
            "wheezing": None,
        },
    },
    {
        "name": "Unknown must not become zero",
        "text": "I might have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
        "expected_features": {
            "fever": None,
        },
    },
    # ============================================================
    # 18. EDGE CASES / ADVERSARIAL LANGUAGE
    # ============================================================
    {
        "name": "Negation before symptom with extra words",
        "text": "At the moment, I do not really have any fever.",
        "expected": {
            "fever": "NO",
        },
    },
    {
        "name": "Uncertainty with extra words",
        "text": "At the moment, I really don't know if I have fever.",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Positive with emphasis",
        "text": "I definitely have fever.",
        "expected": {
            "fever": "YES",
        },
    },
    {
        "name": "Definitely no fever",
        "text": "I definitely do not have fever.",
        "expected": {
            "fever": "NO",
        },
    },
    {
        "name": "Possible cough but definite fever",
        "text": "I definitely have fever, but I may have cough.",
        "expected": {
            "fever": "YES",
            "cough": "UNKNOWN",
        },
    },
    {
        "name": "Possible fever but definite cough",
        "text": "I may have fever, but I definitely have cough.",
        "expected": {
            "fever": "UNKNOWN",
            "cough": "YES",
        },
    },
    {
        "name": "No symptoms except fever",
        "text": "I have no cough, no chills, and no fatigue. I do have fever.",
        "expected": {
            "cough": "NO",
            "chills": "NO",
            "fatigue": "NO",
            "fever": "YES",
        },
    },
    {
        "name": "No symptoms except cough",
        "text": "I don't have fever or chills, but I have a cough.",
        "expected": {
            "fever": "NO",
            "chills": "NO",
            "cough": "YES",
        },
    },
    {
        "name": "Positive fever with negative cough",
        "text": "I have fever and no cough.",
        "expected": {
            "fever": "YES",
            "cough": "NO",
        },
    },
    {
        "name": "Positive cough with uncertain wheezing",
        "text": "I have cough and I am unsure about wheezing.",
        "expected": {
            "cough": "YES",
            "wheezing": "UNKNOWN",
        },
    },
    {
        "name": "Other person is only fever evidence",
        "text": "My brother has fever.",
        "expected": {},
        "expected_excluded": ["fever"],
    },
    {
        "name": "Other person fever with patient cough",
        "text": "My brother has fever and I have cough.",
        "expected": {
            "cough": "YES",
        },
        "expected_excluded": ["fever"],
    },
    {
        "name": "Doctor report is only fever evidence",
        "text": "The doctor said I do not have fever.",
        "expected": {},
        "expected_excluded": ["fever"],
    },
    {
        "name": "Doctor fever with patient cough",
        "text": "The doctor said I do not have fever and I have cough.",
        "expected": {
            "cough": "YES",
        },
        "expected_excluded": ["fever"],
    },
    {
        "name": "Explicit negative feverish",
        "text": "I am not feverish.",
        "expected": {
            "fever": "NO",
        },
    },
    {
        "name": "Explicit negative vomiting",
        "text": "I am not vomiting.",
        "expected": {
            "vomiting": "NO",
        },
    },
    {
        "name": "Positive wording question",
        "text": "I have fever?",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Direct symptom question",
        "text": "Do I have fever?",
        "expected": {
            "fever": "UNKNOWN",
        },
    },
    {
        "name": "Current negative corrects earlier positive",
        "text": "I have fever but I do not have fever now.",
        "expected": {
            "fever": "NO",
        },
    },
    {
        "name": "Curly apostrophe negation",
        "text": "I don’t have fever.",
        "expected": {
            "fever": "NO",
        },
    },
    {
        "name": "No shortness of breath, orthopnea uncertain",
        "text": "I do not have shortness of breath, but I am unsure about orthopnea.",
        "expected": {
            "shortness_of_breath": "NO",
            "orthopnea": "UNKNOWN",
        },
        "expected_features": {
            "shortness_of_breath": None,
        },
    },
    {
        "name": "Orthopnea supports shortness-of-breath feature",
        "text": "I have orthopnea.",
        "expected": {
            "orthopnea": "YES",
        },
        "expected_features": {
            "shortness_of_breath": 1,
        },
    },
    {
        "name": "Orthopnea despite no general shortness of breath",
        "text": "I have orthopnea but no shortness of breath.",
        "expected": {
            "orthopnea": "YES",
            "shortness_of_breath": "NO",
        },
        "expected_features": {
            "shortness_of_breath": 1,
        },
    },
    {
        "name": "Shortness of breath without orthopnea",
        "text": "I have shortness of breath but no orthopnea.",
        "expected": {
            "shortness_of_breath": "YES",
            "orthopnea": "NO",
        },
        "expected_features": {
            "shortness_of_breath": 1,
        },
    },
    {
        "name": "No shortness of breath and orthopnea; exertional unknown",
        "text": "I don't have shortness of breath and I don't have orthopnea.",
        "expected": {
            "shortness_of_breath": "NO",
            "orthopnea": "NO",
        },
        "expected_excluded": ["dyspnea_on_exertion"],
        "expected_features": {
            "shortness_of_breath": None,
        },
    },
    {
        "name": "Uncertain shortness of breath and orthopnea",
        "text": "I am unsure about both shortness of breath and orthopnea.",
        "expected": {
            "shortness_of_breath": "UNKNOWN",
            "orthopnea": "UNKNOWN",
        },
        "expected_features": {
            "shortness_of_breath": None,
        },
    },
    {
        "name": "Chest congestion with phlegm is not productive cough",
        "text": "I have chest congestion with phlegm.",
        "expected": {
            "chest_congestion_with_phlegm": "YES",
        },
        "expected_features": {
            "chest_congestion": 1,
            "productive_cough": None,
        },
    },
    {
        "name": "Negative productive cough with phlegm congestion",
        "text": "I do not have productive cough, but I have chest congestion with phlegm.",
        "expected": {
            "productive_cough": "NO",
            "chest_congestion_with_phlegm": "YES",
        },
        "expected_features": {
            "productive_cough": None,
        },
    },
    {
        "name": "Both productive cough sources explicitly negative",
        "text": "I do not have productive cough and I do not have chest congestion with phlegm.",
        "expected": {
            "productive_cough": "NO",
            "chest_congestion_with_phlegm": "NO",
        },
        "expected_features": {
            "productive_cough": 0,
        },
    },
]


DIRECT_AGGREGATION_CASES = []
STATUS_VALUES = ("YES", "NO", "UNKNOWN")

for shortness_statuses in product(STATUS_VALUES, repeat=3):
    concept_ids = (
        "shortness_of_breath",
        "orthopnea",
        "dyspnea_on_exertion",
    )
    expected = (
        1
        if "YES" in shortness_statuses
        else 0
        if all(status == "NO" for status in shortness_statuses)
        else None
    )
    DIRECT_AGGREGATION_CASES.append(
        {
            "name": f"shortness_of_breath {shortness_statuses}",
            "matches": [
                {"concept_id": concept_id, "status": status}
                for concept_id, status in zip(concept_ids, shortness_statuses)
            ],
            "feature": "shortness_of_breath",
            "expected": expected,
        }
    )

for congestion_statuses in product(STATUS_VALUES, repeat=2):
    concept_ids = ("chest_congestion", "chest_congestion_with_phlegm")
    expected = (
        1
        if "YES" in congestion_statuses
        else 0
        if all(status == "NO" for status in congestion_statuses)
        else None
    )
    DIRECT_AGGREGATION_CASES.append(
        {
            "name": f"chest_congestion {congestion_statuses}",
            "matches": [
                {"concept_id": concept_id, "status": status}
                for concept_id, status in zip(concept_ids, congestion_statuses)
            ],
            "feature": "chest_congestion",
            "expected": expected,
        }
    )

for productive_cough_statuses in product(STATUS_VALUES, repeat=2):
    direct_status, proxy_status = productive_cough_statuses
    expected = (
        1
        if direct_status == "YES"
        else 0
        if direct_status == "NO" and proxy_status == "NO"
        else None
    )
    DIRECT_AGGREGATION_CASES.append(
        {
            "name": f"productive_cough {productive_cough_statuses}",
            "matches": [
                {"concept_id": "productive_cough", "status": direct_status},
                {
                    "concept_id": "chest_congestion_with_phlegm",
                    "status": proxy_status,
                },
            ],
            "feature": "productive_cough",
            "expected": expected,
        }
    )


def run_test(test_number, test_case):
    text = test_case["text"]
    expected = test_case["expected"]

    matches = match_concepts(text)

    actual = {match["concept_id"]: match["status"] for match in matches}

    errors = []

    # ------------------------------------------------------------
    # Concept status validation
    # ------------------------------------------------------------

    for concept_id, expected_status in expected.items():
        actual_status = actual.get(concept_id)

        if actual_status != expected_status:
            errors.append(
                f"{concept_id}: expected {expected_status}, got {actual_status}"
            )

    for concept_id in test_case.get("expected_excluded", []):
        if concept_id in actual:
            errors.append(
                f"{concept_id}: expected to be excluded, got {actual[concept_id]}"
            )

    # ------------------------------------------------------------
    # Feature validation
    # ------------------------------------------------------------

    expected_features = test_case.get("expected_features")

    if expected_features is not None:
        features, feature_sources = concepts_to_features(matches)

        for feature_name, expected_value in expected_features.items():
            actual_value = features.get(feature_name)

            if actual_value != expected_value:
                errors.append(
                    f"ML feature {feature_name}: "
                    f"expected {expected_value}, "
                    f"got {actual_value}"
                )

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

    if errors:
        print(f"\n[FAIL] TEST {test_number}: {test_case['name']}")
        print(f"       Input: {text}")

        for error in errors:
            print(f"       ERROR: {error}")

        print("\n       Actual matches:")

        if matches:
            for match in matches:
                print(f"       - {match['concept_id']}: {match['status']}")
        else:
            print("       - No matches")

        if expected_features is not None:
            features, feature_sources = concepts_to_features(matches)

            print("\n       Actual ML features:")

            for feature_name in expected_features:
                print(f"       - {feature_name}: {features.get(feature_name)}")

        return False

    print(f"[PASS] TEST {test_number}: {test_case['name']}")
    return True


def run_aggregation_test(test_number, test_case):
    features, _ = concepts_to_features(test_case["matches"])
    actual = features.get(test_case["feature"])

    if actual != test_case["expected"]:
        print(f"\n[FAIL] TEST {test_number}: {test_case['name']}")
        print(
            f"       {test_case['feature']}: expected "
            f"{test_case['expected']}, got {actual}"
        )
        return False

    print(f"[PASS] TEST {test_number}: {test_case['name']}")
    return True


def main():
    print("=" * 70)
    print("TERMINOLOGY MATCHER ADVERSARIAL REGRESSION TEST")
    print("=" * 70)

    total_tests = len(TEST_CASES) + len(DIRECT_AGGREGATION_CASES)
    print(f"Total tests: {total_tests}")
    print()

    passed = 0
    failed = 0

    for index, test_case in enumerate(TEST_CASES, start=1):
        if run_test(index, test_case):
            passed += 1
        else:
            failed += 1

    for index, test_case in enumerate(
        DIRECT_AGGREGATION_CASES,
        start=len(TEST_CASES) + 1,
    ):
        if run_aggregation_test(index, test_case):
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"Total:  {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")


if __name__ == "__main__":
    main()
