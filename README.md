\# Clinical AI Decision-Support Prototype



A machine-learning and natural-language processing prototype designed to explore \*\*clinical symptom normalization, medical terminology search, and ranked differential prediction\*\*.



The project is being developed as a learning and research-oriented prototype. It is \*\*not an autonomous medical diagnostic system\*\* and must not be used to make real clinical decisions.



\## Repository layout and commands

Runtime modules live in `src/`; runtime terminology data and model artifacts are resolved from their file locations, not from the current working directory.

```text
src/                 V1 runtime modules
tests/               deterministic matcher, clarification, pipeline, and end-to-end tests
models/              locked V1 model artifact
data/terminology/    runtime terminology vocabulary
data/processed/      standardized and split V1 datasets
data/evaluation/     held-out evaluation output
data/raw/            raw datasets and the unintegrated MeSH RDF resource
scripts/             data preparation, training, and experiment utilities
docs/                project notes
```

From the repository root in PowerShell, use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe src\predict.py
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m uvicorn api:app --app-dir src --reload
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -c "from pipeline import start_pipeline; print(start_pipeline('I have fever.'))"
.\.venv\Scripts\python.exe tests\test_matcher_review.py
.\.venv\Scripts\python.exe tests\test_terminology_edge_cases.py
.\.venv\Scripts\python.exe -m unittest tests\test_clarification.py
.\.venv\Scripts\python.exe -m unittest tests\test_pipeline.py
.\.venv\Scripts\python.exe -m unittest tests\test_end_to_end.py
.\.venv\Scripts\python.exe -m unittest tests\test_api.py
.\.venv\Scripts\python.exe scripts\final_evaluate_model.py
```

The evaluation command reads `data/processed/` and writes the locked artifact to `models/` plus predictions to `data/evaluation/`.

\## Overview



The goal of this project is to build a hybrid system that can take either:



\* A \*\*medical term\*\* entered directly by the user

\* A \*\*natural-language description\*\* of symptoms



and transform that information into structured clinical features that can be analyzed by a machine-learning model.



The planned pipeline is:



```text

User Input

&#x20;   │

&#x20;   ├── Medical Term Search

&#x20;   │       └── Exact / Prefix / Keyword Search

&#x20;   │

&#x20;   └── Describe It

&#x20;           └── Terminology \& NLP Normalization

&#x20;                   │

&#x20;                   ▼

&#x20;           Clinical Concepts

&#x20;                   │

&#x20;                   ▼

&#x20;           Structured Features

&#x20;                   │

&#x20;                   ▼

&#x20;           Machine Learning Model

&#x20;                   │

&#x20;                   ▼

&#x20;           Ranked Differential

&#x20;                   │

&#x20;                   ▼

&#x20;           Educational Explanation

```



An LLM may be integrated later for more complex language understanding and explanations, but it is intentionally \*\*not the primary prediction model\*\*.



\---



\## Current Status



| Component                                | Status         |

| ---------------------------------------- | -------------- |

| Project scope                            | ✅ Complete     |

| Dataset research                         | ✅ Complete     |

| Dataset auditing                         | ✅ Complete     |

| Dataset harmonization                    | ✅ Complete     |

| Leakage-safe train/validation/test split | ✅ Complete     |

| ML baseline models                       | ✅ Complete     |

| Model tuning                             | ✅ Complete     |

| Final Logistic Regression model          | ✅ Complete     |

| Prediction interface                     | ✅ Complete     |

| Medical terminology vocabulary           | ✅ Complete     |

| Terminology matcher                      | ✅ Functional   |

| Terminology edge-case testing            | ✅ 20/20 passed |

| Medical Term search                      | 🚧 Next        |

| Describe It search integration           | 🚧 Next        |

| React frontend                           | ⏳ Planned      |

| FastAPI backend                          | ⏳ Planned      |

| LLM integration                          | ⏳ Planned      |

| RAG                                      | ⏳ Optional     |

| End-to-end evaluation                    | ⏳ Planned      |



\---



\# 1. Project Scope



The initial ML prototype focuses on \*\*15 respiratory and cardiopulmonary conditions\*\*:



1\. Pneumonia

2\. Acute bronchitis

3\. Asthma

4\. Chronic obstructive pulmonary disease (COPD)

5\. Common cold

6\. Flu

7\. Acute sinusitis

8\. Chronic sinusitis

9\. Laryngitis

10\. Pulmonary embolism

11\. Pneumothorax

12\. Pleural effusion

13\. Heart failure

14\. Angina

15\. Pulmonary hypertension



The scope was deliberately limited instead of attempting to classify all diseases present in the original dataset.



\---



\# 2. Canonical Clinical Features



The ML model currently uses 26 binary clinical features:



\### Respiratory



\* `shortness\_of\_breath`

\* `difficulty\_breathing`

\* `pain\_with\_breathing`

\* `cough`

\* `productive\_cough`

\* `wheezing`

\* `chest\_tightness`

\* `chest\_congestion`

\* `hemoptysis`

\* `apnea`

\* `hoarseness`

\* `sore\_throat`

\* `nasal\_congestion`

\* `sinus\_congestion`

\* `coryza`



\### Cardiovascular



\* `sharp\_chest\_pain`

\* `palpitations`

\* `irregular\_heartbeat`

\* `increased\_heart\_rate`

\* `dizziness`



\### Systemic



\* `fever`

\* `chills`

\* `fatigue`

\* `malaise`

\* `general\_weakness`



\### Other



\* `vomiting`



The distinction between `YES`, `NO`, and `UNKNOWN` is part of the application design. `UNKNOWN` is \*\*not\*\* equivalent to `NO`.



\---



\# 3. Dataset



The main ML dataset is the \*\*Disease \& Symptoms\*\* dataset containing approximately 247,000 rows and 773 disease classes.



The raw dataset contains:



\* 246,945 rows

\* 378 columns

\* 773 disease classes

\* 377 symptom columns

\* Binary symptom representation

\* No missing values



After selecting the 15 V1 disease classes and harmonizing the features:



```text

12,098 rows

26 clinical features

1 target column

```



The resulting dataset is stored as:



```text

v1\_standardized.csv

```



\## Important Dataset Limitations



The dataset contains substantial repetition and structural issues.



The audit found:



\* 57,298 excess exact duplicate rows

\* 94,067 rows belonging to duplicate groups

\* 77,057 excess duplicate symptom vectors

\* 121,344 rows sharing a symptom vector with another row

\* 12,634 symptom vectors mapping to multiple diseases

\* 39,728 rows affected by conflicting symptom vectors



There were also rare features, all-zero columns, suspicious feature names, and possible leakage-like columns.



Therefore, the dataset is treated as a \*\*development dataset\*\*, not as clinical ground truth.



\---



\# 4. Leakage-Safe Data Splitting



A normal random row split would have produced misleading results because many patients/rows share identical symptom vectors.



Instead, the complete 26-feature symptom vector was used as the grouping key.



The resulting split is approximately:



```text

Training:    8,401 rows

Validation:  1,816 rows

Test:        1,881 rows

```



The following overlap checks were performed:



```text

Train ∩ Validation = 0

Train ∩ Test       = 0

Validation ∩ Test  = 0

```



This prevents identical symptom vectors from appearing across different splits.



The test set was kept untouched during model selection.



\---



\# 5. Machine Learning



Several baseline models were evaluated:



\* Logistic Regression

\* Decision Tree

\* Random Forest

\* K-Nearest Neighbors



\## Baseline Results



| Model               |  Top-1 | Macro F1 |  Top-3 |  Top-5 |

| ------------------- | -----: | -------: | -----: | -----: |

| Logistic Regression | 75.54% |     \~74% | 95.53% | 98.78% |

| Decision Tree       | 71.29% |     \~63% | 71.13% | 89.90% |

| Random Forest       | 70.23% |     \~68% | 91.76% | 97.82% |

| KNN                 | 58.16% |     \~56% | 78.15% | 81.02% |



Logistic Regression was selected for further tuning.



\---



\# 6. Final Model



The final selected model is:



```python

LogisticRegression(

&#x20;   C=8,

&#x20;   solver="lbfgs",

&#x20;   class\_weight="balanced",

&#x20;   max\_iter=5000,

&#x20;   random\_state=42

)

```



The model was trained on the combined training and validation sets after model selection.



The final model is stored as:



```text

models/final\_logistic\_regression.pkl

```

The model artifact is stored in `models/` and should be included in source-control commits. `src/predict.py` loads it relative to its own location, so prediction does not depend on the current working directory. Install runtime dependencies with `python -m pip install -r requirements.txt`.

To regenerate the artifact from `data/processed/`, run `python scripts/final_evaluate_model.py` from the repository root. This retrains the final model on the training and validation splits and writes `models/final_logistic_regression.pkl` and `data/evaluation/final_test_predictions.csv`.



\---



\# 7. Final Held-Out Results



The final model was evaluated once on the untouched test set.



| Metric           |     Result |

| ---------------- | ---------: |

| Accuracy / Top-1 | \*\*76.34%\*\* |

| Macro F1         | \*\*74.84%\*\* |

| Weighted F1      | \*\*76.98%\*\* |

| Top-3            | \*\*95.75%\*\* |

| Top-5            | \*\*99.10%\*\* |



These results are \*\*specific to this dataset, preprocessing pipeline, and evaluation split\*\*.



They should not be interpreted as clinical diagnostic accuracy.



\## Stronger-performing classes



Examples of stronger test F1 scores:



\* Pulmonary hypertension: 0.981

\* Angina: 0.939

\* Pulmonary embolism: 0.888

\* Heart failure: 0.875

\* Pneumonia: 0.874



\## More difficult classes



Examples of weaker test F1 scores:



\* Pneumothorax: 0.463

\* Flu: 0.610

\* Pleural effusion: 0.625

\* Acute bronchitis: 0.645

\* Laryngitis: 0.678

\* COPD: 0.678



Pneumothorax is particularly interesting: recall reached 1.00, but precision was only about 0.30, meaning the model caught the test cases but generated many false positives.



\---



\# 8. Prediction Interface



A standalone prediction interface was created in:



```text

predict.py

```



It:



1\. Loads the trained model

2\. Defines the exact feature order

3\. Validates the input features

4\. Runs `predict\_proba()`

5\. Sorts the results

6\. Returns a ranked differential



Example model output:



```text

1\. asthma              0.9709

2\. heart failure       0.0141

3\. pulmonary embolism  0.0137

4\. pneumonia           0.0005

5\. acute bronchitis    0.0005

```



These values represent model outputs from the current dataset.



They are \*\*not clinically calibrated probabilities\*\*.



The future UI should therefore use terminology such as:



> Model-ranked possibilities



rather than:



> You have a 97.09% chance of asthma.



\---



\# 9. Medical Terminology System



The terminology layer is designed to bridge natural language and the canonical ML feature representation.



The current vocabulary is:



```text

terminology\_v1.json

```



It contains:



```text

31 concepts

292 indexed phrases

```



Each concept contains:



```json

{

&#x20; "concept\_id": "...",

&#x20; "preferred\_term": "...",

&#x20; "synonyms": \[],

&#x20; "definition": "...",

&#x20; "related\_terms": \[],

&#x20; "ml\_feature\_mapping": {}

}

```



Example:



```json

{

&#x20; "concept\_id": "shortness\_of\_breath",

&#x20; "preferred\_term": "Shortness of breath",

&#x20; "synonyms": \[

&#x20;   "shortness of breath",

&#x20;   "breathlessness",

&#x20;   "difficulty breathing"

&#x20; ],

&#x20; "definition": "A subjective sensation of difficult, uncomfortable, or insufficient breathing.",

&#x20; "related\_terms": \[

&#x20;   "orthopnea",

&#x20;   "difficulty\_breathing",

&#x20;   "apnea",

&#x20;   "wheezing"

&#x20; ],

&#x20; "ml\_feature\_mapping": {

&#x20;   "shortness\_of\_breath": 1

&#x20; }

}

```



The vocabulary also contains concepts that do not directly correspond to an ML feature.



For example, generic `chest\_pain` is kept separate from:



\* `sharp\_chest\_pain`

\* `pain\_with\_breathing`

\* `chest\_tightness`



This prevents the system from inventing clinical specificity that the user did not provide.



\---



\# 10. Terminology Matcher



The main terminology logic is implemented in:



```text

terminology\_matcher.py

```



The matcher currently supports:



\* Text normalization

\* Medical phrase matching

\* Synonyms

\* Aliases

\* Negation

\* Uncertainty

\* Common abbreviations

\* Concept-to-feature conversion

\* Repeated concept handling



The concept-state system is:



```text

YES     → 1

NO      → 0

UNKNOWN → None

```



UNKNOWN is intentionally preserved.



\## Negation Handling



The matcher uses local context and clause boundaries to avoid applying negation too broadly.



For example:



```text

"I don't have fever but I have a cough."

```



becomes:



```text

fever → NO

cough → YES

```



rather than incorrectly marking both as negative.



\## Uncertainty



For example:



```text

"I'm not sure if I have a fever."

```



produces:



```text

fever → UNKNOWN

```



rather than YES or NO.



\---



\# 11. Terminology Testing



A 20-case edge-case test suite was created.



The latest result:



```text

20 / 20 semantically correct

```



The tests cover:



\* Positive symptoms

\* Direct negation

\* “denies” constructions

\* Uncertainty

\* “maybe”

\* Medical aliases

\* Dyspnea

\* Dyspnea on exertion

\* Productive cough

\* Hemoptysis

\* Nasal congestion

\* SOB abbreviation

\* CP abbreviation

\* Negation followed by a positive clause

\* Repeated mentions



Examples:



```text

"I don't have fever but I have a cough."

→ fever NO, cough YES



"SOB when walking, denies CP."

→ shortness of breath YES, chest pain NO



"I'm not sure if I have a fever."

→ fever UNKNOWN



"I've been coughing up blood."

→ hemoptysis YES

```



The current matcher should be considered a \*\*V1 deterministic normalization system\*\*, not a production-grade clinical NLP engine.



\---



\# 12. Planned Search System



The next development phase is search.



There will be two complementary modes.



\## Medical Term



Designed for users who already know what they are looking for.



Example:



```text

User:

orthop

```



Possible result:



```text

Orthopnea

Difficulty breathing when lying flat

```



The search should prioritize:



1\. Exact match

2\. Prefix match

3\. Strong keyword match

4\. Synonym match



This path should remain fast and should not require an LLM.



\## Describe It



Designed for natural-language descriptions.



Example:



```text

I have difficulty breathing when I lie down.

```



The system should identify candidate concepts such as:



```text

Orthopnea

Shortness of breath

```



The user can then confirm the intended concept.



The two modes should eventually converge into the same structured concept representation.



\---



\# 13. Planned Full Architecture



The intended application architecture is:



```text

&#x20;                ┌───────────────────┐

&#x20;                │    React UI       │

&#x20;                └─────────┬─────────┘

&#x20;                          │

&#x20;             ┌────────────┴────────────┐

&#x20;             │                         │

&#x20;      Medical Term                 Describe It

&#x20;             │                         │

&#x20;      Search Engine             Terminology/NLP

&#x20;             │                         │

&#x20;             └────────────┬────────────┘

&#x20;                          │

&#x20;                   Clinical Concepts

&#x20;                          │

&#x20;                   User Confirmation

&#x20;                          │

&#x20;                   Structured Features

&#x20;                          │

&#x20;                   ┌──────▼──────┐

&#x20;                   │     ML      │

&#x20;                   │ Differential│

&#x20;                   └──────┬──────┘

&#x20;                          │

&#x20;                   Ranked Results

&#x20;                          │

&#x20;                   ┌──────▼──────┐

&#x20;                   │ Explanation │

&#x20;                   │ / Education │

&#x20;                   └─────────────┘

```



Future LLM integration will sit around the normalization/explanation layer rather than replacing the ML model.



\---



\# 14. Planned Technology Stack



\### Frontend



```text

React

```



\### Backend



```text

Python

FastAPI

```



\### Machine Learning



```text

pandas

NumPy

scikit-learn

matplotlib

```



\### Terminology



```text

JSON

Deterministic matching

```



\### Future



```text

LLM API

Embeddings

Vector database if justified

RAG if justified

```



These technologies will be introduced progressively rather than all at once.



\---



\# 15. MeSH



The 2026 MeSH RDF dataset was downloaded from the U.S. National Library of Medicine.



The archive is approximately 1.9 GB compressed.



It is currently stored as:



```text

mesh2026.nt.gz

```



It has \*\*not\*\* been integrated into the application yet.



The planned use is potentially:



\* Preferred medical terms

\* Synonyms / entry terms

\* Concept identifiers

\* Basic hierarchy

\* Definitions/descriptions where appropriate



MeSH is \*\*not a prerequisite for V1 search\*\*.



The current project should first establish a working search system using the existing terminology vocabulary.



\---



\# 16. Important Limitations



This project currently has several major limitations.



\### Dataset limitations



The main dataset:



\* Has uncertain provenance

\* Contains many duplicate patterns

\* Contains conflicting symptom vectors

\* Contains possible leakage-like features

\* Lacks demographics

\* Lacks clinical history

\* Lacks duration/onset

\* Lacks severity

\* Lacks many real-world measurements



\### ML limitations



The current model:



\* Has not been externally validated

\* Has not been clinically validated

\* Has not been probability-calibrated

\* Is trained on a limited disease scope

\* Uses a simplified binary feature representation



\### NLP limitations



The terminology matcher:



\* Uses deterministic rules

\* Does not fully understand complex clinical syntax

\* Does not perform robust temporal reasoning

\* Does not resolve every ambiguous medical phrase

\* Does not yet handle every possible clinical abbreviation



\### Application limitations



The complete React/FastAPI/LLM system is not yet finished.



Therefore, this repository represents an \*\*ongoing prototype\*\*, not a finished medical product.



\---



\# 17. Safety / Intended Use



This project is intended for:



\* Educational experimentation

\* Machine-learning research

\* Clinical NLP experimentation

\* Software engineering practice

\* Exploring decision-support interfaces



It is \*\*not intended for\*\*:



\* Autonomous diagnosis

\* Emergency decision-making

\* Treatment recommendations

\* Replacing physicians or other healthcare professionals

\* Real-world clinical deployment



The model's predictions are generated from a research dataset and should not be treated as medical advice.



\---



\# 18. Repository Structure



The current V1 layout is described in the \"Repository layout and commands\" section above. The historic sketch below is superseded and retained only as an earlier roadmap note:



```text

medical-ai-clinical-support/

│

├── README.md

│

├── data/

│   └── ...

│

├── ml/

│   ├── harmonize\_v1.py

│   ├── make\_split.py

│   ├── predict.py

│   └── ...

│

├── terminology/

│   ├── terminology\_v1.json

│   └── terminology\_matcher.py

│

├── models/

│   └── ...

│

├── frontend/

│   └── ...

│

├── backend/

│   └── ...

│

└── tests/

&#x20;   └── ...

```



Future frontend and backend work should be added without changing the V1 runtime layout.



\---



\# 19. Current Development Roadmap



The remaining planned work is:



\### Phase 6 — Search



\* \[ ] Medical Term search

\* \[ ] Prefix matching

\* \[ ] Keyword matching

\* \[ ] Synonym ranking

\* \[ ] Describe It integration

\* \[ ] Search result schema

\* \[ ] Ambiguity handling



\### Phase 7 — React



\* \[ ] Main interface

\* \[ ] Medical Term mode

\* \[ ] Describe It mode

\* \[ ] Concept confirmation

\* \[ ] YES / NO / UNKNOWN interface

\* \[ ] Follow-up questions

\* \[ ] Differential results



\### Phase 8 — FastAPI



\* \[ ] API structure

\* \[ ] Search endpoint

\* \[ ] Terminology endpoint

\* \[ ] Prediction endpoint

\* \[ ] Request validation

\* \[ ] Error handling



\### Phase 9 — LLM



\* \[ ] Controlled natural-language extraction

\* \[ ] Explanation generation

\* \[ ] Terminology assistance

\* \[ ] Guardrails

\* \[ ] Structured output validation



\### Phase 10 — Optional RAG



\* \[ ] Knowledge retrieval

\* \[ ] Source grounding

\* \[ ] Retrieved-context generation

\* \[ ] Evaluation



\### Phase 11 — Evaluation



\* \[ ] Search evaluation

\* \[ ] NLP normalization evaluation

\* \[ ] ML evaluation

\* \[ ] End-to-end cases

\* \[ ] Error analysis

\* \[ ] Safety analysis



\### Phase 12 — UI Polish



\* \[ ] Visual design

\* \[ ] Responsiveness

\* \[ ] Loading states

\* \[ ] Error states

\* \[ ] Accessibility

\* \[ ] Final presentation



\---



\# 20. Current Project Milestone



The project has reached an important checkpoint:



```text

Dataset

&#x20;  ↓

Harmonization

&#x20;  ↓

Leakage-safe split

&#x20;  ↓

ML baseline

&#x20;  ↓

Model tuning

&#x20;  ↓

Final model

&#x20;  ↓

Prediction interface

&#x20;  ↓

Terminology vocabulary

&#x20;  ↓

Terminology matcher

&#x20;  ↓

20/20 edge-case tests

&#x20;  ↓

NEXT: Search

```



The ML foundation is therefore already functional.



The immediate priority is \*\*not more model tuning\*\* and \*\*not rewriting the terminology matcher\*\*.



The next implementation task is to build the search layer around the existing terminology system.



\---



\## Disclaimer



This repository contains a research/learning prototype and is not a medical device or autonomous diagnostic system.



Model results are specific to the datasets and evaluation procedures used in this project. They have not been externally or clinically validated.



Do not use the predictions generated by this project as a substitute for professional medical judgment.



