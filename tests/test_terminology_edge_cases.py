from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terminology_matcher import (
    normalize_clinical_text,
)  # Import the text-normalization function being tested.


# List of edge cases designed to stress negation, uncertainty, and synonym detection.
TEST_CASES = [
    "The patient has fever and cough.",
    "The patient has fever but no cough.",
    "The patient has fever, denies chest pain, and has shortness of breath.",
    "The patient is unsure whether they have fever.",
    "The patient has progressive shortness of breath with cough, wheezing, and fatigue.",
    "The patient reports dyspnea on exertion.",
    "The patient denies hemoptysis.",
    "The patient has no fever or chills.",
    "The patient has no cough but significant dyspnea.",
    "The patient doesn't have fever but has a cough.",
    "pt c/o SOB + cough, no fever",
    "SOB when walking, denies CP",
    "The patient is coughing up phlegm.",
    "The patient reports coughing up blood.",
    "The patient has pain when taking a deep breath.",
    "The patient has a blocked nose and a runny nose.",
    "The patient is not sure if they have shortness of breath.",
    "The patient does not have difficulty breathing.",
    "The patient has fever, cough, fatigue, and no wheezing.",
    "The patient denies fever but later reports having a fever.",
]


def main():  # Run the edge-case test suite for terminology normalization.
    print("TERMINOLOGY MATCHER EDGE-CASE TEST")  # Print the test title.
    print("=" * 80)  # Print a separator line.

    for i, text in enumerate(TEST_CASES, start=1):  # Loop over each test case.
        print(f"\nTEST {i}")  # Print the case number.
        print("-" * 80)  # Print a separator line for the current case.
        print(f"INPUT: {text}")  # Show the raw text used in the test.

        result = normalize_clinical_text(
            text
        )  # Run the terminology and negation logic on the input.

        print("\nRAW RESULT:")  # Label the output from the function.
        print(result)  # Print the normalized/processed result for inspection.


if __name__ == "__main__":
    main()  # Execute the edge-case suite only when this file is run directly.
