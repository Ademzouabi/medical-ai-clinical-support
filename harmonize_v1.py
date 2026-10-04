import pandas as pd  # Import the pandas library so we can read and manipulate CSV data in a DataFrame.

INPUT_FILE = "Final_Augmented_dataset_Diseases_and_Symptoms.csv\\Final_Augmented_dataset_Diseases_and_Symptoms.csv"  # Name of the source dataset file to read.
OUTPUT_FILE = (
    "v1_standardized.csv"  # Name of the cleaned output CSV file that will be created.
)


# 1. Diseases included in V1  # This section defines the subset of diseases we want to keep in the final version.

selected_diseases = [  # Create a list of disease names that belong to the V1 model scope.
    "pneumonia",  # Include pneumonia in the selected diseases list.
    "acute bronchitis",  # Include acute bronchitis in the selected diseases list.
    "asthma",  # Include asthma in the selected diseases list.
    "chronic obstructive pulmonary disease (copd)",  # Include COPD in the selected diseases list.
    "common cold",  # Include common cold in the selected diseases list.
    "flu",  # Include flu in the selected diseases list.
    "acute sinusitis",  # Include acute sinusitis in the selected diseases list.
    "chronic sinusitis",  # Include chronic sinusitis in the selected diseases list.
    "laryngitis",  # Include laryngitis in the selected diseases list.
    "pulmonary embolism",  # Include pulmonary embolism in the selected diseases list.
    "pneumothorax",  # Include pneumothorax in the selected diseases list.
    "pleural effusion",  # Include pleural effusion in the selected diseases list.
    "heart failure",  # Include heart failure in the selected diseases list.
    "angina",  # Include angina in the selected diseases list.
    "pulmonary hypertension",  # Include pulmonary hypertension in the selected diseases list.
]


# 2. Raw dataset columns → canonical feature names  # This section maps original symptom names in the raw data to standardized names used by the model.


feature_mapping = {  # Define a dictionary that translates each source symptom text to a canonical feature name.
    "shortness of breath": "shortness_of_breath",  # Map this symptom phrase to a normalized column name.
    "difficulty breathing": "difficulty_breathing",  # Map this symptom phrase to a normalized column name.
    "hurts to breath": "pain_with_breathing",  # Map this symptom phrase to a normalized column name.
    "cough": "cough",  # Keep cough as a canonical feature name.
    "coughing up sputum": "productive_cough",  # Map this phrase to a standardized productive cough feature.
    "wheezing": "wheezing",  # Keep wheezing as a canonical feature name.
    "chest tightness": "chest_tightness",  # Map chest tightness to a standardized feature.
    "congestion in chest": "chest_congestion",  # Map chest congestion to a standardized feature.
    "hemoptysis": "hemoptysis",  # Keep hemoptysis as a canonical feature name.
    "apnea": "apnea",  # Keep apnea as a canonical feature name.
    "hoarse voice": "hoarseness",  # Map hoarse voice to hoarseness.
    "sore throat": "sore_throat",  # Map sore throat to a normalized feature name.
    "nasal congestion": "nasal_congestion",  # Map nasal congestion to a standardized feature.
    "sinus congestion": "sinus_congestion",  # Map sinus congestion to a standardized feature.
    "coryza": "coryza",  # Keep coryza as a canonical feature name.
    "sharp chest pain": "sharp_chest_pain",  # Map sharp chest pain to a standardized name.
    "palpitations": "palpitations",  # Keep palpitations as a canonical feature name.
    "irregular heartbeat": "irregular_heartbeat",  # Map irregular heartbeat to a normalized feature.
    "increased heart rate": "increased_heart_rate",  # Map elevated heart rate to a standardized feature.
    "dizziness": "dizziness",  # Keep dizziness as a canonical feature name.
    "fever": "fever",  # Keep fever as a canonical feature name.
    "chills": "chills",  # Keep chills as a canonical feature name.
    "fatigue": "fatigue",  # Keep fatigue as a canonical feature name.
    "feeling ill": "malaise",  # Map feeling ill to malaise.
    "weakness": "general_weakness",  # Map weakness to a standardized feature.
    "vomiting": "vomiting",  # Keep vomiting as a canonical feature name.
}


# 3. Load only the columns we actually need  # Read only the disease column and the raw symptom columns we mapped above to save memory and speed.


columns_to_load = ["diseases"] + list(
    feature_mapping.keys()
)  # Build a list of the exact columns needed from the CSV file.

df = pd.read_csv(  # Read the CSV file into a DataFrame using the selected columns only.
    INPUT_FILE,  # The file path to read from the dataset folder.
    usecols=columns_to_load,  # Restrict the import to the disease and symptom columns we care about.
)

print(
    f"original rows loaded: {len(df):,}"
)  # Print the total number of rows loaded before filtering.


# 4. Keep only V1 diseases  # Remove rows whose disease is not part of the V1 disease subset.


df = df[
    df["diseases"].isin(selected_diseases)
].copy()  # Filter the DataFrame to only rows where disease is in the selected V1 list and make a copy.


print(
    f"rows after disease filtering: {len(df):,}"
)  # Print how many rows remain after selecting only V1 diseases.


# 5. Rename features to canonical names  # Standardize symptom column names into a consistent format for downstream modeling.

df = df.rename(
    columns=feature_mapping
)  # Rename original symptom labels to the canonical feature names from the mapping.

df = df.rename(
    columns={"diseases": "disease"}
)  # Rename the diseases column if needed to ensure it is named disease.


# 6. Put columns in a predictable order  # Reorder the DataFrame so it follows a consistent feature layout.


canonical_features = list(
    feature_mapping.values()
)  # Create a fixed list of canonical feature columns in their mapped order.

df = df[
    canonical_features + ["disease"]
]  # Reorder the DataFrame with all symptom columns first, then disease last.


# 7. Basic validation  # Check the cleaned dataset for missing or unexpected values before saving.

print("\n=== VALIDATION ===")  # Print a section banner for validation output.

print("\nShape:")  # Label the next output as the data shape.
print(df.shape)  # Print the number of rows and columns in the filtered DataFrame.

print("\nMissing values:")  # Label the next output as the missing-value count.
print(
    df.isna().sum().sum()
)  # Count the total number of missing values across all columns.

print(
    "\nUnique values:"
)  # Label the next output as the unique values present in features.
print(
    df[canonical_features].stack().unique()
)  # Show the distinct non-null values across the standardized feature columns.

print("\nDisease counts:")  # Label the next output as the counts of each disease.
print(
    df["disease"].value_counts().sort_index()
)  # Print disease counts sorted alphabetically by disease name.


# 8. Save standardized dataset  # Export the cleaned dataset so it can be used by the model or downstream pipeline.


df.to_csv(
    OUTPUT_FILE, index=False
)  # Save the DataFrame to the output CSV without writing the DataFrame index.

print(
    f"\nSaved successfully to: {OUTPUT_FILE}"
)  # Print a confirmation message showing the output file name.
