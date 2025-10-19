import pandas as pd

metadata_path = "data_raw_small/metadata.csv"
df = pd.read_csv(metadata_path)

print("🔍 Checking diagnosis columns for completeness:\n")
for col in ['diagnosis_1', 'diagnosis_2', 'diagnosis_3']:
    non_null = df[col].notnull().sum()
    unique = df[col].nunique()
    print(f"{col}: {non_null} non-null, {unique} unique labels")
    print(df[col].value_counts().head(), "\n")

# Let's check how many rows have *any* diagnosis filled
has_diag = df[['diagnosis_1', 'diagnosis_2', 'diagnosis_3']].notnull().any(axis=1).sum()
print(f"\nTotal rows with any diagnosis: {has_diag} / {len(df)}")
