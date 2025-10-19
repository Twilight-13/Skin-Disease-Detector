import pandas as pd

# Path to metadata file
metadata_path = "data_raw_small/metadata.csv"

# Load metadata
df = pd.read_csv(metadata_path)

# Show structure
print("📊 Metadata columns:\n", df.columns.tolist())
print("\n🔍 Sample data:")
print(df.head())

# Count unique diagnoses
print("\n🧠 Diagnosis distribution:")
print(df['diagnosis'].value_counts(dropna=False))
