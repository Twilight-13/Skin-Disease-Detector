import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "data_raw_small"
IMG_DIR = DATA_DIR
META_PATH = os.path.join(DATA_DIR, "metadata.csv")

# Load metadata
df = pd.read_csv(META_PATH)

# Keep only images that exist
df["image_path"] = df["isic_id"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))
df = df[df["image_path"].apply(os.path.exists)]

# Use diagnosis_3 as label
df = df.dropna(subset=["diagnosis_3"])
df = df.rename(columns={"diagnosis_3": "label"})

# Simplify label names (optional, short versions)
df["label"] = df["label"].str.replace(", NOS", "", regex=False).str.strip()

# Stratified train/val/test split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# Save CSVs
os.makedirs("data_splits", exist_ok=True)
train_df.to_csv("data_splits/train.csv", index=False)
val_df.to_csv("data_splits/val.csv", index=False)
test_df.to_csv("data_splits/test.csv", index=False)

print("✅ Dataset prepared!")
print("Train:", len(train_df), " | Val:", len(val_df), " | Test:", len(test_df))
print("Classes:", df['label'].nunique(), " | Labels:", sorted(df['label'].unique()))
