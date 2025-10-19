import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "data_splits"
BATCH_SIZE = 32
IMG_SIZE = (380, 380)
MODEL_NAME = "skin_disease_model_research.keras"

print("🔹 Loading dataset splits...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

label_col = "diagnosis_2"
path_col = "image_path"
class_names = sorted(train_df[label_col].unique())

print(f"✅ Loaded CSVs: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
print(f"🧬 Classes: {class_names}")

# =========================================================
# CREATE TEST GENERATOR
# =========================================================
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col=path_col,
    y_col=label_col,
    target_size=IMG_SIZE,
    color_mode='rgb',
    class_mode='sparse',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================
print(f"\n🔄 Loading trained model from '{MODEL_NAME}'...")
model = tf.keras.models.load_model(MODEL_NAME)
print("✅ Model loaded successfully!")

model.summary()

# =========================================================
# EVALUATION ON TEST SET
# =========================================================
print("\n🧪 Evaluating on test set...")
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Final Test Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

# =========================================================
# PREDICTIONS & CLASSIFICATION REPORT
# =========================================================
print("\n🔮 Generating predictions...")
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n📊 Classification Report:\n")
print(report)

# Save report to file
with open("classification_report.txt", "w") as f:
    f.write("Classification Report\n")
    f.write("=" * 80 + "\n\n")
    f.write(report)
print("💾 Classification report saved to 'classification_report.txt'")

# =========================================================
# CONFUSION MATRIX
# =========================================================
print("\n📈 Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

# Calculate per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
print("\n📊 Per-Class Accuracy:")
for i, cls in enumerate(class_names):
    print(f"  {cls}: {per_class_acc[i]*100:.2f}%")

# Plot confusion matrix
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title(f"Confusion Matrix - Test Accuracy: {acc*100:.2f}%", fontsize=14, pad=20)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_research.png", bbox_inches='tight', dpi=150)
plt.close()
print("💾 Confusion matrix saved as 'confusion_matrix_research.png'")

# =========================================================
# NORMALIZED CONFUSION MATRIX
# =========================================================
print("\n📈 Generating normalized confusion matrix...")
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12,10))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
plt.title(f"Normalized Confusion Matrix - Test Accuracy: {acc*100:.2f}%", fontsize=14, pad=20)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png", bbox_inches='tight', dpi=150)
plt.close()
print("💾 Normalized confusion matrix saved as 'confusion_matrix_normalized.png'")

# =========================================================
# PER-CLASS PERFORMANCE BAR CHART
# =========================================================
print("\n📊 Generating per-class performance chart...")
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

classes = list(report_dict.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
precision = [report_dict[cls]['precision'] for cls in classes]
recall = [report_dict[cls]['recall'] for cls in classes]
f1 = [report_dict[cls]['f1-score'] for cls in classes]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, precision, width, label='Precision', color='#3498db')
ax.bar(x, recall, width, label='Recall', color='#e74c3c')
ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71')

ax.set_xlabel('Disease Class', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Per-Class Performance Metrics - Overall Accuracy: {acc*100:.2f}%', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig("per_class_performance.png", bbox_inches='tight', dpi=150)
plt.close()
print("💾 Per-class performance chart saved as 'per_class_performance.png'")

# =========================================================
# SUMMARY STATISTICS
# =========================================================
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)
print(f"Model: {MODEL_NAME}")
print(f"Test Samples: {len(test_df)}")
print(f"Number of Classes: {len(class_names)}")
print(f"\nOverall Test Accuracy: {acc*100:.2f}%")
print(f"Overall Test Loss: {loss:.4f}")
print(f"\nMacro Avg Precision: {report_dict['macro avg']['precision']:.4f}")
print(f"Macro Avg Recall: {report_dict['macro avg']['recall']:.4f}")
print(f"Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
print(f"\nWeighted Avg Precision: {report_dict['weighted avg']['precision']:.4f}")
print(f"Weighted Avg Recall: {report_dict['weighted avg']['recall']:.4f}")
print(f"Weighted Avg F1-Score: {report_dict['weighted avg']['f1-score']:.4f}")
print("="*80)

print("\n✅ Evaluation complete! All visualizations saved.")
print("\nGenerated files:")
print("  📄 classification_report.txt")
print("  📊 confusion_matrix_research.png")
print("  📊 confusion_matrix_normalized.png")
print("  📊 per_class_performance.png")