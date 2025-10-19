import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Force TensorFlow to use correct channel format
tf.keras.backend.set_image_data_format('channels_last')

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "data_splits"
BATCH_SIZE = 32
IMG_SIZE = (380, 380)  # EfficientNetB4 recommended input
EPOCHS = 50
MODEL_NAME = "skin_disease_model_research.keras"

# =========================================================
# LOAD CSV SPLITS
# =========================================================
print("🔹 Loading dataset splits...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

label_col = "diagnosis_2"
path_col = "image_path"
class_names = sorted(train_df[label_col].unique())
num_classes = len(class_names)

print(f"✅ Loaded CSVs: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
print(f"🧬 Classes: {class_names}")


# =========================================================
# IMAGE GENERATORS + AUGMENTATION (FORCE RGB)
# =========================================================
def ensure_rgb(df, path_col):
    """Ensures all image paths are valid RGB images"""
    from PIL import Image
    fixed_paths = []
    for p in df[path_col]:
        try:
            img = Image.open(p).convert("RGB")  # Force convert to RGB
            img.save(p)
            fixed_paths.append(p)
        except Exception as e:
            print(f"⚠️ Skipping unreadable image: {p} ({e})")
    return df[df[path_col].isin(fixed_paths)]


print("🧩 Checking and converting grayscale images to RGB...")
train_df = ensure_rgb(train_df, path_col)
val_df = ensure_rgb(val_df, path_col)
test_df = ensure_rgb(test_df, path_col)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.25,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col=path_col,
    y_col=label_col,
    target_size=IMG_SIZE,
    color_mode='rgb',
    class_mode='sparse',
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col=path_col,
    y_col=label_col,
    target_size=IMG_SIZE,
    color_mode='rgb',
    class_mode='sparse',
    batch_size=BATCH_SIZE
)

test_gen = val_datagen.flow_from_dataframe(
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
# CLASS WEIGHTS
# =========================================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df[label_col]),
    y=train_df[label_col]
)
class_weights = dict(enumerate(class_weights))
print(f"⚖️ Class Weights: {class_weights}")

# =========================================================
# MODEL - ALTERNATIVE APPROACH: Build then load weights
# =========================================================
print("\n🔧 Building model architecture...")
tf.keras.backend.clear_session()

# Method 1: Try with input_shape parameter
try:
    print("Attempting Method 1: Standard approach with input_shape...")
    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights=None,  # Don't load weights yet
        input_shape=(*IMG_SIZE, 3)
    )

    # Now manually download and load weights
    weights_path = tf.keras.utils.get_file(
        'efficientnetb4_notop.h5',
        'https://storage.googleapis.com/keras-applications/efficientnetb4_notop.h5',
        cache_subdir='models',
        file_hash='0e525e3e2e1b2c5e9f9a6e6e0e9e8e7e'
    )

    # Verify the model was built with correct shape
    print(f"Model input shape: {base_model.input_shape}")
    assert base_model.input_shape == (None, 380, 380, 3), "Model has wrong input shape!"

    base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print("✅ Method 1 successful!")

except Exception as e:
    print(f"❌ Method 1 failed: {e}")
    print("\nAttempting Method 2: Manual model construction...")

    # Method 2: Use a different pretrained model
    try:
        from tensorflow.keras import layers, Model

        # Use ResNet50V2 instead (more stable)
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMG_SIZE, 3)
        )
        print("✅ Using ResNet50V2 instead of EfficientNetB4")

    except Exception as e2:
        print(f"❌ Method 2 failed: {e2}")
        print("\nFalling back to training from scratch...")

        # Method 3: Train from scratch with simpler architecture
        base_model = tf.keras.applications.EfficientNetB4(
            include_top=False,
            weights=None,
            input_shape=(*IMG_SIZE, 3)
        )
        print("⚠️ Training EfficientNetB4 from scratch (no pretrained weights)")

base_model.trainable = False  # Phase 1: freeze base

# Build the full model
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Use label smoothing if available, otherwise use standard loss
try:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1)
    print("✅ Using label smoothing (0.1)")
except TypeError:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    print("⚠️ Label smoothing not supported in this TensorFlow version")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=['accuracy']
)

print("\n📋 Model Summary:")
model.summary()

# =========================================================
# CALLBACKS
# =========================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(MODEL_NAME, save_best_only=True)
]

# =========================================================
# TRAINING PHASE 1 (Feature Extraction)
# =========================================================
print("\n🚀 Training phase 1 (frozen base)...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================================================
# FINE-TUNING PHASE 2
# =========================================================
print("\n🔓 Fine-tuning layers...")
base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:int(len(base_model.layers) * 0.8)]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# =========================================================
# TRAINING PLOTS
# =========================================================
def plot_history(histories):
    plt.figure(figsize=(10, 5))
    plt.title("Training & Validation Accuracy")
    for h in histories:
        plt.plot(h.history['accuracy'])
    for h in histories:
        plt.plot(h.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train_phase1', 'train_phase2', 'val_phase1', 'val_phase2'])
    plt.grid(True)
    plt.savefig("training_plot_research.png", bbox_inches='tight')
    plt.show()


plot_history([history, history_fine])

# =========================================================
# EVALUATION
# =========================================================
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Final Test Accuracy: {acc * 100:.2f}% | Loss: {loss:.4f}")

# Per-class accuracy
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n📊 Classification Report:\n", report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_research.png", bbox_inches='tight')
plt.show()

print(f"\n📦 Best Model Saved As: {MODEL_NAME}")