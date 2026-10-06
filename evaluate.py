import os
import json
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# =========================================================
# SETTINGS - EXPERIMENT 2
# =========================================================

MODEL_PATH = "model/deepfake_model_experiment2.pth"
CLASSES_PATH = "model/classes_experiment2.json"

VAL_PATH = "dataset/Validation"

IMAGE_SIZE = 224
BATCH_SIZE = 16

CONFUSION_MATRIX_PATH = "results/confusion_matrix_experiment2.png"
METRICS_PATH = "results/metrics_experiment2.json"


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("EXPERIMENT 2 MODEL EVALUATION")
print("=" * 60)

print(f"Device: {device}")


# =========================================================
# CHECK FILES
# =========================================================

if not os.path.exists(MODEL_PATH):
    print(f"\n❌ Model not found:")
    print(MODEL_PATH)
    exit()

if not os.path.exists(CLASSES_PATH):
    print(f"\n❌ Classes file not found:")
    print(CLASSES_PATH)
    exit()


# =========================================================
# LOAD CLASSES
# =========================================================

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

print(f"Classes: {classes}")


# =========================================================
# TRANSFORM
# =========================================================

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# LOAD FULL VALIDATION DATASET
# =========================================================

print("\nLoading FULL validation dataset...")

val_dataset = datasets.ImageFolder(
    VAL_PATH,
    transform=val_transform
)

print(f"Total validation images: {len(val_dataset)}")

print(f"Fake images: {sum(1 for x in val_dataset.targets if x == 0)}")
print(f"Real images: {sum(1 for x in val_dataset.targets if x == 1)}")


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================================================
# LOAD EFFICIENTNET-B0
# =========================================================

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=None
)


# =========================================================
# REPLACE CLASSIFIER
# =========================================================

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 2)
)


# =========================================================
# LOAD EXPERIMENT 2 MODEL
# =========================================================

print("\nLoading Experiment 2 model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# =========================================================
# EVALUATION
# =========================================================

all_predictions = []
all_labels = []

print("\nEvaluating...")

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 60)
print("EXPERIMENT 2 MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=2
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(6, 5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("Experiment 2 - Confusion Matrix")

plt.colorbar()

tick_marks = range(len(classes))

plt.xticks(
    tick_marks,
    classes
)

plt.yticks(
    tick_marks,
    classes
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(len(classes)):
    for j in range(len(classes)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)

plt.close()


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "total_validation_images": len(all_labels),
    "model": MODEL_PATH,
    "classes": classes
}

with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 60)
print("EXPERIMENT 2 EVALUATION COMPLETE")
print("=" * 60)

print(f"\nModel evaluated:")
print(MODEL_PATH)

print("\nConfusion matrix saved at:")
print(CONFUSION_MATRIX_PATH)

print("\nMetrics saved at:")
print(METRICS_PATH)

print("\n✅ DONE!")