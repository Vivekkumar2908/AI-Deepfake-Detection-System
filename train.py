import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


# =========================================================
# SETTINGS - EXPERIMENT 2
# =========================================================

BATCH_SIZE = 16
EPOCHS = 10
IMAGE_SIZE = 224

# 🔥 Experiment 2: Learning Rate increased
LEARNING_RATE = 0.0002

MAX_TRAIN_PER_CLASS = 5000
MAX_VAL_PER_CLASS = 1000

TRAIN_PATH = "dataset/train"
VAL_PATH = "dataset/Validation"

MODEL_PATH = "model/deepfake_model_experiment2.pth"
CLASSES_PATH = "model/classes_experiment2.json"


# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("EXPERIMENT 2 - DEEPFAKE DETECTION TRAINING")
print("=" * 60)
print(f"Device: {device}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"Training per class: {MAX_TRAIN_PER_CLASS}")
print(f"Validation per class: {MAX_VAL_PER_CLASS}")
print("=" * 60)


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# LOAD DATASETS
# =========================================================

print("\nLoading datasets...")

train_dataset_full = datasets.ImageFolder(
    TRAIN_PATH,
    transform=train_transform
)

val_dataset_full = datasets.ImageFolder(
    VAL_PATH,
    transform=val_transform
)

print(f"Full training dataset: {len(train_dataset_full)}")
print(f"Full validation dataset: {len(val_dataset_full)}")

print(f"Classes: {train_dataset_full.classes}")


# =========================================================
# SELECT BALANCED SUBSET
# =========================================================

def get_balanced_subset(dataset, samples_per_class):

    class_indices = {}

    for index, label in enumerate(dataset.targets):

        if label not in class_indices:
            class_indices[label] = []

        class_indices[label].append(index)

    generator = torch.Generator()
    generator.manual_seed(42)

    selected_indices = []

    for label in sorted(class_indices.keys()):

        indices = class_indices[label]

        # Shuffle deterministically
        permutation = torch.randperm(
            len(indices),
            generator=generator
        ).tolist()

        selected = [
            indices[i]
            for i in permutation[:samples_per_class]
        ]

        selected_indices.extend(selected)

    return Subset(dataset, selected_indices)


train_dataset = get_balanced_subset(
    train_dataset_full,
    MAX_TRAIN_PER_CLASS
)

val_dataset = get_balanced_subset(
    val_dataset_full,
    MAX_VAL_PER_CLASS
)

print("\nBalanced subset:")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================================================
# SAVE CLASS NAMES
# =========================================================

os.makedirs("model", exist_ok=True)

with open(CLASSES_PATH, "w") as f:
    json.dump(train_dataset_full.classes, f, indent=4)

print(f"\nClasses saved at:")
print(CLASSES_PATH)


# =========================================================
# LOAD EFFICIENTNET-B0
# =========================================================

print("\nLoading EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=weights)


# =========================================================
# FREEZE FEATURE EXTRACTOR
# =========================================================

for param in model.features.parameters():
    param.requires_grad = False


# Unfreeze last 2 feature blocks
for param in model.features[-2:].parameters():
    param.requires_grad = True


# =========================================================
# REPLACE CLASSIFIER
# =========================================================

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 2)
)

model = model.to(device)


# =========================================================
# LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =========================================================
# LEARNING RATE SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# =========================================================
# TRAINING
# =========================================================

best_accuracy = 0.0

print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)


for epoch in range(EPOCHS):

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss_total / val_total

    val_accuracy = 100 * val_correct / val_total


    # -----------------------------------------------------
    # SCHEDULER
    # -----------------------------------------------------

    scheduler.step(val_accuracy)


    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


    # -----------------------------------------------------
    # SAVE BEST MODEL
    # -----------------------------------------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("\n✅ BEST MODEL SAVED!")


# =========================================================
# TRAINING COMPLETE
# =========================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

print(f"Model saved at:")
print(MODEL_PATH)

print(f"\nClasses saved at:")
print(CLASSES_PATH)

print("\nExperiment 2 completed successfully! 🚀")