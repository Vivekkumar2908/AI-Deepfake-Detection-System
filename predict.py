import json
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# EXPERIMENT 2 - BEST MODEL
# Accuracy on full validation set: 90.81%
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "model",
    "deepfake_model_experiment2.pth"
)

CLASSES_PATH = os.path.join(
    PROJECT_DIR,
    "model",
    "classes_experiment2.json"
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. GET CLASS NAME
# ============================================================

def get_class_name(classes, index):

    if isinstance(classes, list):
        return classes[index]

    return classes.get(
        str(index),
        classes.get(index, str(index))
    )


# ============================================================
# 5. LOAD MODEL
# ============================================================

def load_model():

    # --------------------------------------------------------
    # Check classes file
    # --------------------------------------------------------

    if not os.path.isfile(CLASSES_PATH):

        raise FileNotFoundError(
            f"classes_experiment2.json not found:\n{CLASSES_PATH}"
        )

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    if not os.path.isfile(MODEL_PATH):

        raise FileNotFoundError(
            f"Experiment 2 model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load classes
    # --------------------------------------------------------

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        classes = json.load(file)

    print("Classes:", classes)

    # --------------------------------------------------------
    # EfficientNet-B0
    # --------------------------------------------------------

    model = models.efficientnet_b0(
        weights=None
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Experiment 2 was trained with:
    #
    # Dropout = 0.3
    # Linear = 2 classes
    #
    # Therefore the prediction model must have
    # exactly the same classifier structure.
    # --------------------------------------------------------

    model.classifier[1] = nn.Sequential(
        nn.Dropout(
            p=0.3
        ),

        nn.Linear(
            model.classifier[1].in_features,
            2
        )
    )

    # --------------------------------------------------------
    # Load Experiment 2 trained weights
    # --------------------------------------------------------

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    # --------------------------------------------------------
    # Move model to CPU/GPU
    # --------------------------------------------------------

    model = model.to(device)

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    return model, classes


# ============================================================
# 6. PREDICTION FUNCTION
# ============================================================

def predict(image_path):

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.isfile(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, classes = load_model()

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Transform image
    # --------------------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction_index = torch.max(
            probabilities,
            1
        )

    # --------------------------------------------------------
    # Get label
    # --------------------------------------------------------

    label = get_class_name(
        classes,
        prediction_index.item()
    ).upper()

    # --------------------------------------------------------
    # Confidence percentage
    # --------------------------------------------------------

    confidence_percent = (
        confidence.item() * 100
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("==============================")
    print("       DEEPFAKE RESULT")
    print("==============================")

    print(
        f"Prediction: {label}"
    )

    print(
        f"Confidence: {confidence_percent:.2f}%"
    )

    print("==============================")


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    # If image path is provided:
    #
    # python predict.py "image.jpg"
    #
    # Otherwise ask manually.

    if len(sys.argv) > 1:

        image_path = sys.argv[1]

    else:

        image_path = input(
            "Enter image path: "
        ).strip()

        # Remove quotes if user pasted
        # a path surrounded by quotes

        if (
            len(image_path) >= 2
            and image_path[0] == '"'
            and image_path[-1] == '"'
        ):

            image_path = image_path[1:-1]

    try:

        predict(image_path)

    except Exception as error:

        print(
            f"Error: {error}",
            file=sys.stderr
        )

        sys.exit(1)