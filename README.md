# 🛡️ DeepGuard – AI-Based Image Deepfake Detection System

DeepGuard is an AI-based image classification system designed to detect whether an image is **REAL** or **FAKE**.

The system uses **EfficientNet-B0 with Transfer Learning** and provides predictions through a simple desktop GUI.

---

## 🚀 Features

- 🧠 AI-based image deepfake detection
- 🔍 REAL / FAKE image classification
- 📊 Confidence score for predictions
- 🖥️ User-friendly desktop GUI
- ⚡ Local image processing
- 🔒 No image upload to external servers
- 📈 Confusion matrix and evaluation metrics
- 🔄 Transfer Learning using EfficientNet-B0

---

## 🧠 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Torchvision | Computer Vision / Model |
| EfficientNet-B0 | Deep Learning Architecture |
| Transfer Learning | Model Training |
| Pillow | Image Processing |
| Tkinter | Desktop GUI |
| NumPy | Numerical Processing |
| Matplotlib | Visualization |
| Scikit-learn | Model Evaluation |

---

## 🏗️ System Architecture

```text
                Input Image
                    │
                    ▼
             Image Preprocessing
                    │
                    ▼
              EfficientNet-B0
                    │
             Transfer Learning
                    │
                    ▼
             Classification Layer
                    │
             ┌──────┴──────┐
             ▼             ▼
           REAL           FAKE
             │             │
             └──────┬──────┘
                    ▼
             Confidence Score
                    │
                    ▼
                GUI Result

📁 Project Structure
AI-Deepfake-Detection-System/
│
├── train.py
├── predict.py
├── evaluate.py
├── gui.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── deepfake_model.pth
│   ├── deepfake_model_experiment1.pth
│   ├── deepfake_model_experiment2.pth
│   ├── classes.json
│   └── classes_experiment2.json
│
└── results/
    ├── metrics.json
    ├── metrics_experiment1.json
    ├── metrics_experiment2.json
    ├── confusion_matrix.png
    ├── confusion_matrix_experiment1.png
    └── confusion_matrix_experiment2.png

The dataset and Python virtual environment are intentionally excluded from the repository.

🧪 Model
The project uses EfficientNet-B0 with Transfer Learning.
The pretrained feature extractor is used as the base model, while the final classification layer is adapted for two classes:
0 → Fake
1 → Real

The final classifier uses:
Dropout
   ↓
Linear Layer
   ↓
2 Classes

📊 Model Performance
The final Experiment 2 model was evaluated on the full validation dataset containing 39,428 images.
Final Results
Metric	Score
Accuracy	90.81%
Precision	90.91%
Recall	90.81%
F1-Score	90.80%


Classification Results
Class	Precision	Recall	F1-Score
Fake	0.93	0.88	0.91
Real	0.89	0.93	0.91


📈 Confusion Matrix
Experiment 2 produced the following confusion matrix:
                 Predicted
                 Fake   Real

Actual Fake     17324   2317
Actual Real      1308  18479

The corresponding visualization is available in:
results/confusion_matrix_experiment2.png

🔬 Experiments
Three model versions were evaluated during development.
Model	Full Validation Accuracy
Baseline	84.37%
Experiment 1	89.44%
Experiment 2	90.81%


Experiment 2 used a learning rate of:
0.0002

with 10 training epochs.
🖥️ GUI
DeepGuard provides a desktop interface where users can:
1. Select an image
2. Preview the image
3. Analyze the image
4. View REAL / FAKE prediction
5. View prediction confidence
6. Clear the current image
The GUI uses the trained Experiment 2 model through predict.py.
⚙️ Installation
1. Clone the repository
git clone https://github.com/vasus5harma/AI-Deepfake-Detection-System.git

2. Enter the project directory
cd AI-Deepfake-Detection-System

3. Create a virtual environment
Windows:
python -m venv venv

4. Activate the environment
PowerShell:
.\venv\Scripts\Activate.ps1

5. Install dependencies
pip install -r requirements.txt

▶️ Run the GUI
After installing the dependencies:
python gui.py

The DeepGuard desktop application will open.
Select an image and click:
Analyze Image

The system will display:
REAL IMAGE

or
FAKE IMAGE

along with the model's confidence score.
🔍 Command-Line Prediction
The prediction system can also be used directly:
python predict.py "path/to/image.jpg"

Example output:
==============================
       DEEPFAKE RESULT
==============================
Prediction: FAKE
Confidence: 99.99%
==============================

🏋️ Training
Training can be performed using:
python train.py

The training configuration can be modified inside train.py.
The original dataset is not included in this repository because of its size.

📊 Evaluation
To evaluate the trained model:
python evaluate.py

Evaluation results are saved inside:
results/

🔐 Privacy
The application performs image analysis locally.
Images selected through the desktop application are processed locally by the trained model and are not intentionally uploaded to an external server by this application.
⚠️ Limitations
DeepGuard is a machine-learning classification system and its predictions are not guaranteed to be correct for every image.
Performance can vary depending on:
- Image quality
- Image resolution
- Compression
- Deepfake generation method
- Dataset characteristics
- Types of manipulation not represented during training
The reported 90.81% accuracy applies to the project's specified validation dataset and should not be interpreted as universal real-world accuracy.
🔮 Future Scope
Possible future improvements include:
- Video deepfake detection
- Face-specific manipulation detection
- Real-time detection
- Transformer-based architectures
- Larger and more diverse datasets
- Explainable AI / heatmap visualization
- Web-based deployment
- Mobile application
- Detection of multiple manipulation types
👨‍💻 Project
Project: AI-Based Image Deepfake Detection System
Application: DeepGuard
Model: EfficientNet-B0
Approach: Transfer Learning
Task: Binary Image Classification
⭐ Acknowledgement
This project was developed as an academic project to study deep learning, transfer learning, computer vision, and image manipulation detection.