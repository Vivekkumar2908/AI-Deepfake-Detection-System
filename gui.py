import os
import re
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk, ImageOps


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICT_SCRIPT = os.path.join(
    PROJECT_DIR,
    "predict.py"
)


# ============================================================
# COLORS
# ============================================================

BG = "#10141c"
CARD = "#181e29"
CARD2 = "#202735"

TEXT = "#f4f7fb"
MUTED = "#9aa6b5"

ACCENT = "#4f8cff"
SUCCESS = "#2ecc71"
DANGER = "#ff4d5d"


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "DeepGuard - AI Deepfake Detection"
)

root.geometry(
    "1000x700"
)

root.minsize(
    900,
    650
)

root.configure(
    bg=BG
)


# ============================================================
# VARIABLES
# ============================================================

selected_image = None
preview_image = None


# ============================================================
# SELECT IMAGE
# ============================================================

def select_image():

    global selected_image
    global preview_image

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp *.webp"
            ),
            (
                "All Files",
                "*.*"
            )
        ]
    )

    if not file_path:
        return

    selected_image = file_path

    try:

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            file_path
        )

        # ----------------------------------------------------
        # FIX PHONE/CAMERA EXIF ORIENTATION
        # ----------------------------------------------------

        image = ImageOps.exif_transpose(
            image
        )

        # Convert to RGB
        image = image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # Resize preview
        # ----------------------------------------------------

        image.thumbnail(
            (480, 400)
        )

        preview_image = ImageTk.PhotoImage(
            image
        )

        # ----------------------------------------------------
        # Show image
        # ----------------------------------------------------

        image_label.config(
            image=preview_image,
            text=""
        )

        filename_label.config(
            text=os.path.basename(
                file_path
            )
        )

        # ----------------------------------------------------
        # Reset result
        # ----------------------------------------------------

        result_label.config(
            text="READY TO ANALYZE",
            fg=MUTED
        )

        confidence_label.config(
            text="Confidence: --"
        )

        status_label.config(
            text="Image selected. Click Analyze Image."
        )

    except Exception as error:

        messagebox.showerror(
            "Image Error",
            f"Unable to open image:\n\n{error}"
        )


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image():

    if not selected_image:

        messagebox.showwarning(
            "No Image",
            "Please select an image first."
        )

        return

    if not os.path.isfile(
        PREDICT_SCRIPT
    ):

        messagebox.showerror(
            "Error",
            "predict.py was not found."
        )

        return

    # --------------------------------------------------------
    # Disable button while analyzing
    # --------------------------------------------------------

    analyze_button.config(
        state="disabled",
        text="ANALYZING..."
    )

    result_label.config(
        text="ANALYZING...",
        fg=MUTED
    )

    confidence_label.config(
        text="Confidence: --"
    )

    status_label.config(
        text="AI model is analyzing the image..."
    )

    root.update_idletasks()

    try:

        # ----------------------------------------------------
        # Run predict.py
        # ----------------------------------------------------

        process = subprocess.run(
            [
                sys.executable,
                PREDICT_SCRIPT,
                selected_image
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR
        )

        output = process.stdout

        error_output = process.stderr

        # ----------------------------------------------------
        # Check prediction error
        # ----------------------------------------------------

        if process.returncode != 0:

            raise RuntimeError(
                error_output.strip()
                or output.strip()
                or "Prediction failed."
            )

        # ----------------------------------------------------
        # Extract prediction
        # ----------------------------------------------------

        prediction_match = re.search(
            r"Prediction:\s*(FAKE|REAL)",
            output,
            re.IGNORECASE
        )

        # ----------------------------------------------------
        # Extract confidence
        # ----------------------------------------------------

        confidence_match = re.search(
            r"Confidence:\s*([\d.]+)%",
            output,
            re.IGNORECASE
        )

        if not prediction_match:

            raise RuntimeError(
                "Could not read prediction from predict.py."
            )

        prediction = (
            prediction_match
            .group(1)
            .upper()
        )

        if confidence_match:

            confidence = (
                confidence_match
                .group(1)
            )

        else:

            confidence = "--"

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        if prediction == "FAKE":

            result_label.config(
                text="⚠  FAKE IMAGE",
                fg=DANGER
            )

            status_label.config(
                text=(
                    "The AI model detected "
                    "this image as potentially fake."
                )
            )

        else:

            result_label.config(
                text="✓  REAL IMAGE",
                fg=SUCCESS
            )

            status_label.config(
                text=(
                    "The AI model classified "
                    "this image as real."
                )
            )

        confidence_label.config(
            text=f"Confidence: {confidence}%"
        )

    except Exception as error:

        result_label.config(
            text="ANALYSIS FAILED",
            fg=DANGER
        )

        confidence_label.config(
            text="Confidence: --"
        )

        status_label.config(
            text="An error occurred during analysis."
        )

        messagebox.showerror(
            "Prediction Error",
            str(error)
        )

    finally:

        analyze_button.config(
            state="normal",
            text="Analyze Image"
        )


# ============================================================
# CLEAR IMAGE
# ============================================================

def clear_image():

    global selected_image
    global preview_image

    selected_image = None
    preview_image = None

    image_label.config(
        image="",
        text="No Image Selected"
    )

    filename_label.config(
        text="Select an image to begin"
    )

    result_label.config(
        text="READY",
        fg=MUTED
    )

    confidence_label.config(
        text="Confidence: --"
    )

    status_label.config(
        text="Waiting for image..."
    )


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=BG
)

header.pack(
    fill="x",
    padx=40,
    pady=(30, 10)
)


title_label = tk.Label(
    header,
    text="DeepGuard",
    font=("Segoe UI", 28, "bold"),
    fg=TEXT,
    bg=BG
)

title_label.pack(
    anchor="w"
)


subtitle_label = tk.Label(
    header,
    text="AI-Based Image Deepfake Detection System",
    font=("Segoe UI", 12),
    fg=MUTED,
    bg=BG
)

subtitle_label.pack(
    anchor="w",
    pady=(2, 0)
)


# ============================================================
# MAIN CONTENT
# ============================================================

main_frame = tk.Frame(
    root,
    bg=BG
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=40,
    pady=20
)


# ============================================================
# LEFT CARD - IMAGE PREVIEW
# ============================================================

left_card = tk.Frame(
    main_frame,
    bg=CARD
)

left_card.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 12)
)


image_title = tk.Label(
    left_card,
    text="IMAGE PREVIEW",
    font=("Segoe UI", 11, "bold"),
    fg=MUTED,
    bg=CARD
)

image_title.pack(
    anchor="w",
    padx=25,
    pady=(25, 10)
)


image_container = tk.Frame(
    left_card,
    bg=CARD2,
    width=480,
    height=400
)

image_container.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=10
)

image_container.pack_propagate(
    False
)


image_label = tk.Label(
    image_container,
    text="No Image Selected",
    font=("Segoe UI", 14),
    fg=MUTED,
    bg=CARD2
)

image_label.pack(
    expand=True
)


filename_label = tk.Label(
    left_card,
    text="Select an image to begin",
    font=("Segoe UI", 10),
    fg=MUTED,
    bg=CARD
)

filename_label.pack(
    pady=(5, 20)
)


# ============================================================
# RIGHT CARD - RESULT
# ============================================================

right_card = tk.Frame(
    main_frame,
    bg=CARD,
    width=350
)

right_card.pack(
    side="right",
    fill="y",
    padx=(12, 0)
)

right_card.pack_propagate(
    False
)


result_title = tk.Label(
    right_card,
    text="DETECTION RESULT",
    font=("Segoe UI", 11, "bold"),
    fg=MUTED,
    bg=CARD
)

result_title.pack(
    pady=(30, 25)
)


result_label = tk.Label(
    right_card,
    text="READY",
    font=("Segoe UI", 25, "bold"),
    fg=MUTED,
    bg=CARD
)

result_label.pack(
    pady=20
)


confidence_label = tk.Label(
    right_card,
    text="Confidence: --",
    font=("Segoe UI", 16, "bold"),
    fg=TEXT,
    bg=CARD
)

confidence_label.pack(
    pady=10
)


separator = tk.Frame(
    right_card,
    height=1,
    bg=CARD2
)

separator.pack(
    fill="x",
    padx=30,
    pady=25
)


info_label = tk.Label(
    right_card,
    text=(
        "Powered by\n"
        "EfficientNet-B0\n\n"
        "Transfer Learning\n"
        "Image Classification"
    ),
    font=("Segoe UI", 11),
    justify="center",
    fg=MUTED,
    bg=CARD
)

info_label.pack(
    pady=10
)


# ============================================================
# BUTTONS
# ============================================================

button_frame = tk.Frame(
    root,
    bg=BG
)

button_frame.pack(
    fill="x",
    padx=40,
    pady=(5, 20)
)


select_button = tk.Button(
    button_frame,
    text="Select Image",
    command=select_image,
    font=("Segoe UI", 11, "bold"),
    fg=TEXT,
    bg=ACCENT,
    activebackground=ACCENT,
    activeforeground=TEXT,
    relief="flat",
    cursor="hand2",
    padx=25,
    pady=12
)

select_button.pack(
    side="left"
)


analyze_button = tk.Button(
    button_frame,
    text="Analyze Image",
    command=analyze_image,
    font=("Segoe UI", 11, "bold"),
    fg=TEXT,
    bg=SUCCESS,
    activebackground=SUCCESS,
    activeforeground=TEXT,
    relief="flat",
    cursor="hand2",
    padx=25,
    pady=12
)

analyze_button.pack(
    side="left",
    padx=12
)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_image,
    font=("Segoe UI", 11),
    fg=TEXT,
    bg=CARD2,
    activebackground=CARD2,
    activeforeground=TEXT,
    relief="flat",
    cursor="hand2",
    padx=25,
    pady=12
)

clear_button.pack(
    side="left"
)


# ============================================================
# STATUS BAR
# ============================================================

status_label = tk.Label(
    root,
    text="Waiting for image...",
    font=("Segoe UI", 9),
    fg=MUTED,
    bg=BG
)

status_label.pack(
    fill="x",
    padx=40,
    pady=(0, 15)
)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()