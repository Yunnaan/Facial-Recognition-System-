# Facial Recognition System

## Overview

This project is a Python-based facial recognition system built with OpenCV. It explores computer vision techniques to detect and recognize human faces using a trained recognition model. A real-time facial recognition system that detects faces, extracts embeddings, and matches them against a database of enrolled identities.

The project includes a facial recognition demo, a camera testing script, and a trained model for recognition. 

## Features

* **Face Detection** — Detects human faces using computer vision techniques.
* **Face Recognition** — Recognizes faces using a trained model.
* **Real-Time Camera Testing** — Supports webcam testing through Python scripts.
* **Trained Model Support** — Uses a saved model file (`trained_model.yml`).
* **Python and OpenCV** — Implements facial recognition using Python and OpenCV.
* **VS Code Development** — Developed and tested in Visual Studio Code.
* **Privacy Protection** — Excludes attendance records, names, and face photos from the GitHub repository using `.gitignore`.

## Technologies Used

* **Python** — Core programming language.
* **OpenCV** — Computer vision and image processing.
* Visual Studio Code (VS Code) — Development environment used to write and test the project.

## Project Structure

```text
Facial-recognition/
├── camera_test.py
├── simple_facial_recognition_demo.py
├── Untitled-1.ipynb
├── trained_model.yml
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

* Python 3.9 or later.
* pip package manager.
* A webcam for camera-based testing.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Yunnaan/Facial-recognition.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Facial-recognition
   ```

3. Install the required dependencies:

   ```bash
   pip install opencv-contrib-python numpy
   ```

   Depending on the implementation, additional dependencies may be required.

## Usage

### Test the Camera

Run:

```bash
python camera_test.py
```

### Run the Facial Recognition Demo

Run:

```bash
python simple_facial_recognition_demo.py
```
Press Space bar to capture an image.

Press Q to quit the demo.

Ensure your webcam is connected and accessible. If the program requires additional files or configuration, provide them before running it.

## Privacy and Data Protection

Attendance records, names, and personal face images are excluded from this repository using `.gitignore`.

Please ensure that any facial images or biometric data used during testing are collected and handled with appropriate permission and privacy safeguards.

## Future Improvements

* Improve recognition accuracy under different lighting conditions.
* Add documentation and testing for different environments.

GitHub: [@Yunnnaan](https://github.com/Yunnaan)
