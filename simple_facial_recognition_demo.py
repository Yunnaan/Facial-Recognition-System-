"""
FACIAL RECOGNITION ATTENDANCE -- SIMPLE CLASSROOM DEMO
========================================================
One file. Runs top to bottom. Three things it does:

    1. REGISTER  -- capture face photos of a student from the webcam
    2. TRAIN     -- teach the model what each registered student looks like
    3. ATTEND    -- recognise faces live and log attendance to a CSV

Run it with:
    python simple_facial_recognition_demo.pyu

A menu appears. Choose 1, 2, or 3.

Requires: pip install opencv-contrib-python
"""

import cv2
import os
import csv
import numpy as np
from datetime import datetime

# ────────────────────────────────────────────────────────────
# CONFIGURATION -- everything the script needs to know, in one place
# ────────────────────────────────────────────────────────────
PHOTOS_FOLDER    = "student_photos"
MODEL_FILE       = "trained_model.yml"
ATTENDANCE_FILE  = "attendance.csv"
FACE_SIZE        = (200, 200)
CONFIDENCE_LIMIT = 80

CAMERA_INDEX = 0          # confirmed by camera_test.py -- index 0 on this laptop is a hidden
                          # infrared camera used for Windows Hello face-unlock, not a usable picture
CAMERA_BACKENDS = [cv2.CAP_DSHOW]

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def open_camera():
    """
    Opens the webcam, trying several connection methods in order until
    one of them actually produces a real picture.

    Why this is necessary: cv2.VideoCapture(0) alone sometimes connects
    to a camera "successfully" (isOpened() returns True) but then hands
    back nothing but black frames forever -- particularly common on
    Windows laptops that have a hidden infrared camera (for Windows
    Hello face-unlock) sharing camera index 0 alongside the ordinary
    colour camera. If that happens here, run camera_test.py on its own
    first -- it checks every index and backend and tells you exactly
    which combination to use, which you can then set in CAMERA_INDEX
    and reorder in CAMERA_BACKENDS above.
    """
    for backend in CAMERA_BACKENDS:
        cam = cv2.VideoCapture(CAMERA_INDEX, backend)

        if not cam.isOpened():
            cam.release()
            continue

        # Warm-up: throw away the first frames, then check whether the
        # frames that follow are an actual picture or just solid black
        for _ in range(10):
            cam.read()

        ok, frame = cam.read()
        if ok and frame is not None and frame.std() > 5:
            return cam   # this backend works -- use it

        cam.release()

    print("\nERROR: Could not get a real picture from any camera backend.")
    print("Run camera_test.py on its own first -- it will tell you exactly")
    print("which camera index and backend actually work on this computer.")
    print("Also check:")
    print("  1. Close any other app using the camera (Zoom, Teams, browser tab).")
    print("  2. Windows Settings -> Privacy & security -> Camera -> make sure")
    print("     'Let desktop apps access your camera' is turned ON.")
    print("  3. Try unplugging and replugging an external webcam.\n")
    return None


def register_student():
    name = input("Student name: ").strip()
    folder = os.path.join(PHOTOS_FOLDER, name)
    os.makedirs(folder, exist_ok=True)

    cam = open_camera()
    if cam is None:
        return
    count = 0

    print("\nA window titled 'Register Student' should now be open.")
    print("CLICK ON THAT WINDOW FIRST so your keyboard presses reach it")
    print("(pressing keys while the terminal is focused does nothing).")
    print("Look at the camera. Press SPACE to capture a photo.")
    print("Capture at least 10 photos. Press Q when done.\n")

    while True:
        ok, frame = cam.read()
        if not ok:
            print("Lost connection to the camera.")
            break

        # Mirror the frame left-right so the preview behaves like a real
        # mirror -- move your right hand, your right hand moves on screen.
        # Without this, the video feels backwards, like looking at someone
        # else instead of yourself. axis=1 means flip horizontally.
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Photos taken: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Click this window, then SPACE=capture  Q=done",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow("Register Student", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and len(faces) > 0:
            x, y, w, h = faces[0]
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, FACE_SIZE)
            count += 1
            cv2.imwrite(os.path.join(folder, f"{count}.jpg"), face)
            print(f"  Saved photo {count}")

        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {count} photos saved for {name}.\n")


# ────────────────────────────────────────────────────────────
# STEP 2 -- TRAIN THE MODEL
# ────────────────────────────────────────────────────────────
def train_model():
    faces = []
    labels = []
    names = []

    if not os.path.exists(PHOTOS_FOLDER):
        print("No students registered yet. Run Register first.")
        return

    for label, student_name in enumerate(sorted(os.listdir(PHOTOS_FOLDER))):
        student_folder = os.path.join(PHOTOS_FOLDER, student_name)
        if not os.path.isdir(student_folder):
            continue

        names.append(student_name)

        for filename in os.listdir(student_folder):
            path = os.path.join(student_folder, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            faces.append(image)
            labels.append(label)

        print(f"  Loaded {student_name} (label {label})")

    if len(faces) == 0:
        print("No photos found. Register a student first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)

    # Save the names list so attendance.py knows which label = which name
    with open("names.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for label, student_name in enumerate(names):
            writer.writerow([label, student_name])

    print(f"\nTraining complete. {len(faces)} photos, {len(names)} students.\n")


# ────────────────────────────────────────────────────────────
# STEP 3 -- TAKE ATTENDANCE
# ────────────────────────────────────────────────────────────
def take_attendance():
    if not os.path.exists(MODEL_FILE):
        print("No trained model found. Run Train first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    names = {}
    with open("names.csv", "r") as f:
        for row in csv.reader(f):
            names[int(row[0])] = row[1]

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["name", "time"])

    already_marked = set()
    with open(ATTENDANCE_FILE, "r") as f:
        for row in csv.DictReader(f):
            already_marked.add(row["name"])

    cam = open_camera()
    if cam is None:
        return
    print("\nA window titled 'Attendance' should now be open.")
    print("CLICK ON THAT WINDOW FIRST so your keyboard presses reach it.")
    print("Looking for faces. Press Q to stop.\n")

    while True:
        ok, frame = cam.read()
        if not ok:
            print("Lost connection to the camera.")
            break

        # Mirror the frame the same way as registration, so a student
        # looking at themselves during attendance sees the same natural
        # mirror view they saw while being registered.
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            label, confidence = recognizer.predict(face)

            if confidence < CONFIDENCE_LIMIT:
                name = names.get(label, "Unknown")
                colour = (0, 255, 0)

                if name not in already_marked:
                    with open(ATTENDANCE_FILE, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [name, datetime.now().strftime("%H:%M:%S")]
                        )
                    already_marked.add(name)
                    print(f"  Marked present: {name}")
            else:
                name = "Unknown"
                colour = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, f"{name} ({int(confidence)})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\nAttendance saved to {ATTENDANCE_FILE}\n")


# ────────────────────────────────────────────────────────────
# MENU
# ────────────────────────────────────────────────────────────
def main():
    while True:
        print("=" * 40)
        print("FACIAL RECOGNITION ATTENDANCE -- DEMO")
        print("=" * 40)
        print("1. Register a student")
        print("2. Train the model")
        print("3. Take attendance")
        print("4. Exit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            register_student()
        elif choice == "2":
            train_model()
        elif choice == "3":
            take_attendance()
        elif choice == "4":
            break
        else:
            print("Please enter 1, 2, 3, or 4.\n")


if __name__ == "__main__":
    main()