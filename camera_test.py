"""
CAMERA DIAGNOSTIC -- FIND OUT WHY YOUR WEBCAM WON'T SHOW A PICTURE
====================================================================
Run this FIRST, on its own, before touching the attendance script again:

    python camera_test.py

It tries every combination of camera index (0, 1, 2) and connection
method (called a "backend" -- DSHOW, MSMF, ANY) and tells you exactly
which combination actually produces a real picture on your machine.

Why this is necessary: a Windows laptop often has MORE THAN ONE camera
device even if you only ever see one lens on the lid. Laptops with
Windows Hello face-unlock frequently have a hidden infrared camera
sitting at index 0, with the real, ordinary colour camera sitting at
index 1. If your code always opens index 0, it may be silently talking
to the WRONG camera -- one that either refuses to hand over frames to
a normal Python program, or was never meant to produce a viewable
picture at all.

This script will not touch your attendance system. It only tests and
reports. Once you know the right index and backend, you tell the main
script which ones to use, and the problem goes away for good.
"""

import cv2
import time

# Every backend worth trying, with a human-readable name for the report
BACKENDS_TO_TRY = [
    ("CAP_DSHOW (DirectShow)", cv2.CAP_DSHOW),
    ("CAP_MSMF (Media Foundation)", cv2.CAP_MSMF),
    ("CAP_ANY (let OpenCV choose)", cv2.CAP_ANY),
]

# Camera indices worth trying -- most laptops have at most 2 or 3 cameras
INDICES_TO_TRY = [0, 1, 2]


def test_one_combination(index, backend_name, backend_value):
    """
    Tries to open one specific camera index with one specific backend.
    Reads a few frames and reports whether real, non-blank pictures
    actually came through.
    """
    print(f"  Trying index={index}  backend={backend_name} ... ", end="", flush=True)

    cam = cv2.VideoCapture(index, backend_value)

    if not cam.isOpened():
        print("FAILED -- could not open at all")
        cam.release()
        return False

    # Give the camera a moment to wake up before judging it
    time.sleep(0.3)

    good_frames = 0
    for _ in range(15):
        ok, frame = cam.read()
        if ok and frame is not None:
            # A frame is only "real" if it is not just a solid black
            # or solid grey image -- check that it actually has some
            # variation in it (a real picture always does)
            if frame.std() > 5:
                good_frames += 1

    cam.release()

    if good_frames >= 5:
        print(f"WORKS -- {good_frames}/15 good frames")
        return True
    elif good_frames > 0:
        print(f"WEAK -- only {good_frames}/15 good frames (might be unreliable)")
        return False
    else:
        print("OPENED BUT BLACK -- camera connects but sends no real picture")
        return False


def main():
    print("Testing every camera index and connection method.")
    print("This takes about 15-20 seconds. Please wait.\n")

    working_combinations = []

    for index in INDICES_TO_TRY:
        for backend_name, backend_value in BACKENDS_TO_TRY:
            works = test_one_combination(index, backend_name, backend_value)
            if works:
                working_combinations.append((index, backend_name, backend_value))

    print()
    print("=" * 60)
    if working_combinations:
        print("RESULT: at least one working combination was found.\n")
        for index, backend_name, backend_value in working_combinations:
            print(f"  index = {index}   backend = {backend_name}")
        print()
        best_index, best_name, best_value = working_combinations[0]
        print(f"Use this in your main script:")
        print(f"  CAMERA_INDEX   = {best_index}")
        print(f"  CAMERA_BACKEND = cv2.{best_name.split()[0]}")
    else:
        print("RESULT: no combination produced a real picture.\n")
        print("This means the problem is not about which index or backend to")
        print("use -- something is blocking Python from reaching any camera.")
        print("Check, in this order:")
        print("  1. Close every other app that might be using the camera:")
        print("     Zoom, Teams, Skype, your browser (check for open tabs),")
        print("     the Windows Camera app, OBS, Discord.")
        print("  2. Windows Settings -> Privacy & security -> Camera:")
        print("     turn ON 'Camera access' and 'Let apps access your camera'")
        print("     and specifically 'Let desktop apps access your camera'.")
        print("  3. If this is an external USB webcam, unplug it, wait 5")
        print("     seconds, and plug it back in.")
        print("  4. Restart the laptop. Camera driver locks in Windows")
        print("     sometimes only clear on a full restart, not just closing apps.")
    print("=" * 60)


if __name__ == "__main__":
    main()
