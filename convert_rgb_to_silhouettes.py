import os
import cv2
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
VIDEO_DIR = "/home/himanshu/tumiit/tumiitgait/uploaded_video"
OUTPUT_DIR = "/home/himanshu/tumiit/tumiitgait/frames"

FRAME_WIDTH = 88
FRAME_HEIGHT = 128

# MOG2 Background Subtractor
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)

kernel_open = np.ones((3,3), np.uint8)
kernel_close = np.ones((5,5), np.uint8)


def process_frame(frame):

    # -----------------------------
    # Background subtraction
    # -----------------------------
    fg = backSub.apply(frame)

    # -----------------------------
    # Binary threshold
    # -----------------------------
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    # -----------------------------
    # Morphological operations
    # -----------------------------
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)

    # -----------------------------
    # Find contours
    # -----------------------------
    contours, _ = cv2.findContours(
        fg,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 400:
        return None

    x, y, w, h = cv2.boundingRect(largest)

    person = fg[y:y+h, x:x+w]

    # -----------------------------
    # Maintain aspect ratio
    # -----------------------------
    scale = min(
        FRAME_WIDTH / w,
        FRAME_HEIGHT / h
    )

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        person,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros(
        (FRAME_HEIGHT, FRAME_WIDTH),
        dtype=np.uint8
    )

    x_offset = (FRAME_WIDTH - new_w) // 2
    y_offset = (FRAME_HEIGHT - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized

    return canvas


# -------------------------------------
# Process all videos
# -------------------------------------
for root, dirs, files in os.walk(VIDEO_DIR):

    for file in files:

        if not file.endswith(".avi"):
            continue

        video_path = os.path.join(root, file)

        relative_dir = os.path.relpath(root, VIDEO_DIR)

        video_name = os.path.splitext(file)[0]

        output_folder = os.path.join(
            OUTPUT_DIR,
            relative_dir,
            video_name
        )

        os.makedirs(output_folder, exist_ok=True)

        print(f"\nProcessing: {video_path}")

        cap = cv2.VideoCapture(video_path)

        frame_no = 1

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            silhouette = process_frame(frame)

            if silhouette is None:
                continue

            cv2.imwrite(
                os.path.join(
                    output_folder,
                    f"frame_{frame_no:04d}.png"
                ),
                silhouette
            )

            frame_no += 1

        cap.release()

print("\nDone!")