import os
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_DIR = "/home/himanshu/tumiit/tumiitgait/uploaded_video"
OUTPUT_DIR = "/home/himanshu/tumiit/tumiitgait/frames"

WIDTH = 88
HEIGHT = 128

DEVICE = 0          # GPU
CONF = 0.30

# -----------------------------
# Load model
# -----------------------------
model = YOLO("yolov8n-seg.pt")

kernel = np.ones((3,3), np.uint8)

for root, dirs, files in os.walk(VIDEO_DIR):

    for file in files:

        if not file.endswith(".avi"):
            continue

        video_path = os.path.join(root, file)

        rel = os.path.relpath(root, VIDEO_DIR)

        out_dir = os.path.join(
            OUTPUT_DIR,
            rel,
            os.path.splitext(file)[0]
        )

        os.makedirs(out_dir, exist_ok=True)

        print(f"\nProcessing {video_path}")

        cap = cv2.VideoCapture(video_path)

        frame_id = 1

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(
                frame,
                conf=CONF,
                verbose=False,
                device=DEVICE
            )

            if len(results)==0:
                continue

            r = results[0]

            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()

            boxes = r.boxes

            best = None
            best_area = 0

            for i, box in enumerate(boxes):

                if int(box.cls) != 0:
                    continue

                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()

                area = (x2-x1)*(y2-y1)

                if area > best_area:
                    best_area = area
                    best = i

            if best is None:
                continue

            mask = masks[best]

            mask = (mask*255).astype(np.uint8)

            mask = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=2
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=1
            )

            contours,_ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours)==0:
                continue

            cnt = max(contours,key=cv2.contourArea)

            x,y,w,h = cv2.boundingRect(cnt)

            person = mask[y:y+h,x:x+w]

            scale = min(WIDTH/w,HEIGHT/h)

            nw = max(1,int(w*scale))
            nh = max(1,int(h*scale))

            person = cv2.resize(
                person,
                (nw,nh),
                interpolation=cv2.INTER_NEAREST
            )

            canvas = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)

            ox = (WIDTH-nw)//2
            oy = (HEIGHT-nh)//2

            canvas[
                oy:oy+nh,
                ox:ox+nw
            ] = person

            cv2.imwrite(
                os.path.join(
                    out_dir,
                    f"frame_{frame_id:04d}.png"
                ),
                canvas
            )

            frame_id += 1

        cap.release()

print("\nFinished.")