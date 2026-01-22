import cv2
import os
from ultralytics import YOLO

#################################

MODEL = 'yolo.v8.nano-face.pt'

model = YOLO(MODEL)

#################################

image_files = sorted([
    f for f in os.listdir('images')  # image dir
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if not image_files:
    raise RuntimeError('No images')

#################################
#################################

for filename in image_files:
    path = os.path.join('images', filename)
    img = cv2.imread(path)

    if img is None:
        continue

    results = model(img)
    faces_detected = False

    # Face detection
    for r in results:
        for box in r.boxes:
            faces_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Not detected:
    if not faces_detected:
        cv2.putText(
            img,
            'No detected',
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

    # print filename
    cv2.putText(
        img,
        filename,
        (30, img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow('Face detection', img)

    # Keys
    while True:
        key = cv2.waitKey(0)
        if key in (13, 32):   # Enter or Space
            break
        if key == 27:        # Esc 
            cv2.destroyAllWindows()
            exit(0)

#################################
#################################

cv2.destroyAllWindows()

print(f'Job finished')
