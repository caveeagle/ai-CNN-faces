import cv2
from ultralytics import YOLO

model = YOLO('yolo.v8.nano-face.pt')

test_file = './images/Cave-002.png'

img = cv2.imread(test_file)
if img is None:
    raise FileNotFoundError('File image.png not found')

# 3. Детекция
results = model(img)

# 4. Отрисовка рамок
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # рамка лица
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),
            thickness=2
        )

# 5. Показ результата
cv2.imshow('Face detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
