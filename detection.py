import cv2
from ultralytics import YOLO

################################################

MIN_FACE_AREA_RATIO = 0.05  

MODEL_PATH = 'yolo.v8.nano-face.pt'

TEST_FILE = './images/Cave-003.png'

model = YOLO(MODEL_PATH)

################################################

img = cv2.imread(TEST_FILE)
if img is None:
    raise FileNotFoundError('Image file not found')

img_h, img_w = img.shape[:2]
img_area = img_h * img_w           

################################################

results = model(img)

max_box = None
max_area = 0

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area > max_area:
            max_area = area
            max_box = (x1, y1, x2, y2)

################################################

if max_box is None:
    print('No detected: no faces found')
    exit(0)

area_ratio = max_area / img_area

if area_ratio < MIN_FACE_AREA_RATIO:
    print(f'No detected: face too small ({area_ratio:.3f})')
    exit(0)

################################################

x1, y1, x2, y2 = max_box
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Face detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################################

print('Job finished')

################################################

