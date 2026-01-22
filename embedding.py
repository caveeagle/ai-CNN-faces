import cv2
from ultralytics import YOLO

################################################

MIN_FACE_AREA_RATIO = 0.05  

MODEL_PATH = 'yolo.v8.nano-face.pt'

TEST_FILE = './images/Cave-003.png'

################################################

img = cv2.imread(TEST_FILE)
if img is None:
    raise FileNotFoundError('Image file not found')

img_h, img_w = img.shape[:2]
img_area = img_h * img_w           

################################################

model = YOLO(MODEL_PATH)

results = model(img)

################################################
################################################
################################################

best = None
best_area = 0

for r in results:
    if r.boxes is None or len(r.boxes) == 0:
        continue

    has_kps = (getattr(r, "keypoints", None) is not None) and (r.keypoints.xy is not None)

    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area > best_area:
            left_eye = right_eye = None
            if has_kps and i < len(r.keypoints.xy):
                kps = r.keypoints.xy[i]
                # ожидаем: kps[0]=left_eye, kps[1]=right_eye
                left_eye = (float(kps[0][0]), float(kps[0][1]))
                right_eye = (float(kps[1][0]), float(kps[1][1]))

            best_area = area
            best = (x1, y1, x2, y2, left_eye, right_eye)

if best is None:
    print('No detected: no faces found')
    raise SystemExit

if best_area / img_area < MIN_FACE_AREA_RATIO:
    print(f'No detected: face too small (ratio={best_area / img_area:.4f})')
    raise SystemExit

x1, y1, x2, y2, left_eye, right_eye = best

################################################
################################################
################################################

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

################################################

def align_crop_by_eyes(crop_bgr, left_eye_xy, right_eye_xy):
    lx, ly = left_eye_xy
    rx, ry = right_eye_xy

    dx = rx - lx
    dy = ry - ly
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    h, w = crop_bgr.shape[:2]
    aligned = cv2.warpAffine(crop_bgr, M, (w, h), flags=cv2.INTER_LINEAR)
    return aligned

################################################
################################################
################################################

CROP_MARGIN_RATIO = 0.25

# -----------------------------
# CROP С ЗАПАСОМ
# -----------------------------
bw = x2 - x1
bh = y2 - y1
mx = int(bw * CROP_MARGIN_RATIO)
my = int(bh * CROP_MARGIN_RATIO)

cx1 = clamp(x1 - mx, 0, img_w - 1)
cy1 = clamp(y1 - my, 0, img_h - 1)
cx2 = clamp(x2 + mx, 1, img_w)
cy2 = clamp(y2 + my, 1, img_h)

crop = img[cy1:cy2, cx1:cx2].copy()

# -----------------------------
# ВЫРАВНИВАНИЕ ВНУТРИ CROP (если есть глаза)
# -----------------------------
if left_eye is not None and right_eye is not None:
    # перевод глаз в координаты crop
    le = (left_eye[0] - cx1, left_eye[1] - cy1)
    re = (right_eye[0] - cx1, right_eye[1] - cy1)

    # если глаза вдруг вне crop — пропускаем выравнивание
    ch, cw = crop.shape[:2]
    if 0 <= le[0] < cw and 0 <= le[1] < ch and 0 <= re[0] < cw and 0 <= re[1] < ch:
        crop = align_crop_by_eyes(crop, le, re)

################################################
################################################
################################################

OUTPUT_SIZE = (112, 112)

face = cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)

CHECK = 0
if(CHECK):
    print("dtype:", face.dtype)
    print("shape:", face.shape)
    print("min:", face.min())
    print("max:", face.max())
    print("mean:", face.mean())
    raise SystemExit()


#pip install insightface onnxruntime

#import insightface
#
## инициализация (делается ОДИН РАЗ)
#app = insightface.app.FaceAnalysis(
#    name="buffalo_l",
#    providers=["CPUExecutionProvider"]  # или CUDAExecutionProvider
#)
#app.prepare()
#
## === ВОТ ЭТА СТРОКА ДЕЛАЕТ EMBEDDING ===
#embedding = app.models["recognition"].get(face)
#
#print(embedding.shape)   # (512,)





################################################
################################################
################################################

print('Job finished')

################################################

