import os
import sqlite3

import cv2
from ultralytics import YOLO

import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1

################################################
################################################
################################################

file_model_detected = 'yolo.v8.nano-face.pt'

model_detected = YOLO(file_model_detected)

model_embedding = InceptionResnetV1(pretrained='vggface2').eval()

################################################
################################################
################################################

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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

image_files = sorted([
    f for f in os.listdir('images')  # image dir
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if not image_files:
    raise RuntimeError('No images')


EMBEDDINGS = []

for filename in image_files:
    path = os.path.join('images', filename)
    img = cv2.imread(path)

    if img is None:
        continue

    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w           
    
    ################################################
    ################################################
    
    detected_results = model_detected(img)
    
    best = None
    best_area = 0
    
    for r in detected_results:
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
        continue
    
    MIN_FACE_AREA_RATIO = 0.05  
    
    if best_area / img_area < MIN_FACE_AREA_RATIO:
        print(f'No detected: face too small (ratio={best_area / img_area:.4f})')
        continue
    
    x1, y1, x2, y2, left_eye, right_eye = best
    
    ################################################
    ################################################
    
    CROP_MARGIN_RATIO = 0.25
    
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * CROP_MARGIN_RATIO)
    my = int(bh * CROP_MARGIN_RATIO)
    
    cx1 = clamp(x1 - mx, 0, img_w - 1)
    cy1 = clamp(y1 - my, 0, img_h - 1)
    cx2 = clamp(x2 + mx, 1, img_w)
    cy2 = clamp(y2 + my, 1, img_h)
    
    crop = img[cy1:cy2, cx1:cx2].copy()
    
    ###  Align by eyes:
    
    if left_eye is not None and right_eye is not None:
        # перевод глаз в координаты crop
        leye = (left_eye[0] - cx1, left_eye[1] - cy1)
        reye = (right_eye[0] - cx1, right_eye[1] - cy1)
    
        # если глаза вдруг вне crop — пропускаем выравнивание
        ch, cw = crop.shape[:2]
        if 0 <= leye[0] < cw and 0 <= leye[1] < ch and 0 <= reye[0] < cw and 0 <= reye[1] < ch:
            crop = align_crop_by_eyes(crop, leye, reye)
    
    ################################################
    ################################################
    
    output_size = (112, 112)
    face = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    face_tensor = torch.from_numpy(face_rgb).permute(2,0,1).unsqueeze(0).float()
    face_tensor = face_tensor / 255.0
    
    torch.set_grad_enabled(False)

    print(f'File: {filename}')
    
    embedding = model_embedding(face_tensor)
    
    embedding = embedding.cpu().numpy()[0]
    
    EMBEDDINGS.append((filename,embedding))
    
    ### End of files cycle

################################################
################################################
################################################

db = 'faces.sqlite'

with sqlite3.connect(db) as conn:
    cur = conn.cursor()
    for filename, embedding in EMBEDDINGS:
        emb_blob = embedding.astype(np.float32).tobytes()
        cur.execute(
            "INSERT INTO testing (filename, embedding) VALUES (?, ?)",
            (filename, emb_blob)
        )
    conn.commit()

################################################

print('\n\nJob finished')


