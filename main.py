import os
import sys
import cv2
import json
import copy
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), 'SPIGA'))
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

INPUT_DIR = 'images'
OUTPUT_DIR = 'results'
YOLO_MODEL = './yolov11l-face.pt'
SPIGA_CONFIG = 'wflw'
USE_CPU = True
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

os.makedirs(OUTPUT_DIR, exist_ok=True)

yolo = YOLO(YOLO_MODEL)
spiga = SPIGAFramework(ModelConfig(SPIGA_CONFIG), gpus=[-1 if USE_CPU else 0])

def draw_landmarks(img, landmarks, size=3):
    """
    Vẽ các điểm landmark lên ảnh.
    """
    for (x, y) in landmarks.astype(int):
        cv2.circle(img, (x, y), size + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), size, (0, 255, 255), -1, cv2.LINE_AA)
    return img

def detect_faces_and_landmarks(img):
    """
    Dò khuôn mặt bằng YOLO, lấy landmark từ SPIGA, trả ảnh và dữ liệu.
    """
    img_copy = copy.deepcopy(img)
    results = yolo(img)[0]
    faces = []

    if not results.boxes:
        return img_copy, faces

    bboxes = results.boxes.xyxy.cpu().numpy().astype(int)

    for box in bboxes:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        features = spiga.inference(img, [[x0, y0, w, h]])

        if 'landmarks' in features:
            landmarks = np.array(features['landmarks'][0])
            draw_landmarks(img_copy, landmarks)
            faces.append({
                'bbox': [int(x0), int(y0), int(x1), int(y1)],
                'landmarks': landmarks.tolist()
            })

    return img_copy, faces

def process_images():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXTS)]
    all_data = []

    for name in files:
        path = os.path.join(INPUT_DIR, name)
        img = cv2.imread(path)

        if img is None:
            print(f"Lỗi ảnh: {name}")
            continue

        out_img, face_list = detect_faces_and_landmarks(img)
        out_path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(out_path, out_img)

        if face_list:
            all_data.append({
                'image': name,
                'faces': face_list
            })

    json_path = os.path.join(OUTPUT_DIR, 'landmarks.json')
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(all_data, jf, indent=2)

    print("Xử lý xong")

if __name__ == '__main__':
    process_images()
