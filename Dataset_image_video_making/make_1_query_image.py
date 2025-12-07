# query 이미지들을 768×768에서 512×512로 크롭하고 BGR 포맷으로 변환하여 저장
# h5 파일에서 이미지 데이터를 읽어와 처리


import h5py
import os
import cv2
import sys

query_h5_path = "dataset_NewYork/NewYork/test_queries.h5"
output_dir = "dataset_NewYork/query"


os.makedirs(output_dir, exist_ok=True)

with h5py.File(query_h5_path, "r") as f:
    query_names = [n.decode("utf-8") for n in f["image_name"][:]]
    image_data = f["image_data"]

    for i, name in enumerate(query_names):
        img = image_data[i]  # 768×768 이미지
        
        # 중앙 기준으로 512×512 크롭
        h, w = img.shape[:2]
        crop_size = 512
        half = crop_size // 2
        center_y, center_x = h // 2, w // 2
        
        y1 = center_y - half
        y2 = center_y + half
        x1 = center_x - half
        x2 = center_x + half
        
        # 크롭 (768×768 → 512×512)
        img_cropped = img[y1:y2, x1:x2]
        
        # BGR로 변환
        img_bgr = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR)

        # q1_@좌표@좌표.png 형식으로 저장
        filename = f"q{i+1}_{name}.png"  # q1, q2, q3...
        save_path = os.path.join(output_dir, filename)

        cv2.imwrite(save_path, img_bgr)

        if i % 500 == 0:
            print(f"{i}/{len(query_names)} saved")

print(f"모든 이미지 저장 완료 → {output_dir}")