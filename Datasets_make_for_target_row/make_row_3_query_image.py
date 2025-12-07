# -- 이 코드는 특정 행 번호에 해당하는 thermal query HDF5 파일에서 쿼리 이미지를 추출하여 PNG 형식으로 저장합니다. --
# output example:
# [Row 3131] 모든 이미지 저장 완료 → t_datasets/3131_query_images/      

import h5py
import os
import cv2  # OpenCV로 이미지 저장
import sys

# ===== Target Row 받기 =====
if len(sys.argv) < 2:
    raise ValueError("❌ target number (row) 인자가 필요합니다! 예: python3 make_row_1_query.py 2276")

try:
    target_row = int(sys.argv[1])
except ValueError:
    raise ValueError(f"❌ 잘못된 target number 입력: {sys.argv[1]}")

# 입력 HDF5 경로: t_datasets/{target_row}_datasets/test_queries.h5
query_h5_path = f"t_datasets/{target_row}_datasets/test_queries.h5"

# 출력 폴더: t_datasets/{target_row}_query_images/
output_dir = f"t_datasets/{target_row}_query_images"
os.makedirs(output_dir, exist_ok=True)

with h5py.File(query_h5_path, "r") as f:
    query_names = [n.decode("utf-8") for n in f["image_name"][:]]
    image_data = f["image_data"]  # HDF5 dataset (lazy loading)

    for i, name in enumerate(query_names):
        img = image_data[i]  # (H,W,C) numpy 배열
        # HDF5에서 RGB가 BGR 순서일 수 있으니 OpenCV 저장 전에 변환
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 안전한 파일명 (쿼리 이름 그대로 저장)
        filename = f"{name}.png"
        save_path = os.path.join(output_dir, filename)

        cv2.imwrite(save_path, img_bgr)

        if i % 500 == 0:  # 중간 진행 상황 표시
            print(f"{i}/{len(query_names)} saved")

print(f"[Row {target_row}] 모든 이미지 저장 완료 → {output_dir}")
