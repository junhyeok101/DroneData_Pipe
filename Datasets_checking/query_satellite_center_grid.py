# query - database 쌍 시각화 및 저장 (수동 좌표 지정 버전)
# 위성이미지와 쿼리 간의 좌표 일치를 수동으로 확인하기 위함. 

# 여기서 테스트 -> 일치 되는 좌표 확인 -> 그 좌표에 가깝게 Dataset_making 재실행. 

import h5py
import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np


sat_path = "dataset_china/maps/satellite/20201117_BingSatellite.png"
row_query_path = "dataset_china/china/test_queries.h5"

# 출력 폴더
save_dir = "dataset_china/center_query_datbase_manual"
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# ✅ 여기에 원하는 좌표 직접 입력 (row, col)
# ============================================================
manual_coords = [
    (940, 4125)  # 첫 번째 샘플
]

# query 인덱스도 직접 지정 (None이면 순서대로 0, 1, 2, ...)
query_indices = None  # 또는 [0, 5, 10, 15, 20] 처럼 직접 지정
# ============================================================

# === 원본 위성 이미지 로드 ===
sat_img = cv2.imread(sat_path)
if sat_img is None:
    print(f"❌ Failed to load satellite image: {sat_path}")
    sys.exit(1)

sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
sat_h, sat_w = sat_img.shape[:2]
print(f"✅ Satellite image loaded: {sat_w}×{sat_h}")

# === query 읽기 ===
with h5py.File(row_query_path, "r") as fq:
    query_names = [n.decode("utf-8") for n in fq["image_name"][:]]
    query_imgs = fq["image_data"][:] 
print(f"✅ Loaded {len(query_names)} query images")

# query 인덱스 설정
if query_indices is None:
    query_indices = list(range(len(manual_coords)))

# === 시각화 및 저장 ===
for i, ((r, c), q_idx) in enumerate(zip(manual_coords, query_indices)):
    print(f"\n[{i+1}/{len(manual_coords)}] Database 좌표: ({r}, {c}), Query index: {q_idx}")
    
    # query 이미지 가져오기
    qimg = query_imgs[q_idx]
    qname = query_names[q_idx]
    
    # crop 영역 계산
    crop_size = 400
    half = crop_size // 2
    
    y1 = max(0, r - half)
    y2 = min(sat_h, r + half)
    x1 = max(0, c - half)
    x2 = min(sat_w, c + half)
    
    # 유효성 검증
    if y2 <= y1 or x2 <= x1:
        print(f"⚠️ Skipping: Invalid crop coordinates ({x1},{y1}) to ({x2},{y2})")
        continue
    
    # satellite에서 crop
    crop = sat_img[y1:y2, x1:x2]
    
    if crop.size == 0:
        print(f"⚠️ Skipping: Empty crop")
        continue

    # === 시각화 ===
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Database 이미지
    axes[0].imshow(crop)
    actual_h, actual_w = crop.shape[:2]
    center_x_db, center_y_db = actual_w // 2, actual_h // 2
    axes[0].plot(center_x_db, center_y_db, 'ro', markersize=10, markeredgewidth=2, markeredgecolor='white')
    axes[0].set_title(f"Database Crop\nCenter: ({r}, {c}) | Size: {actual_w}×{actual_h}px", fontsize=12)
    axes[0].axis("off")

    # Query 768→512 crop
    q_h, q_w = qimg.shape[:2]
    target_size = 512
    half_target = target_size // 2
    
    q_y1 = q_h // 2 - half_target
    q_y2 = q_h // 2 + half_target
    q_x1 = q_w // 2 - half_target
    q_x2 = q_w // 2 + half_target
    
    qimg_crop = qimg[q_y1:q_y2, q_x1:q_x2]
    
    axes[1].imshow(qimg_crop)
    crop_h, crop_w = qimg_crop.shape[:2]
    center_x_q, center_y_q = crop_w // 2, crop_h // 2
    axes[1].plot(center_x_q, center_y_q, 'ro', markersize=10, markeredgewidth=2, markeredgecolor='white')
    axes[1].set_title(f"Query (idx={q_idx}): {qname}\nSize: {crop_w}×{crop_h}px", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"manual_pair_{i:04d}_r{r}_c{c}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Saved: {save_path}")

print(f"\n✅ 완료! 총 {len(manual_coords)}개 저장 → {save_dir}")