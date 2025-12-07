# query - database 쌍 시각화 및 저장 (Query 768→512 crop)
# 위성 이미지에서 database 위치 crop 및 query 512 crop 후 시각화
# query와 database 이미지 중앙에 빨간 점 표시
# 정확한 좌표 일치를 확인하기 위함. ~!!!!

import h5py
import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np


sat_path = "dataset_china/maps/satellite/20201117_BingSatellite.png"
# === 경로 설정 ===
row_query_path = "dataset_china/china/test_queries.h5"
row_db_path = "dataset_china/china/test_database.h5"

# 출력 폴더
save_dir = "dataset_china/center_query_datbase"
os.makedirs(save_dir, exist_ok=True)

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

# === database 읽기 (좌표만) ===
with h5py.File(row_db_path, "r") as fd:
    db_names = [n.decode("utf-8") for n in fd["image_name"][:]]

print(f"✅ Loaded {len(db_names)} database names")

def parse_coord(name):
    """@row@col 형식에서 좌표 추출"""
    parts = name.split("@")
    return int(parts[-2]), int(parts[-1])

# === query와 database 짝 맞춰 저장 ===
skipped = 0
for i, (qname, qimg) in enumerate(zip(query_names, query_imgs)):
    r, c = parse_coord(qname)

    # crop 영역 계산
    crop_size = 928
    half = crop_size // 2
    
    y1 = r - half
    y2 = r + half
    x1 = c - half
    x2 = c + half
    
    # Boundary 체크
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(sat_h, y2)
    x2 = min(sat_w, x2)
    
    # 유효성 검증
    if y2 <= y1 or x2 <= x1:
        print(f"⚠️ Skipping {i}: Invalid crop coordinates ({x1},{y1}) to ({x2},{y2})")
        skipped += 1
        continue
    
    # full satellite에서 crop (리사이즈 하지 않음)
    crop = sat_img[y1:y2, x1:x2]
    
    # Empty crop 체크
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        print(f"⚠️ Skipping {i}: Empty crop at ({x1},{y1}) to ({x2},{y2})")
        skipped += 1
        continue


    # === 시각화 (Database와 Query만) ===
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Database 이미지 표시
    axes[0].imshow(crop)
    
    # 실제 crop 크기의 중앙에 빨간 점 찍기 ✓
    actual_h, actual_w = crop.shape[:2]  # 실제 크기 사용!
    center_x_db, center_y_db = actual_w // 2, actual_h // 2
    axes[0].plot(center_x_db, center_y_db, 'ro', markersize=10, markeredgewidth=2, markeredgecolor='white')
    
    axes[0].set_title(f"Database Crop\n({actual_w}×{actual_h}px)", fontsize=12)
    axes[0].axis("off")

    # ============================================================
    # Query 원본 768×768을 512×512로 중앙 기준 crop ✓
    # ============================================================
    q_h, q_w = qimg.shape[:2]  # 원본 Query 크기 (768×768)
    target_size = 512  # 512로 crop
    half_target = target_size // 2  # 256
    
    # 중앙 기준 crop 범위 계산
    # 768의 중앙: 384
    # 384 - 256 = 128 (시작)
    # 384 + 256 = 640 (끝)
    q_y1 = q_h // 2 - half_target  # 128
    q_y2 = q_h // 2 + half_target  # 640
    q_x1 = q_w // 2 - half_target  # 128
    q_x2 = q_w // 2 + half_target  # 640
    
    # 실제 crop (768×768 → 512×512)
    qimg_crop = qimg[q_y1:q_y2, q_x1:q_x2]
    
    # Query 이미지 표시
    axes[1].imshow(qimg_crop)
    
    # Query 중앙에 빨간 점 찍기 ✓
    crop_h, crop_w = qimg_crop.shape[:2]  # 512, 512
    center_x_q, center_y_q = crop_w // 2, crop_h // 2
    axes[1].plot(center_x_q, center_y_q, 'ro', markersize=10, markeredgewidth=2, markeredgecolor='white')
    
    axes[1].set_title(f"Query (Thermal - 512×512 Cropped from 768×768)\n({crop_w}×{crop_h}px)", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"pair_{i:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    if i % 20 == 0:
        print(f"Saved {i}/{len(query_names)} images")

print(f"\n✅ 이미지 쌍 시각화 완료 → {save_dir}")
print(f"   총 {len(query_names)}개 중 {len(query_names) - skipped}개 저장 완료")
if skipped > 0:
    print(f"   ⚠️ {skipped}개 스킵됨 (좌표 범위 초과)")