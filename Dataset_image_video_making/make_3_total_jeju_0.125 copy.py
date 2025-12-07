# query - database 쌍 시각화 및 저장
# korean_1에서 크롭한 query 이미지들과
# korean_3에서 좌표 기반으로 크롭한 database 이미지들을
# 한 쌍으로 시각화하여 저장

import h5py
import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

sat_path = "dataset_jeju_0.125/maps/satellite/jeju_0.125m_px.jpg"
# === 경로 설정 ===
row_query_path = "dataset_jeju_0.125/jeju/test_queries.h5"
row_db_path = "dataset_jeju_0.125/jeju/test_database.h5"

# 출력 폴더
save_dir = "dataset_jeju_0.125/total_images"
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
    query_imgs = fq["image_data"][:]  # (N, 768, 768, 3)

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
    crop_size = 984
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
    
    # full satellite에서 crop
    crop = sat_img[y1:y2, x1:x2]
    
    # Empty crop 체크
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        print(f"⚠️ Skipping {i}: Empty crop at ({x1},{y1}) to ({x2},{y2})")
        skipped += 1
        continue
    
    # Resize to 768×768
    crop_resized = cv2.resize(crop, (768, 768))

    # 대응 database 이미지 찾기
    if qname in db_names:
        db_r, db_c = parse_coord(qname)
        
        db_y1 = max(0, db_r - half)
        db_y2 = min(sat_h, db_r + half)
        db_x1 = max(0, db_c - half)
        db_x2 = min(sat_w, db_c + half)
        
        if db_y2 > db_y1 and db_x2 > db_x1:
            db_crop = sat_img[db_y1:db_y2, db_x1:db_x2]
            if db_crop.size > 0:
                db_resized = cv2.resize(db_crop, (768, 768))
            else:
                db_resized = crop_resized  # fallback
        else:
            db_resized = crop_resized  # fallback
    else:
        db_resized = crop_resized  # fallback

    # === 시각화 (Database와 Query만) ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Database 이미지 표시
    axes[0].imshow(crop_resized)
    
    # 중앙에 빨간 점 찍기
    center_x, center_y = 768 // 2, 768 // 2
    axes[0].plot(center_x, center_y, 'ro', markersize=3, markeredgewidth=2, markeredgecolor='white')
    
    axes[0].set_title(f"Database Crop\n(512px)", fontsize=12)
    axes[0].axis("off")

    # Query 이미지 표시
    axes[1].imshow(qimg)
    
    # Query 중앙에도 빨간 점 찍기
    axes[1].plot(center_x, center_y, 'ro', markersize=3, markeredgewidth=2, markeredgecolor='white')
    
    axes[1].set_title(f"Query (Thermal)\n(768px)", fontsize=12)
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