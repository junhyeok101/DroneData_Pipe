# 위성 이미지에서 특정 좌표 기준으로 원하는 크기 crop
# 좌표는 @row@col 형식으로 입력
# crop된 이미지는 RGB에서 BGR로 변환하여 저장

import cv2
import numpy as np
import os

# === 설정 ===
sat_path = "korea_datasets/maps/satellite/20201117_BingSatellite.png"
output_dir = "korea_datasets/sample"
os.makedirs(output_dir, exist_ok=True)

# === 원하는 좌표와 크롭 크기 입력 ===
coord_str = "@703@1268"  # 예시: @row@col 형식
crop_size = 462  # 원하는 crop 크기 (픽셀)

# === 좌표 파싱 ===
def parse_coord(coord_str):
    """@row@col 형식에서 좌표 추출"""
    parts = coord_str.strip().split("@")
    return int(parts[-2]), int(parts[-1])

# === 위성 이미지 로드 ===
sat_img = cv2.imread(sat_path)
if sat_img is None:
    print(f"❌ Failed to load satellite image: {sat_path}")
    exit(1)

sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
sat_h, sat_w = sat_img.shape[:2]
print(f"✅ Satellite image loaded: {sat_w}×{sat_h}")

# === 좌표 추출 ===
r, c = parse_coord(coord_str)
print(f"🎯 Center coordinate: row={r}, col={c}")

# === Crop 영역 계산 ===
half = crop_size // 2

y1 = r - half
y2 = r + half
x1 = c - half
x2 = c + half

# Boundary 체크
y1_clipped = max(0, y1)
x1_clipped = max(0, x1)
y2_clipped = min(sat_h, y2)
x2_clipped = min(sat_w, x2)

# 유효성 검증
if y2_clipped <= y1_clipped or x2_clipped <= x1_clipped:
    print(f"❌ Invalid crop coordinates: ({x1},{y1}) to ({x2},{y2})")
    print(f"   Image size: {sat_w}×{sat_h}")
    exit(1)

# === Crop 수행 ===
crop = sat_img[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

# Empty crop 체크
if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
    print(f"❌ Empty crop at ({x1},{y1}) to ({x2},{y2})")
    exit(1)

print(f"✅ Cropped region: ({x1_clipped},{y1_clipped}) to ({x2_clipped},{y2_clipped})")
print(f"   Crop size: {crop.shape[1]}×{crop.shape[0]}")

# === 저장 ===
# RGB를 BGR로 변환 (OpenCV 저장용)
crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

# 파일명 생성
filename = f"crop_{crop_size}px_{coord_str.replace('@', '')}.png"
save_path = os.path.join(output_dir, filename)

cv2.imwrite(save_path, crop_bgr)
print(f"💾 Saved to: {save_path}")

# === 실제 영역 계산 (0.5m/px 기준) ===
actual_width_m = crop.shape[1] * 0.5
actual_height_m = crop.shape[0] * 0.5
print(f"📏 Actual area: {actual_width_m}m × {actual_height_m}m")