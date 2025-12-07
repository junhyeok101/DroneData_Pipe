# 실제 데이터셋 query와 위성 전체 큰 사진에서 어디를 비행하는 것인지 선으로 그음 

import h5py
import matplotlib.pyplot as plt
import cv2
import sys
import os

# =====================
# 1. target number 받기
# =====================


query_h5_path = f"t_datasets/datasets/test_queries.h5"
sat_img_path = "datasets/maps/satellite/20201117_BingSatellite.png"  # 위성 지도 경로
save_path = f"t_datasets/satellite_trajectory/{target}_trajectory.png"

# -------------------------
# 2. 쿼리 이름에서 좌표 뽑기
# -------------------------
with h5py.File(query_h5_path, "r") as f:
    query_names = [n.decode("utf-8") for n in f["image_name"][:]]

def parse_coord(name):
    parts = name.split("@")
    return int(parts[-2]), int(parts[-1])  # (row, col)

coords = [parse_coord(n) for n in query_names]

print(coords)

# -------------------------
# 3. 위성 이미지 불러오기
# -------------------------
sat_img = cv2.imread(sat_img_path)
sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12), dpi=300)
plt.imshow(sat_img)

# -------------------------
# 4. Trajectory overlay
# -------------------------
unique_rows = sorted(set(r for (r, _) in coords))
for r in unique_rows:
    row_points = [(rr, c) for (rr, c) in coords if rr == r]
    row_points = sorted(row_points, key=lambda x: x[1])
    cols = [c for (_, c) in row_points]
    rows = [r for (_, _) in row_points]
    plt.plot(cols, rows, "r-", linewidth=0.2, alpha=0.7)

plt.gca().invert_yaxis()
plt.title("Trajectory on Satellite Map")
plt.axis("off")

os.makedirs("t_outputs", exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(" 저장 완료: {save_path}")
