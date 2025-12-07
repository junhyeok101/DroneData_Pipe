# 실제 데이터셋의 경로를 위성 이미지 위에 좌표로 표현해서 이미지 저장
# output은 이미지 파일로 저장.

import h5py
import matplotlib.pyplot as plt
import cv2

query_h5_path = "dataset_china/china/test_queries.h5"
sat_img_path = "dataset_china/maps/satellite/20201117_BingSatellite.png"

# -------------------------
# 1. 쿼리 이름에서 좌표 뽑기
# -------------------------
with h5py.File(query_h5_path, "r") as f:
    query_names = [n.decode("utf-8") for n in f["image_name"][:300000]]

def parse_coord(name):
    parts = name.split("@")
    return int(parts[-2]), int(parts[-1])  # (row, col)

coords = [parse_coord(n) for n in query_names]

# -------------------------
# 2. 위성 이미지 불러오기
# -------------------------
sat_img = cv2.imread(sat_img_path)

# 이미지 로드 확인
if sat_img is None:
    print(f"❌ 이미지를 불러올 수 없습니다: {sat_img_path}")
    print(f"파일 경로를 확인하세요.")
    exit()

print(f"✓ 이미지 로드 성공: {sat_img.shape}")
sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

# # ========== 180도 회전 추가 ==========
# sat_img = cv2.flip(sat_img, 0)
# sat_img = cv2.rotate(sat_img, cv2.ROTATE_90_CLOCKWISE)


plt.figure(figsize=(12, 12), dpi=300)
plt.imshow(sat_img)

# -------------------------
# 3. Trajectory overlay - 점으로 표현
# -------------------------
unique_rows = sorted(set(r for (r, _) in coords))
for r in unique_rows:
    row_points = [(rr, c) for (rr, c) in coords if rr == r]
    row_points = sorted(row_points, key=lambda x: x[1])
    cols = [c for (_, c) in row_points]
    rows = [r for (_, _) in row_points]
    
    # 옵션 1: 점만 그리기
    plt.scatter(cols, rows, c="red", s=0.5, alpha=0.7)
    
    # 옵션 2: 선 + 점
    # plt.plot(cols, rows, "r-", linewidth=0.2, alpha=0.5)
    # plt.scatter(cols, rows, c="red", s=0.5, alpha=0.7)

plt.title("Trajectory on Satellite Map")
plt.axis("off")

plt.savefig("dataset_china/satellite_trajectory.png", dpi=300, bbox_inches="tight")
plt.show()