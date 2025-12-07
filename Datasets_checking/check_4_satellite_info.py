#큰 satellite 전체 정보 확인 

import cv2
import h5py


db_h5_path = "korea_datasets/jeju/test_database.h5"
#query_h5_path = "korea_datasets/jeju/test_queries.h5"
sat_img_path = "korea_datasets/maps/satellite/20201117_BingSatellite.png"

# 위성 원본 이미지 경로
#sat_img_path = "datasets/maps/satellite/20201117_BingSatellite.png"

# 위성 이미지 불러오기
sat_img = cv2.imread(sat_img_path)
h, w, c = sat_img.shape
print(f"[Satellite Image]")
print(f" - Size (H×W): {h} × {w}")
print(f" - Channels   : {c}")

# ======================
# HDF5 파일로 해상도 확인
# ======================
#db_h5_path = "datasets/satellite_0_thermalmapping_135/train_database.h5"
with h5py.File(db_h5_path, "r") as f:
    if "image_size" in f:
        print("\n[Database Info]")
        print("image_size shape:", f["image_size"].shape)
        print("첫 5개 사이즈 예시:", f["image_size"][:5].tolist())

    if "image_name" in f:
        print("DB image 개수:", len(f["image_name"]))
        print("첫 5개 이름 예시:", [n.decode("utf-8") for n in f["image_name"][:5]])