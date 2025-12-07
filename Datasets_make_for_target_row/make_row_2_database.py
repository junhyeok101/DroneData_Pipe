# 이 코드는 특정 행 번호에 해당하는 satellite database의 HDF5 파일에서 중복된 항목을 제거하고, 해당 행에 속한 데이터만을 새로운 HDF5 파일로 저장합니다.
# output example:
# [Row 3131] Database 중복 제거 완료 → t_datasets/3131_datasets/test_database.h5
# 총 1500개 중 1450개만 남음

import h5py
import os


import sys

# ===== Target Row 받기 =====
if len(sys.argv) < 2:
    raise ValueError("❌ target number (row) 인자가 필요합니다! 예: python3 make_row_1_query.py 2276")

try:
    target_row = int(sys.argv[1])
except ValueError:
    raise ValueError(f"❌ 잘못된 target number 입력: {sys.argv[1]}")



db_h5_path = "datasets/satellite_0_thermalmapping_135/test_database.h5"

# row 별 폴더 생성 (예: t_datasets/3131_datasets/)
output_dir = f"t_datasets/{target_row}_datasets"
os.makedirs(output_dir, exist_ok=True)

# 저장 파일 경로
output_h5_path = os.path.join(output_dir, f"test_database.h5")

def parse_coord(name):
    parts = name.split("@")
    return int(parts[-2]), int(parts[-1])  # (row, col)

with h5py.File(db_h5_path, "r") as f, h5py.File(output_h5_path, "w") as f_out:
    db_names = [n.decode("utf-8") for n in f["image_name"][:]]
    db_sizes = f["image_size"][:]

    # target row 필터링
    selected_idxs = [i for i, n in enumerate(db_names) if parse_coord(n)[0] == target_row]
    selected_names = [db_names[i] for i in selected_idxs]
    selected_sizes = db_sizes[selected_idxs]

    # ✅ 중복 제거
    unique_dict = {}
    for name, size in zip(selected_names, selected_sizes):
        unique_dict[name] = size  # dict은 key 중복 제거됨

    unique_names = list(unique_dict.keys())
    unique_sizes = list(unique_dict.values())

    # 새 h5에 저장
    f_out.create_dataset("image_name", data=[n.encode("utf-8") for n in unique_names])
    f_out.create_dataset("image_size", data=unique_sizes)

    print(f"[Row {target_row}] Database 중복 제거 완료 → {output_h5_path}")
    print(f"총 {len(selected_names)}개 중 {len(unique_names)}개만 남음")
