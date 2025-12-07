## 특정 데이터셋에서 특정 기준 직선 경로 query만 뽑아옴 (빠른 중복 제거 버전)
## input : 실제 데이터셋 속 하나 
## output: trajectory 용 query h5 -> t_datasets/3131_datasets/

import h5py
import os
import numpy as np

import sys

# ===== Target Row 받기 =====
if len(sys.argv) < 2:
    raise ValueError("❌ target number (row) 인자가 필요합니다! 예: python3 make_row_1_query.py 2276")

try:
    target_row = int(sys.argv[1])
except ValueError:
    raise ValueError(f"❌ 잘못된 target number 입력: {sys.argv[1]}")


query_h5_path = "datasets/satellite_0_thermalmapping_135/test_queries.h5"

output_dir = f"t_datasets/{target_row}_datasets"
os.makedirs(output_dir, exist_ok=True)

output_h5_path = os.path.join(output_dir, f"test_queries.h5")

def parse_coord(name):
    parts = name.split("@")
    return int(parts[-2]), int(parts[-1])  # (row, col)

with h5py.File(query_h5_path, "r") as f, h5py.File(output_h5_path, "w") as f_out:
    query_names = [n.decode("utf-8") for n in f["image_name"][:]]

    # target row 필터링
    selected_idxs = [i for i, q in enumerate(query_names) if parse_coord(q)[0] == target_row]

    unique_dict = {}
    for i in selected_idxs:
        name = query_names[i]
        if name not in unique_dict:  # ✅ 중복 제거
            unique_dict[name] = f["image_data"][i]  # 필요한 것만 읽기

    unique_names = list(unique_dict.keys())
    unique_imgs = np.stack(list(unique_dict.values()), axis=0)

    # 새 h5에 저장
    f_out.create_dataset("image_name", data=[n.encode("utf-8") for n in unique_names])
    f_out.create_dataset("image_data", data=unique_imgs)

    print(f"[Row {target_row}] Query 중복 제거 완료 → {output_h5_path}")
    print(f"총 {len(selected_idxs)}개 중 {len(unique_names)}개만 남음")
