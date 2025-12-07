# -- 이 코드는 특정 행 번호에 해당하는 thermal query와 satellite database의 HDF5 파일에서
# 중복되거나 순서가 맞지 않는 항목을 제거하고, 해당 행
# 에 속한 데이터만을 새로운 HDF5 파일로 저장합니다. --
# output example:
# test_queries.h5: 전체 500개 중 row=3131 → 450개 keep
# ✅ 덮어쓰기 완료 → t_datasets/3131_datasets/test_queries.h5
# test_database.h5: 전체 1500개 중 row=3131 → 1400개 keep
# ✅ 덮어쓰기 완료 → t_datasets/3131_datasets/test_database.h5  

import h5py
import numpy as np
import os
import sys

# ===== Target Row 받기 =====
if len(sys.argv) < 2:
    raise ValueError("❌ target number (row) 인자가 필요합니다! 예: python3 make_row_2.5_refine.py 2276")

try:
    row_target = int(sys.argv[1])
except ValueError:
    raise ValueError(f"❌ 잘못된 target number 입력: {sys.argv[1]}")

# ===== 경로 설정 =====
target_dir = f"t_datasets/{row_target}_datasets"
database_path = os.path.join(target_dir, "test_database.h5")
query_path = os.path.join(target_dir, "test_queries.h5")


def parse_coord(name):
    """이름에서 (row, col) 좌표 추출"""
    parts = name.split("@")
    return int(parts[1]), int(parts[2])


def keep_first_increasing(input_path, row_target):
    tmp_path = input_path + ".tmp"  # 임시 파일
    with h5py.File(input_path, "r") as f:
        names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in f["image_name"][:]]

        # row_target에 해당하는 것만 선택
        selected = [(i, parse_coord(n)) for i, n in enumerate(names) if parse_coord(n)[0] == row_target]

        # col 값 오름차순 구간만 keep
        kept_indices = []
        last_col = -1
        for i, (row, col) in selected:
            if col > last_col:
                kept_indices.append(i)
                last_col = col
            else:
                break

        print(f"{input_path}: 전체 {len(names)}개 중 row={row_target} → {len(kept_indices)}개 keep")

        # 임시 파일에 저장
        with h5py.File(tmp_path, "w") as f_out:
            f_out.create_dataset("image_name", data=[n.encode("utf-8") for n in np.array(names)[kept_indices]])
            if "image_data" in f:  # queries
                f_out.create_dataset("image_data", data=f["image_data"][kept_indices])
            if "image_size" in f:  # database
                f_out.create_dataset("image_size", data=f["image_size"][kept_indices])

    # 원본 파일 덮어쓰기
    os.replace(tmp_path, input_path)
    print(f"✅ 덮어쓰기 완료 → {input_path}")


# ===== 실행 =====
keep_first_increasing(query_path, row_target)
keep_first_increasing(database_path, row_target)
