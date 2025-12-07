# query 및 database 이미지 이름에서 좌표 정보 추출에 집중.
# 이미지 이름 형식: q1_@좌표@좌표.png   
# output example:
# 총 1000개 database 이미지 발견
# [DB    0] db_@12345@67890  →  row=12345, col=67890
# [DB    1] db_@23456@78901  →  row=23456, col=78901
# ...
# 총 1000개 query 이미지 발견
# [Q    0] q1_@12345@67890  →  row=12345, col=67890
# [Q    1] q2_@23456@78901  →  row=234

import h5py

# DB / Query h5 경로
#db_h5_path = "datasets/satellite_0_thermalmapping_135/train_database.h5"
#query_h5_path = "datasets/satellite_0_thermalmapping_135/train_queries.h5"

db_h5_path = "dataset_china/china/test_database.h5"
query_h5_path = "dataset_china/china/test_queries.h5"

def parse_coord(name):
    parts = name.split("@")
    row, col = int(parts[-2]), int(parts[-1])
    return row, col

# Database 읽기
with h5py.File(db_h5_path, "r") as f:
    db_names = [n.decode("utf-8") for n in f["image_name"][:]]

print(f"총 {len(db_names)}개 database 이미지 발견\n")
for i, name in enumerate(db_names):
    row, col = parse_coord(name)
    print(f"[DB {i:4d}] {name}  →  row={row}, col={col}")       

print("\n" + "="*80 + "\n")

print("\n")

# Query 읽기
with h5py.File(query_h5_path, "r") as f:
    query_names = [n.decode("utf-8") for n in f["image_name"][:]]

print(f"총 {len(query_names)}개 query 이미지 발견\n")
for i, name in enumerate(query_names):
    row, col = parse_coord(name)
    print(f"[Q {i:4d}] {name}  →  row={row}, col={col}")
