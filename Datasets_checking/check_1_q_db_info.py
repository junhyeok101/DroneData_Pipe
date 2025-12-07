# query 및 database h5 파일의 구조를 확인하는 스크립트
# 각 파일의 key 목록과 데이터셋의 shape, dtype 등을 출력
# ex) image_name, image_data, image_size 등


# output example:
# [파일] dataset_daegu/daegu/test_queries.h5
# Key 목록: ['image_name', 'image_data', 'image_size']
#  - image_name: shape=(1000,), dtype=object
#  - image_data: shape=(1000, 768, 768, 3), dtype=uint8
#  - image_size: shape=(1000, 2), dtype=int32
#  - image_name 샘플: ['q1_@12345@67890', 'q2_@23456@78901', ...]   

import h5py
import json

def inspect_h5(path):
    with h5py.File(path, "r") as f:
        print(f"\n[파일] {path}")
        print("Key 목록:", list(f.keys()))

        for key in f.keys():
            dset = f[key]
            if hasattr(dset, "shape"):
                print(f" - {key}: shape={dset.shape}, dtype={dset.dtype}")
            else:
                print(f" - {key}: (non-array dataset)")

        # 이름 샘플 출력
        if "image_name" in f:
            names = [name.decode("utf-8") for name in f["image_name"][:5]]
            print(" - image_name 샘플:", names)

        if "image_size" in f:
            print(" - image_size 샘플:", f["image_size"][:5].tolist())




db_h5_path = "dataset_daegu/daegu/test_queries.h5"
query_h5_path = "dataset_daegu/daegu/test_database.h5"

# Query 파일
inspect_h5(query_h5_path)

# Database 파일
inspect_h5(db_h5_path)

# Annotation (있을 경우)
anno_path = "datasets/satellite_0_thermalmapping_135/train_annotations.json"
try:
    with open(anno_path, "r") as f:
        anno = json.load(f)
        print("\n[Annotation 샘플]")
        for k, v in list(anno.items())[:5]:
            print(f"Query: {k} → Matches: {v}")
except FileNotFoundError:
    print("\nAnnotation 파일 없음")


