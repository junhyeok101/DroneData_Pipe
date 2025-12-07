# 데이터셋을 전처리하여 HDF5 형식으로 저장하는 스크립트
# CSV 파일에서 이미지 메타데이터를 읽고, 이미지를 지정된 크기로 리사이즈한 후,
# 쿼리 및 데이터베이스 HDF5 파일로 저장.
# CSV 파일의 마지막 두 열은 위성 이미지의 픽셀 좌표(X, Y)여야 가능함.

import h5py, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image

# ===== 사용자 설정 =====
CSV_PATH   = "output_china_train_with_pixels.csv"
IMG_DIR    = Path("../AerialVL_sequence/long_trajtr/2023-03-18-14-38-32_영상/cropped_rotate")
OUT_DIR    = Path("251203_china")
TARGET_SIZE = 512                                # 768x768로 리사이즈 # 512 실험 간다
DROP_DUP_XY = True                               # 같은 (X,Y) 좌표 중복 제거

# ===== 헬퍼 =====
def resize_image(im: Image.Image, target: int) -> Image.Image:
    """자르지 않고 직접 리사이즈"""
    return im.resize((target, target), Image.LANCZOS)

# ===== 메인 =====
def main():
    df = pd.read_csv(CSV_PATH)
    
    # 검증
    if df.shape[1] < 3:
        raise ValueError("CSV 마지막 두 열이 위성 픽셀 X,Y 여야 합니다")
    if "image_name" not in df.columns:
        raise ValueError("CSV에 image_name 컬럼이 필요합니다")

    # 마지막 두 열을 X,Y로 사용해 @X@Y 이름 생성
    xcol, ycol = df.columns[-2], df.columns[-1]
    df["__X__"] = df[xcol].astype(float).round().astype(int)
    df["__Y__"] = df[ycol].astype(float).round().astype(int)
    df["__name__"] = df.apply(lambda r: f"@{r['__X__']}@{r['__Y__']}", axis=1)

    if DROP_DUP_XY:
        df = df.drop_duplicates(subset=["__X__", "__Y__"]).reset_index(drop=True)
        print(f"ℹ️  중복 제거 후: {len(df)}개 레코드")

    # 실제 파일 존재하는 레코드만 필터링
    records = []
    for _, r in df.iterrows():
        p = IMG_DIR / str(r["image_name"])
        if p.exists():
            records.append((r, p))
        else:
            print(f"[WARN] 파일 없음: {p}")
    
    N = len(records)
    if N == 0:
        print("[ERR] 유효한 이미지가 없습니다")
        return

    print(f"ℹ️  처리할 이미지: {N}개")
    
    H = W = TARGET_SIZE
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q_path = OUT_DIR / "test_queries.h5"
    d_path = OUT_DIR / "test_database.h5"

    # 먼저 모든 이미지를 처리하여 성공한 것만 리스트에 담기
    processed_data = []
    for idx, (row, path) in enumerate(records):
        try:
            im = Image.open(path).convert("RGB")
            im = resize_image(im, TARGET_SIZE)
            arr = np.asarray(im, dtype=np.uint8)
            name = row["__name__"]
            processed_data.append((arr, name))
            
            if (idx + 1) % 100 == 0:
                print(f"  처리 중... {idx + 1}/{N}")
        except Exception as e:
            print(f"[WARN] 스킵 {path}: {e}")

    actual_N = len(processed_data)
    print(f"ℹ️  성공적으로 처리된 이미지: {actual_N}개")

    # HDF5 파일에 저장
    with h5py.File(q_path, "w") as fq, h5py.File(d_path, "w") as fd:
        # queries
        ds_img   = fq.create_dataset("image_data", (actual_N, H, W, 3), dtype="uint8", chunks=(1, H, W, 3))
        ds_nameq = fq.create_dataset("image_name", (actual_N,), dtype=h5py.string_dtype("utf-8"))
        ds_sizeq = fq.create_dataset("image_size", (actual_N, 2), dtype="int64")
        
        # database
        ds_named = fd.create_dataset("image_name", (actual_N,), dtype=h5py.string_dtype("utf-8"))
        ds_sized = fd.create_dataset("image_size", (actual_N, 2), dtype="int64")

        for i, (arr, name) in enumerate(processed_data):
            ds_img[i]   = arr
            ds_nameq[i] = name
            ds_sizeq[i] = [H, W]
            ds_named[i] = name
            ds_sized[i] = [H, W]

    print(f"✅ 저장 완료: {q_path}")
    print(f"✅ 저장 완료: {d_path}")

if __name__ == "__main__":
    main()