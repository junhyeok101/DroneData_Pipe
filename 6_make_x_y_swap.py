# HDF5 파일 내 image_name 데이터에서 @row@col 패턴을 @col@row로 바꾸는 스크립트
# 이는 이전에 잘못 저장된 좌표 순서를 수정하기 위함임
# 두 파일 모두에 적용됨: test_queries.h5, test_database.h5

import h5py
import re

def swap_row_col_in_names(file_path):
    print(f"\n=== {file_path} 처리 중 ===")
    
    with h5py.File(file_path, 'r+') as f:
        # image_name 데이터 읽기
        image_names = f['image_name'][:]
        
        # 변환된 이름들을 저장할 리스트
        new_names = []
        
        for name in image_names:
            # bytes인 경우 string으로 변환
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # @row@col 패턴 찾아서 바꾸기
            match = re.search(r'@(\d+)@(\d+)', name)
            if match:
                row = match.group(1)
                col = match.group(2)
                # @row@col을 @col@row로 변경
                new_name = name.replace(f'@{row}@{col}', f'@{col}@{row}')
                new_names.append(new_name)
            else:
                new_names.append(name)
        
        # 기존 데이터셋 삭제하고 새로 생성
        del f['image_name']
        f.create_dataset('image_name', data=new_names, dtype=h5py.special_dtype(vlen=str))
        
        print(f"✓ 변환 완료!")
        print(f"  샘플 (변환 전): {image_names[:3]}")
        print(f"  샘플 (변환 후): {new_names[:3]}")

# 두 파일 모두 처리
swap_row_col_in_names('251203_china/test_queries.h5')
swap_row_col_in_names('251203_china/test_database.h5')

print("\n✅ 모든 파일 변환 완료!")