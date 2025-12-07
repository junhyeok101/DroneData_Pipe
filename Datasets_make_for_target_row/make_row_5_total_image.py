#test() 함수가 데이터셋을 준비해서 → 최종적으로 evaluate_SNet()을 호출하고, 여기서 모델 forward와 메트릭 계산/시각화를 전부 수행하는 구조.
# output: 매칭 결과 사진 + matrix

import numpy as np
import os
import torch
import argparse
from model.network import STHN
from utils import save_overlap_img, save_img, setup_seed, save_overlap_bbox_img
import datasets_4cor_img as datasets
import scipy.io as io
import torchvision
import numpy as np
import time
from tqdm import tqdm
import cv2
import kornia.geometry.transform as tgm
import matplotlib.pyplot as plt
from plot_hist import plot_hist_helper
import torch.nn.functional as F
import parser
from datetime import datetime
from os.path import join
import commons
import logging
import wandb
import platform

#테스트/ 검증 데이터셋에 대한 평가 수행 함수 
def test(args, wandb_log):
    if not args.identity:
        model = STHN(args)

        # ---- checkpoint load ----
        model_med = torch.load(args.eval_model, map_location='cuda:0')
        for key in list(model_med['netG'].keys()):
            model_med['netG'][key.replace('module.', '')] = model_med['netG'][key]
        for key in list(model_med['netG'].keys()):
            if key.startswith('module'):
                del model_med['netG'][key]
        model.netG.load_state_dict(model_med['netG'], strict=False)

        if args.two_stages:
            model_med = torch.load(args.eval_model, map_location='cuda:0')
            for key in list(model_med['netG_fine'].keys()):
                model_med['netG_fine'][key.replace('module.', '')] = model_med['netG_fine'][key]
            for key in list(model_med['netG_fine'].keys()):
                if key.startswith('module'):
                    del model_med['netG_fine'][key]
            model.netG_fine.load_state_dict(model_med['netG_fine'])

        model.setup()
        model.netG.eval()
        if args.two_stages:
            model.netG_fine.eval()
    else:
        model = None

    if args.test:
        val_dataset = datasets.fetch_dataloader(args, split='test')
    else:
        val_dataset = datasets.fetch_dataloader(args, split='val')

    evaluate_SNet(model, val_dataset, batch_size=args.batch_size, args=args, wandb_log=wandb_log)



def evaluate_SNet(model, val_dataset, batch_size=0, args=None, wandb_log=False):
    """
    모델 평가를 수행하는 핵심 함수
    - 배치별로 forward pass 수행
    - MACE, CE 등의 메트릭 계산
    - 매칭 결과 시각화 및 저장
    """
    assert batch_size > 0, "batchsize > 0"

    # 변수 정리 
    # 메트릭 누적용 텐서 초기화
    total_mace = torch.empty(0)  # Mean Average Corner Error
    total_flow = torch.empty(0)  # Ground Truth Flow 크기
    total_ce = torch.empty(0)  # Center Error
    total_mace_conf_error = torch.empty(0)  # Uncertainty 관련 에러

    # ✅ 이 부분 추가
    final_mace = 0.0
    final_ce = 0.0
    final_flow = 0.0
    final_mace_conf_error = 0.0

    timeall = []
    mace_conf_list = []

    # Recall@1을 위한 변수들 추가
    correct_predictions_25 = 0
    total_predictions_25 = 0
    correct_predictions_10 = 0
    total_predictions_10 = 0
    correct_predictions_1= 0
    total_predictions_1= 0

    # ==================== 시각화 범위 설정 ====================
    VIS_START_INDEX = 0      # 시작 인덱스 (이 값부터 저장)
    VIS_END_INDEX = None     # 끝 인덱스 (None이면 끝까지, 숫자면 해당 인덱스 전까지)
    # 예시: VIS_START_INDEX = 100, VIS_END_INDEX = 200 → 100번부터 199번까지 저장
    # 예시: VIS_START_INDEX = 0, VIS_END_INDEX = None → 전체 저장
    
    saved_vis_count = 0     # 저장된 시각화 개수

    # 샘플 개수로 제한
    MAX_EVAL_SAMPLES = 2200  # None = 전체 평가, 숫자 지정 시 해당 개수만 평가
    processed_samples = 0   # 처리된 샘플 카운터
    # ==================== 

    if args.generate_test_pairs:
        test_pairs = torch.zeros(len(val_dataset.dataset), dtype=torch.long)

    # GPU 메모리 캐시 정리
    torch.cuda.empty_cache()

    # 배치별 평가 루프
    for i_batch, data_blob in enumerate(tqdm(val_dataset)):
        # 샘플 개수 기준으로 조기 종료 (MAX_EVAL_SAMPLES가 None이 아닐 때만)
        if MAX_EVAL_SAMPLES is not None and processed_samples >= MAX_EVAL_SAMPLES:
            break

        # 데이터 언팩
        img1, img2, flow_gt, H, query_utm, database_utm, index, pos_index = [x for x in data_blob]
        current_batch_size = img1.shape[0]  # 현재 배치의 실제 샘플 수

        if args.generate_test_pairs:
            test_pairs[index] = pos_index

        # 첫 번째 배치에서 재현성 확인용 로그 출력
        """
        if i_batch == 0:
            logging.info("Check the reproducibility by UTM:")
            logging.info(f"the first 5th query UTMs: {query_utm[:5]}")
            logging.info(f"the first 5th database UTMs: {database_utm[:5]}")
        """

        # 1000 배치마다 입력 이미지 저장
        if i_batch % 1000 == 0:
            save_img(torchvision.utils.make_grid((img1)),
                     args.save_dir + "/b1_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.png')
            save_img(torchvision.utils.make_grid((img2)),
                     args.save_dir + "/b2_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.png')
            torch.cuda.empty_cache()

        if not args.identity:
            # 모델에 입력 설정
            model.set_input(img1, img2, flow_gt)

        # ==================== 모델 Forward Pass & 메트릭 계산 ====================
        if args.train_ue_method != 'train_only_ue_raw_input':
            if not args.identity:
                # 그래디언트 계산 비활성화 (평가 모드, 메모리 절약)
                with torch.no_grad():
                    model.forward()
                four_pred = model.four_pred  # 예측된 4개 코너 오프셋 (B,2,2,2)
            else:
                four_pred = torch.zeros((flow_gt.shape[0], 2, 2, 2))

            # ==================== 🔍 첫 배치 디버깅 ====================
            if i_batch == 0:
                print("\n" + "="*80)
                print("🚨 FIRST BATCH DIAGNOSIS")
                print("="*80)
                
                # Ground Truth 4개의 코너의 flow 추출 (여기서 먼저 계산)
                flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
                flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]      # 좌상단
                flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]     # 우상단
                flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]     # 좌하단 
                flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]    # 우하단
                
                # 예측값 확인
                print(f"\n[1] Predictions (first sample):")
                print(f"four_pred[0]:\n{four_pred[0]}")
                print(f"Prediction stats: min={four_pred.min():.2f}, max={four_pred.max():.2f}, mean={four_pred.mean():.2f}, std={four_pred.std():.2f}")
                
                # GT 확인
                print(f"\n[2] Ground Truth (first sample):")
                print(f"flow_4cor[0]:\n{flow_4cor[0]}")
                print(f"GT stats: min={flow_4cor.min():.2f}, max={flow_4cor.max():.2f}, mean={flow_4cor.mean():.2f}")
                
                # Alpha 계산
                original_pixel_size_m = 0.5
                database_actual_size_m = args.database_size * original_pixel_size_m
                alpha = database_actual_size_m / args.resize_width
                
                # MACE 확인 (간단 계산)
                mace_temp = (flow_4cor - four_pred.cpu().detach()) ** 2
                mace_temp = ((mace_temp[:, 0, :, :] + mace_temp[:, 1, :, :]) ** 0.5)
                mace_temp_mean = torch.mean(torch.mean(mace_temp, dim=1), dim=1)
                
                print(f"\n[3] MACE Calculation:")
                print(f"mace_vec[0] (pixels): {mace_temp_mean[0]:.2f}")
                print(f"alpha: {alpha:.6f} m/px")
                print(f"mace_vec[0] (meters): {mace_temp_mean[0] * alpha:.2f}")
                
                # UTM 거리
                utm_dist = torch.sqrt(
                    (query_utm[0,0,0] - database_utm[0,0,0])**2 + 
                    (query_utm[0,0,1] - database_utm[0,0,1])**2
                ).item()
                print(f"\n[4] UTM Distance: {utm_dist:.2f} m")
                
                # 입력 이미지
                print(f"\n[5] Input Images:")
                print(f"img1 range: [{img1.min():.3f}, {img1.max():.3f}]")
                print(f"img2 range: [{img2.min():.3f}, {img2.max():.3f}]")
                
                # 예측이 실제로 작동하는지
                if torch.all(four_pred == 0):
                    print("\n⚠️  WARNING: All predictions are ZERO!")
                
                print("="*80 + "\n")
            # ==================== 첫 배치 디버깅 종료 ====================

# ==================== 배치 내 각 샘플별 시각화 ====================
            if not args.identity:
                for b_idx in range(current_batch_size):
                    # 실제 데이터셋 인덱스 가져오기
                    actual_index = index[b_idx].item()
                    
                    # 시각화 범위 체크
                    if actual_index < VIS_START_INDEX:
                        continue
                    if VIS_END_INDEX is not None and actual_index >= VIS_END_INDEX:
                        continue
                    
                    # 텐서를 numpy로 변환
                    q_img = img1[b_idx].permute(1, 2, 0).cpu().numpy()
                    d_img = img2[b_idx].permute(1, 2, 0).cpu().numpy()
                    # [0,1] 범위 → [0,255] 범위로 변환
                    q_img = (q_img * 255).astype(np.uint8)
                    d_img = (d_img * 255).astype(np.uint8)

                    # 원본 이미지 크기 저장
                    h, w = q_img.shape[:2]
                    
                    # 모델이 사용하는 해상도 (예: 384x384)
                    S = int(args.resize_width)

                    # 이미지를 모델 해상도로 리사이즈
                    q_small = cv2.resize(q_img, (S, S))
                    d_small = cv2.resize(d_img, (S, S))

                    # 원본 4개 코너 좌표 정의 (S×S 이미지 기준)
                    four_point_org_single = torch.zeros((1, 2, 2, 2))
                    four_point_org_single[:, :, 0, 0] = torch.tensor([0, 0])            # 좌상단
                    four_point_org_single[:, :, 0, 1] = torch.tensor([S - 1, 0])        # 우상단
                    four_point_org_single[:, :, 1, 0] = torch.tensor([0, S - 1])        # 좌하단
                    four_point_org_single[:, :, 1, 1] = torch.tensor([S - 1, S - 1])    # 우하단

                    # 호모그래피 계산을 위한 점 집합 생성
                    # src: 원본 4개 코너
                    src_pts = four_point_org_single.flatten(2).permute(0, 2, 1)[0].numpy().astype(np.float32)
                    # dst: 예측된 오프셋을 더한 4개 코너
                    dst_pts_pred  = (four_pred[b_idx].cpu().detach().unsqueeze(0) + four_point_org_single) \
                                .flatten(2).permute(0, 2, 1)[0].numpy().astype(np.float32)

                    # 4점 호모그래피 행렬 계산 및 워핑
                    H_pred  = cv2.getPerspectiveTransform(src_pts, dst_pts_pred)
                    warped_pred  = cv2.warpPerspective(d_small, H_pred, (S, S))

                    ## 알파 블렌딩으로 겹친 이미지 생성
                    alpha_blend = 0.5
                    overlay_small = cv2.addWeighted(q_small, 1 - alpha_blend, warped_pred, alpha_blend, 0)
                    
                    # ========== GT 호모그래피 계산 (초록색으로 표시) ==========
                    # GT flow에서 4개 코너 추출
                    flow_4cor_single = torch.zeros((1, 2, 2, 2))
                    flow_4cor_single[:, :, 0, 0] = flow_gt[b_idx, :, 0, 0]      # 좌상단
                    flow_4cor_single[:, :, 0, 1] = flow_gt[b_idx, :, 0, -1]     # 우상단
                    flow_4cor_single[:, :, 1, 0] = flow_gt[b_idx, :, -1, 0]     # 좌하단 
                    flow_4cor_single[:, :, 1, 1] = flow_gt[b_idx, :, -1, -1]    # 우하단
                    
                    dst_pts_gt = (flow_4cor_single + four_point_org_single) \
                                .flatten(2).permute(0, 2, 1)[0].numpy().astype(np.float32)

                    # GT 호모그래피로 워핑
                    H_gt = cv2.getPerspectiveTransform(src_pts, dst_pts_gt)
                    warped_gt = cv2.warpPerspective(d_small, H_gt, (S, S))
                    
                    # GT 중심 좌표 계산 (4개 코너의 평균)
                    center_gt_x = int(np.mean(dst_pts_gt[:, 0]))
                    center_gt_y = int(np.mean(dst_pts_gt[:, 1]))

                    # 예측 중심 좌표 계산 (4개 코너의 평균)
                    center_pred_x = int(np.mean(dst_pts_pred[:, 0]))
                    center_pred_y = int(np.mean(dst_pts_pred[:, 1]))

                    # ========== 예측, gt 중심점 그리기 ==========
                    # GT 사각형 (초록색)
                    cv2.circle(overlay_small, (center_gt_x, center_gt_y), 5, (0, 255, 0), -1)
 
                    
                    # 예측 중심점 (빨간색 점)
                    cv2.circle(overlay_small, (center_pred_x, center_pred_y), 5, (255, 0, 0), -1)
                    # ========== 오버레이 이미지 생성 ==========                    

                    # 원본 해상도로 업샘플하여 시각화
                    d_big = cv2.resize(d_small, (w, h))
                    overlay_big = cv2.resize(overlay_small, (w, h))

                    # Query | Database | Overlay 형태로 배치 
                    vis3 = np.hstack([q_img, d_big, overlay_big])
                    
                    # 결과 저장 (실제 인덱스로 저장)
                    save_dir = "outputs_NewYork_NY_trained_match_vp100/match_images"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"match_{actual_index:05d}.png")
                    cv2.imwrite(save_path, cv2.cvtColor(vis3, cv2.COLOR_RGB2BGR))
                    
                    saved_vis_count += 1
# ==================== 배치 내 각 샘플별 시각화 ====================


            # ==================== 메트릭 계산 ====================
            # Ground Truth 4개의 코너의 flow 추출
            flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
            flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]      # 좌상단
            flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]     # 우상단
            flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]     # 좌하단 
            flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]    # 우하단

            # Flow 크기 계산 (L2 norm)
            flow_ = (flow_4cor) ** 2
            flow_ = ((flow_[:, 0, :, :] + flow_[:, 1, :, :]) ** 0.5)
            flow_vec = torch.mean(torch.mean(flow_, dim=1), dim=1)

            # MACE 계산 (예측과 GT의 차이)
            mace_ = (flow_4cor - four_pred.cpu().detach()) ** 2
            mace_ = ((mace_[:, 0, :, :] + mace_[:, 1, :, :]) ** 0.5)
            mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)


# ==================== dataset 마다 바꿔야 함  ====================
# 올바른 mace 구하기  mace 나온거 =x
# x *crop 한 사이즈 * 해상도 / resize_width 

            # 올바른 mace 시도
            # 실제 데이터셋의 해상도 정보
            original_pixel_size_m = 0.5  # 원본 위성사진 1픽셀 = 0.5미터

            # Database가 실제로 커버하는 영역
            database_actual_size_m = args.database_size * original_pixel_size_m  # 제주 기준 294픽셀 × 0.5m/픽셀 = 147미터

            # 리사이즈 후 1픽셀이 나타내는 실제 거리
            alpha = database_actual_size_m / args.resize_width  # 147m / 384px ≈ 0.383 m/px


# ==================== dataset 마다 바꿔야 함  ====================


            # MACE를 미터로 변환
            mace_vec = mace_vec * alpha  # 픽셀 → 미터



            # 누적
            total_mace = torch.cat([total_mace, mace_vec], dim=0)
            final_mace = torch.mean(total_mace).item()
            total_flow = torch.cat([total_flow, flow_vec], dim=0)
            final_flow = torch.mean(total_flow).item()

            # ==================== Center Error (CE) 계산 ====================
            # 중심점 오프셋 계산을 위한 코너 정의
            four_point_org_single_w = torch.zeros((1, 2, 2, 2))
            four_point_org_single_w[:, :, 0, 0] = torch.Tensor([0, 0])
            four_point_org_single_w[:, :, 0, 1] = torch.Tensor([args.resize_width - 1, 0])
            four_point_org_single_w[:, :, 1, 0] = torch.Tensor([0, args.resize_width - 1])
            four_point_org_single_w[:, :, 1, 1] = torch.Tensor([args.resize_width - 1, args.resize_width - 1])

            # 예측 및 GT 코너 좌표 계산
            four_point_1 = four_pred.cpu().detach() + four_point_org_single_w
            four_point_org = four_point_org_single_w.repeat(four_point_1.shape[0], 1, 1, 1).flatten(2).permute(0, 2, 1).contiguous()
            four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
            four_point_gt = flow_4cor.cpu().detach() + four_point_org_single_w
            four_point_gt = four_point_gt.flatten(2).permute(0, 2, 1).contiguous()

            # 예측 호모그래피로 중심점 변환
            H_k = tgm.get_perspective_transform(four_point_org, four_point_1)
            center_T = torch.tensor([args.resize_width / 2 - 0.5, args.resize_width / 2 - 0.5, 1]).unsqueeze(1).unsqueeze(0).repeat(H_k.shape[0], 1, 1)
            w_ = torch.bmm(H_k, center_T).squeeze(2)
            center_pred_offset = w_[:, :2] / w_[:, 2].unsqueeze(1) - center_T[:, :2].squeeze(2)

            # GT 호모그래피로 중심점 변환
            H_gt = tgm.get_perspective_transform(four_point_org, four_point_gt)
            w_gt = torch.bmm(H_gt, center_T).squeeze(2)
            center_gt_offset = w_gt[:, :2] / w_gt[:, 2].unsqueeze(1) - center_T[:, :2].squeeze(2)

            # CE 계산 (예측 중심과 GT 중심의 거리)
            ce_ = (center_pred_offset - center_gt_offset) ** 2
            ce_ = ((ce_[:, 0] + ce_[:, 1]) ** 0.5)
            ce_vec = ce_


            # CE를 미터로 변환
            ce_meters = ce_vec * alpha  # 픽셀 → 미터


            total_ce = torch.cat([total_ce, ce_meters], dim=0)
            final_ce = torch.mean(total_ce).item()
            # ==================== Center Error (CE) 계산 종료====================



            # ==================== Recall  계산 ====================
            # CE가 특정 threshold 이하이면 correct prediction으로 간주
            recall_threshold_25 = 25.0  # 25 미터
            correct_in_batch_25 = torch.sum(ce_meters <= recall_threshold_25).item()
            correct_predictions_25 += correct_in_batch_25
            total_predictions_25 += len(ce_vec)

            recall_threshold_10 = 10.0  # 10 미터
            correct_in_batch_10 = torch.sum(ce_meters <= recall_threshold_10).item()
            correct_predictions_10 += correct_in_batch_10
            total_predictions_10 += len(ce_vec)

            recall_threshold_1 = 1.0  # 1 미터
            correct_in_batch_1 = torch.sum(ce_meters <= recall_threshold_1).item()
            correct_predictions_1 += correct_in_batch_1
            total_predictions_1 += len(ce_vec)

            # ==================== Recall  계산 종료 ====================
        
        # 처리된 샘플 수 업데이트
        processed_samples += current_batch_size
    
                
    # 루프 종료 후 Recall 최종 계산
    recall_at_25 = correct_predictions_25 / total_predictions_25 if total_predictions_25 > 0 else 0.0
    recall_at_10 = correct_predictions_10 / total_predictions_10 if total_predictions_10 > 0 else 0.0
    recall_at_1 = correct_predictions_1 / total_predictions_1 if total_predictions_1 > 0 else 0.0

    # ==================== 평가 결과 출력 및 저장 ====================
    if not args.train_ue_method == "train_only_ue_raw_input":
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"MACE Metric: {final_mace:.4f} m")
        print(f'CE Metric: {final_ce:.4f} m')
        print(f'Recall@1m:  {recall_at_1:.4f} ({correct_predictions_1}/{total_predictions_1})')
        print(f'Recall@10m: {recall_at_10:.4f} ({correct_predictions_10}/{total_predictions_10})')
        print(f'Recall@25m: {recall_at_25:.4f} ({correct_predictions_25}/{total_predictions_25})') 
        print(f"{'='*60}\n")

        if wandb_log:
            wandb.log({"test_mace": final_mace})
            wandb.log({"test_ce": final_ce})
            wandb.log({"test_recall_at_1": recall_at_1})
            wandb.log({"test_recall_at_10": recall_at_10})
            wandb.log({"test_recall_at_25": recall_at_25})
            
    # Uncertainty 관련 시각화
    if args.use_ue:
        mace_conf_list = np.array(mace_conf_list)
        
        # MACE vs Confidence 산점도
        plt.figure()
        plt.scatter(mace_conf_list[:, 0], mace_conf_list[:, 1], s=1)
        x = np.linspace(0, 100, 400)
        y = np.exp(args.ue_alpha * x)
        plt.plot(x, y, label='f(x) = exp(-0.1x)', color='red')
        plt.legend()
        plt.savefig(args.save_dir + f'/final_conf.png')
        plt.close()
        
        # Confidence 히스토그램
        plt.figure()
        n, bins, patches = plt.hist(x=mace_conf_list[:, 1], bins=np.linspace(0, 1, 20))
        logging.info(n)
        plt.close()
    
            
    # 결과를 .mat 및 .npy 파일로 저장
    io.savemat(args.save_dir + '/resmat', {'matrix': total_mace.numpy()})
    np.save(args.save_dir + '/resnpy.npy', total_mace.numpy())
    io.savemat(args.save_dir + '/flowmat', {'matrix': total_flow.numpy()})
    np.save(args.save_dir + '/flownpy.npy', total_flow.numpy())
    
    # 히스토그램 플롯 생성
    plot_hist_helper(args.save_dir)

    # ==================== 메트릭 텍스트 파일 저장 ====================
    os.makedirs("outputs_NewYork_NY_trained_match_vp100", exist_ok=True)
    metrics_path = os.path.join("outputs_NewYork_NY_trained_match_vp100", "metrics.txt")

    # 실행 시간 계산
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_str = str(elapsed_time).split(".")[0]


    with open(metrics_path, "w") as f:
        f.write("=== Evaluation Results ===\n")
        f.write(f"MACE: {final_mace:.6f}\n")
        f.write(f"CE:   {final_ce:.6f}\n")
        f.write(f"Recall 10:   {recall_at_10:.6f}\n\n")
        f.write(f"Recall 25:   {recall_at_25:.6f}\n\n")
        f.write(f"Recall 1:   {recall_at_1:.6f}\n\n")

        f.write("=== Data Augmentation ===\n")
        f.write(f"Augment Type     : {args.augment}\n")
        f.write(f"Rotate Max       : {args.rotate_max:.2f} rad ({np.degrees(args.rotate_max):.2f} deg)\n")
        f.write(f"Resize Max       : {args.resize_max}\n")
        f.write(f"Perspective Max  : {args.perspective_max}\n\n")


        f.write("=== Dataset Info ===\n")
        f.write(f"Dataset Name     : {args.dataset_name}\n")
        f.write(f"Database Size    : {args.database_size}\n")
        f.write(f"Positive Thres   : {args.val_positive_dist_threshold}\n")
        f.write(f"Correlation Lvl  : {args.corr_level}\n")
        f.write(f"Generate Pairs   : {args.generate_test_pairs}\n\n")

        f.write("=== Runtime Settings ===\n")
        f.write(f"Batch Size       : {args.batch_size}\n")
        f.write(f"Num Workers      : {args.num_workers}\n")
        f.write(f"Lev0             : {args.lev0}\n")
        f.write(f"Start Time       : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time         : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed Time     : {elapsed_str}\n\n")

        f.write("=== Model Info ===\n")
        f.write(f"Eval Model       : {args.eval_model}\n")
        f.write(f"Two Stages       : {args.two_stages}\n\n")

        f.write("=== System Info ===\n")
        f.write(f"Python Version   : {platform.python_version()}\n")
        f.write(f"PyTorch Version  : {torch.__version__}\n")
        f.write(f"CUDA Version     : {torch.version.cuda}\n")
        f.write(f"GPU              : NVIDIA Tesla V100\n")

        f.write("=== Extra ===\n")
        f.write(f"Flow Mean: {final_flow:.6f}\n")
        f.write(f"Total Samples Evaluated: {processed_samples}\n")
        f.write(f"Total Visualizations Saved: {saved_vis_count}\n\n")


    print(f"[INFO] Metrics saved at {metrics_path}")
    print(f"[INFO] Evaluated {processed_samples} samples, saved {saved_vis_count} visualizations")



if __name__ == '__main__':
    args = parser.parse_arguments()
    start_time = datetime.now()
    if args.identity:
        pass
    else:
        args.save_dir = join(
        "test",
        args.save_dir,
        args.eval_model.split("/")[-2] if args.eval_model is not None else args.eval_model_ue.split("/")[-2],
        f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        commons.setup_logging(args.save_dir, console='info')
    setup_seed(0)
    logging.debug(args)
    wandb_log = False
    if wandb_log:
        wandb.init(project="STHN-eval", entity="aeaaea898-yonsei-university", config=vars(args))
    test(args, wandb_log)



    """
    python3 local_pipeline/t_evaluate_1_image_matrix.py   --datasets_folder t_datasets     --dataset_name 2276_datasets     --eval_model pretrained_models/1536_two_stages/STHN.pth     --val_positive_dist_threshold 512     --lev0     --database_size 1536     --corr_level 4     --test     --num_workers 0     --batch_size 1  
    python3 local_pipeline/t_evaluate_1_image_matrix.py   --datasets_folder dataset_jeju   --dataset_name jeju   --eval_model pretrained_models/1536_two_stages/STHN.pth   --val_positive_dist_threshold 50   --lev0   --database_size 294   --corr_level 4   --test   --num_workers 0   --batch_size 1  --batch_size 1     

    """


    """
    python3 local_pipeline/t_evaluate_1_image_matrix.py   --datasets_folder datasets_jeju   --dataset_name jeju   --eval_model trained/1000_STHN.pth   --val_positive_dist_threshold 100   --lev0   --database_size 294   --corr_level 4   --test   --num_workers 0   --batch_size 1                      

    """