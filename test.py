import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3', help='')
parser.add_argument('--data', type=str, default='data/MD', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/md_adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=404, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', type=str, default='./garage/md_masking_exp1_best_3.3.pth', help='')
parser.add_argument('--plotheatmap', type=str, default='True', help='')

args = parser.parse_args()

def main():
    device = torch.device(args.device)

    # 모델에 필요한 인접행렬 로드
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]
    if args.aptonly:
        supports = None

    # 모델 생성 및 로드
    model = gwnet(device, args.num_nodes, args.dropout,
                  supports=supports,
                  gcn_bool=args.gcn_bool,
                  addaptadj=args.addaptadj,
                  aptinit=adjinit,
                  in_dim=args.in_dim,
                  out_dim=args.seq_length)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print('model load successfully')

    # 데이터셋 로드 및 전처리
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]  # shape: (num_samples, num_nodes, 12)

    # 모델 추론 (테스트 셋에 대해)
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  # shape: (num_samples, num_nodes, 12)

    # 각 horizon별 평가 (전체 테스트 셋)
    amae = []
    amape = []
    armse = []
    awmape = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2], metrics[3]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f},  Test WMAPE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse), np.mean(awmape)))

    # Heatmap 출력 (옵션)
    # if args.plotheatmap == "True":
    #     adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
    #     adp = adp.cpu().detach().numpy()
    #     adp = adp * (1/np.max(adp))
    #     df = pd.DataFrame(adp)
    #     sns.heatmap(df, cmap="viridis")
    #     plt.savefig("./emb.png")
    #     plt.close()

    # 일부 결과 CSV 저장 (예: 센서 99의 horizon 12, 3)
    y12 = realy[:, 99, 11].cpu().detach().numpy()
    yhat12 = scaler.inverse_transform(yhat[:, 99, 11]).cpu().detach().numpy()
    y3 = realy[:, 99, 2].cpu().detach().numpy()
    yhat3 = scaler.inverse_transform(yhat[:, 99, 2]).cpu().detach().numpy()
    df2 = pd.DataFrame({'real12': y12, 'pred12': yhat12, 'real3': y3, 'pred3': yhat3})
    df2.to_csv('./wave.csv', index=False)
    
    # ---------------------------------------------------------------------
    # 추가: 마지막 12일의 연속 시계열 그래프 (하루 = 288프레임, 5분 간격)
    # 여기서는 horizon=0 예측값을 사용하여 연속 시계열을 구성합니다.
    sensor_id = 265           # 선택한 센서 번호 (원하는 센서로 변경 가능)
    frames_per_day = 288     # 하루 288프레임 (5분 간격)
    num_days = 15            # 마지막 12일만 사용
    total_frames = num_days * frames_per_day  # 3456 프레임

    # horizon=0 예측값 및 ground truth 추출 (연속 시계열)
    pred_series = scaler.inverse_transform(yhat[:, sensor_id, 0].cpu().detach().numpy())
    true_series = realy[:, sensor_id, 0].cpu().detach().numpy()

    N = len(true_series)
    if N >= total_frames:
        start_index = N - total_frames
        pred_last = pred_series[start_index:]
        true_last = true_series[start_index:]
    else:
        pred_last = pred_series
        true_last = true_series

    # 일별 그래프를 저장할 디렉토리 생성
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # 하루 288프레임에 해당하는 x축 (0~24시간, 5분 간격)
    time_points = np.linspace(0, 24, frames_per_day, endpoint=False)

    for day in range(num_days):
        day_start = day * frames_per_day
        day_end = (day + 1) * frames_per_day
        day_true = true_last[day_start:day_end]
        day_pred = pred_last[day_start:day_end]

        # 해당 일의 ground truth에서 피크 시간(최대값 인덱스) 찾기
        peak_idx = np.argmax(day_true)
        # ±1시간 (12프레임) 범위 설정 및 평가
        samples_per_hour = 12
        start_idx = max(0, peak_idx - samples_per_hour)
        end_idx = min(len(day_true), peak_idx + samples_per_hour + 1)
        peak_true = day_true[start_idx:end_idx]
        peak_pred = day_pred[start_idx:end_idx]
        peak_mae = np.mean(np.abs(peak_true - peak_pred))
        peak_wmape = np.sum(np.abs(peak_true - peak_pred)) / (np.sum(peak_true) + 1e-6) * 100
        peak_time = time_points[peak_idx]
        ymin, ymax = plt.ylim()
        time_ticks = np.arange(0, 25, 2)  # 0, 2, 4, ..., 24
        time_labels = [f'{int(t):02d}:00' for t in time_ticks]
        peak_y = day_true[peak_idx]
        
        day_wmape = np.sum(np.abs(day_true - day_pred)) / (np.sum(day_true) + 1e-6) * 100
        
        plt.figure(figsize=(15, 6))  # 가로 15, 세로 6인치

        plt.xticks(time_ticks, time_labels)
        plt.xlabel('Time (KST)', fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(time_points, day_true, 'blue',label='Ground Truth', linewidth=2, alpha=0.7)
        plt.plot(time_points, day_pred, 'red',label='Prediction', linestyle='--', linewidth=2, alpha=0.7)
        plt.axvline(time_points[peak_idx], color='green', linestyle='--', label='Peak Time',  alpha=0.8)
        plt.axvspan(time_points[start_idx], time_points[end_idx-1], color='yellow', alpha=0.2, label='±1h Range')
        plt.text(peak_time, peak_y - 100,
                 f'±1h MAE: {peak_mae:.2f}\n±1h WMAPE: {peak_wmape:.2f}%',
                 color='black', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8))
        plt.title(f'Day {day+1} (Node {sensor_id})\nPeak at {time_points[peak_idx]:.2f}h, ±1h MAE: {peak_mae:.2f}, WMAPE: {peak_wmape:.2f}% ')
        plt.xlabel('Time (Hour)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'md_day_{day+1}_Node_{sensor_id}.png'), dpi=300)
        plt.close()
        print(f'Day {day+1} graph saved. Peak: {time_points[peak_idx]:.2f}h, ±1h MAE: {peak_mae:.2f}, '
          f'±1h WMAPE: {peak_wmape:.2f}%, Daily WMAPE: {day_wmape:.2f}%')
        
if __name__ == "__main__":
    main()
