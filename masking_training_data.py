import argparse
import numpy as np
import os
import pandas as pd
import time
import util  # 인접 행렬 로드에 필요한 util.load_adj(..)

def spatial_interpolate_by_neighbors(df, adj_mx, print_interval=1000):
    """
    df: pandas.DataFrame, shape = (T, V)
        행: 시간, 열: 노드
    adj_mx: numpy array, shape = (V, V)
        인접 행렬(A). 1-hop 이웃 파악용
        adj_mx[v,u] != 0 이면 노드 v와 u가 연결되었다고 가정

    전 구간 결측된 노드도 이웃 노드의 같은 시점 데이터로 채움.
    (전 구간 결측이면 시간축 보간 불가 → 이웃 보간만 시도)
    """
    T, V = df.shape
    print(f"[INFO] spatial_interpolate_by_neighbors: Data shape = {T}x{V}")

    # 1) 각 노드별 이웃 목록 수집
    neighbors = []
    for v in range(V):
        ngh = np.where(adj_mx[v, :] != 0)[0]  # v노드와 연결된 이웃
        neighbors.append(ngh)

    nan_before = df.isna().sum().sum()
    print(f"[DEBUG] Before interpolation: total NaN count = {nan_before}")

    start_time = time.time()
    fill_count = 0

    # 2) 시계열 전체 순회하며 NaN이면 이웃값 평균으로 대체
    for t in range(T):
        if (t % print_interval == 0) and (t > 0):
            elapsed = time.time() - start_time
            print(f"[DEBUG] processed row t={t}/{T} (elapsed {elapsed:.2f}s)")

        for v in range(V):
            if pd.isna(df.iat[t, v]):
                vals = []
                for u in neighbors[v]:
                    val = df.iat[t, u]
                    if not pd.isna(val):
                        vals.append(val)
                if len(vals) > 0:
                    df.iat[t, v] = np.mean(vals)
                else:
                    # 이웃도 모두 결측이면 0으로 대체
                    df.iat[t, v] = 0
                fill_count += 1

    nan_after = df.isna().sum().sum()
    end_time = time.time()

    print(f"[DEBUG] spatial_interpolate_by_neighbors done. Filled {fill_count} cells.")
    print(f"[DEBUG] After interpolation: total NaN count = {nan_after}")
    print(f"[DEBUG] Time taken: {end_time - start_time:.2f} seconds.")

    return df


def generate_graph_seq2seq_io_data(
    df, x_offsets, y_offsets, 
    add_time_in_day=True, 
    add_day_in_week=False
):
    """
    df: 시계열 DataFrame (shape: (num_times, num_nodes))
    x_offsets, y_offsets: 예) [-11..0], [1..12]
    add_time_in_day, add_day_in_week: 부가적 시간 피처 추가 여부

    return: (x, y)
      x.shape = (N, len(x_offsets), num_nodes, input_dim)
      y.shape = (N, len(y_offsets), num_nodes, input_dim)
    """
    num_times, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)  # (T, V, 1)

    feature_list = [data]
    # 일중시간(time_in_day) 추가
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [num_nodes, 1]).T  # (T, V)
        time_in_day = np.expand_dims(time_in_day, axis=-1) # (T, V, 1)
        feature_list.append(time_in_day)
    # 요일(day_of_week) 추가
    if add_day_in_week:
        dow = df.index.dayofweek  # 0=월, 6=일
        dow_tiled = np.tile(dow, [num_nodes, 1]).T  # (T, V)
        dow_tiled = np.expand_dims(dow_tiled, axis=-1)
        feature_list.append(dow_tiled)

    # concat along last dim
    data = np.concatenate(feature_list, axis=-1)  # (T, V, input_dim)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = num_times - abs(max(y_offsets))  # exclusive

    for t in range(min_t, max_t):
        x_window = data[t + x_offsets, ...]  # (len(x_offsets), V, input_dim)
        y_window = data[t + y_offsets, ...]  # (len(y_offsets), V, input_dim)
        x.append(x_window)
        y.append(y_window)

    x = np.stack(x, axis=0)  # (N, len(x_offsets), V, input_dim)
    y = np.stack(y, axis=0)  # (N, len(y_offsets), V, input_dim)
    return x, y


def generate_train_val_test(args):
    # 1) 원본 DF 로드
    df = pd.read_hdf(args.traffic_df_filename)  # shape: (time_len, num_nodes)
    print("[INFO] Raw data shape:", df.shape)

    # 2) 인접 행렬 로드
    sensor_ids, sensor_id_to_ind, adj_list = util.load_adj(args.adjdata, args.adjtype)
    adj_mx = adj_list[0] if isinstance(adj_list, list) else adj_list
    adj_mx = np.array(adj_mx)  # (V, V)

    # 3) Train/Val/Test 시간 분할
    num_times = df.shape[0]
    num_train = int(num_times * args.train_ratio)
    num_val = int(num_times * args.val_ratio)
    num_test = num_times - num_train - num_val

    train_end = num_train
    val_end = num_train + num_val
    # df_train: 0 ~ train_end
    # df_val:   train_end ~ val_end
    # df_test:  val_end ~ 끝
    df_train = df.iloc[:train_end].copy()
    df_val   = df.iloc[train_end:val_end].copy()
    df_test  = df.iloc[val_end:].copy()

    print("[INFO] Split by time:")
    print("  Train =", df_train.shape, ", Val =", df_val.shape, ", Test =", df_test.shape)

    # 4) 마스킹 (Train, Val)에만 적용 / Test는 그대로 유지 => GT
    #    필요에 따라 Val에도 마스킹할지 선택
    if args.mask_nodes_num > 0:
        np.random.seed(args.mask_nodes_seed)
        V = df.shape[1]
        if args.mask_nodes_num > V:
            raise ValueError(f"mask_nodes_num({args.mask_nodes_num}) > total nodes({V})")

        # 무작위로 mask_nodes_num개 노드 추출
        masked_nodes = np.random.choice(V, args.mask_nodes_num, replace=False)
        print(f"[INFO] Masking these node indices => {masked_nodes[:40]}... (only first 10 shown)")

        # 마스킹을 Train에만 적용 (Val에도 적용하려면 아래 주석 해제)
        df_train.iloc[:, masked_nodes] = np.nan
        df_val.iloc[:, masked_nodes]   = np.nan  # ← 검증에도 마스킹하려면 주석 해제

        # 마스킹된 노드 저장 (추후 모델에서 마스킹 노드만 성능 확인할 때 사용)
        np.save(os.path.join(args.output_dir, "masked_nodes.npy"), masked_nodes)

    # 5) 이웃 보간 (마스킹된 구간만)
    #    - Test 구간은 마스킹이 없으므로 이웃 보간 불필요
    df_train = spatial_interpolate_by_neighbors(df_train, adj_mx, print_interval=1000)
    df_val = spatial_interpolate_by_neighbors(df_val, adj_mx, print_interval=1000)
    # df_test는 GT 그대로

    # 남아있는 NaN이 있으면 0으로 채움
    df_train = df_train.fillna(0)
    df_val   = df_val.fillna(0)
    df_test  = df_test.fillna(0)

    # 6) (x, y) 윈도우화 할 인덱스 범위
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)  
    y_offsets = np.arange(args.y_start, args.y_start + seq_length_y, 1)

    # 7) Train/Val/Test 각각을 윈도우화
    x_train, y_train = generate_graph_seq2seq_io_data(
        df_train,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=args.add_time_in_day,
        add_day_in_week=args.add_day_in_week
    )
    x_val, y_val = generate_graph_seq2seq_io_data(
        df_val,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=args.add_time_in_day,
        add_day_in_week=args.add_day_in_week
    )
    x_test, y_test = generate_graph_seq2seq_io_data(
        df_test,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=args.add_time_in_day,
        add_day_in_week=args.add_day_in_week
    )

    print("x_train:", x_train.shape, ", y_train:", y_train.shape)
    print("x_val:",   x_val.shape,   ", y_val:",   y_val.shape)
    print("x_test:",  x_test.shape,  ", y_test:",  y_test.shape)

    # 8) .npz로 저장
    for cat in ["train", "val", "test"]:
        _x = locals()[f"x_{cat}"]
        _y = locals()[f"y_{cat}"]
        out_path = os.path.join(args.output_dir, f"{cat}.npz")
        np.savez_compressed(
            out_path,
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape((-1, 1)),
            y_offsets=y_offsets.reshape((-1, 1))
        )
        print(f"[INFO] Saved {cat}.npz to {out_path}.")

    print("[INFO] Done generate_train_val_test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/MD", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/MD/md_count.h5", help="Raw traffic readings.")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/md_adj_mx.pkl", help="Adj data path.")
    parser.add_argument("--adjtype", type=str, default="doubletransition", help="Adj type.")
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence length for input.")
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence length for prediction.")
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start index")

    parser.add_argument("--add_time_in_day", action='store_true', help="일중시간 feature 사용 여부")
    parser.add_argument("--add_day_in_week", action='store_true', help="요일 feature 사용 여부")

    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    # test_ratio는 나머지(1 - train_ratio - val_ratio)로 계산

    # 노드 마스킹을 '개수'로 지정
    parser.add_argument("--mask_nodes_num", type=int, default=20,
                        help="전 구간 결측으로 만들 노드 개수(0이면 마스킹 안 함).")
    parser.add_argument("--mask_nodes_seed", type=int, default=1234,
                        help="마스킹 노드를 고를 때 사용할 랜덤 시드")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generate_train_val_test(args)
