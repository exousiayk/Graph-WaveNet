import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
import os
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()

def save_adaptive_adj_heatmap(engine, epoch, results_dir):
    if hasattr(engine.model, 'nodevec1') and hasattr(engine.model, 'nodevec2'):
        with torch.no_grad():
            nodevec1 = engine.model.nodevec1
            nodevec2 = engine.model.nodevec2
            adp = torch.softmax(torch.relu(torch.mm(nodevec1, nodevec2)), dim=1)
            adp_np = adp.cpu().numpy()

        # 1) 전체 인접행렬 시각화
        plt.figure(figsize=(6, 5))
        plt.imshow(adp_np, cmap='viridis')
        plt.colorbar()
        plt.title(f'Adaptive Adjacency Matrix at Epoch {epoch}')
        plt.xlabel('Node')
        plt.ylabel('Node')
        heatmap_path = f"{results_dir}/adaptive_adj_epoch_{epoch}.png"
        plt.savefig(heatmap_path)
        plt.close()
        print(f"[INFO] Saved Adaptive Adjacency heatmap -> {heatmap_path}")

        # 2) 중앙에서 50개 노드만 잘라내어 별도로 시각화
        center = adp_np.shape[0] // 2  # 보통 207이면 103s
        half_size = 25                # 양옆으로 25개씩 = 50개
        start_idx = 0
        end_idx = 50

        adp_crop = adp_np[start_idx:end_idx, start_idx:end_idx]

        plt.figure(figsize=(6, 5))
        plt.imshow(adp_crop, cmap='viridis')
        plt.colorbar()
        plt.title(f'Adaptive Adjacency Matrix (Nodes 1-50) at Epoch {epoch}')
        plt.xlabel('Node')
        plt.ylabel('Node')
        heatmap_crop_path = f"{results_dir}/adaptive_adj_epoch_{epoch}_crop_{start_idx}_{end_idx}.png"
        plt.savefig(heatmap_crop_path)
        plt.close()
        print(f"[INFO] Saved cropped Adaptive Adjacency heatmap -> {heatmap_crop_path}")
    else:
        print("[WARN] Model does not have nodevec1/nodevec2. Cannot plot adjacency matrix.")

        
def main():
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    # trainer 엔진 초기화
    engine = trainer(
        scaler=scaler,
        in_dim=args.in_dim,
        seq_length=args.seq_length,
        num_nodes=args.num_nodes,
        nhid=args.nhid,
        dropout=args.dropout,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        supports=supports,
        gcn_bool=args.gcn_bool,
        addaptadj=args.addaptadj,
        aptinit=adjinit
    )

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    # 결과 저장 폴더 (Adjacency Heatmap 등)
    results_dir = os.path.join(os.path.dirname(args.save), "results")
    os.makedirs(results_dir, exist_ok=True)

    for i in range(1, args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []

        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2-t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)

            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        val_time.append(s2 - s1)

        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        
        if args.addaptadj and (i % 10 == 0):
            save_adaptive_adj_heatmap(engine, i, results_dir)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)

        log = ('Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, '
               'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, '
               'Training Time: {:.4f}/epoch')
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse,
                         mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    best_model_path = f"{args.save}_epoch_{bestid+1}_{round(his_loss[bestid],2)}.pth"
    engine.model.load_state_dict(torch.load(best_model_path))
    print(f"[INFO] Loaded the best model: {best_model_path}")

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred, real)
        log = ('Evaluate best model on test data for horizon {:d}, '
               'Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}')
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(),
               f"{args.save}_exp{args.expid}_best_{round(his_loss[bestid],2)}.pth")

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
