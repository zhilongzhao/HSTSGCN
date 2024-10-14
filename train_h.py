import torch
import numpy as np
import argparse
import time
import util

# import matplotlib.pyplot as plt
from engine import *
import os
import shutil
import random


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/Jinan', help='data path')
parser.add_argument('--adjdata', type=str, default='data/Jinan/adj_mat.pkl', help='adj data path')
parser.add_argument('--adjdatacluster', type=str, default='data/Jinan/adj_mat_cluster.pkl', help='adj data path')
parser.add_argument('--transmit', type=str, default='data/Jinan/transmit.csv', help='data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--in_dim_cluster', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=561, help='number of nodes')
parser.add_argument('--cluster_nodes', type=int, default=20, help='number of cluster')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0000, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=50, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument("--force", type=str, default=True, help="remove params dir", required=False)
parser.add_argument('--save', type=str, default='./garage/chengdu', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--model', type=str, default='H_GCN', help='adj type')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
parser.add_argument('--d_model', type=int, default=64, help='')
parser.add_argument('--q', type=int, default=8, help='')
parser.add_argument('--v', type=int, default=8, help='')
parser.add_argument('--m', type=int, default=8, help='')
parser.add_argument('--N', type=int, default=1, help='')

parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=False,
                    help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
parser.add_argument('--end_channels', type=int, default=128, help='end channels')
parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')
parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=3, help='dim of nodes')

args = parser.parse_args()
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():
    # load data
    early_stop = 0
    minl = 1e5
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    sensor_ids_cluster, sensor_id_to_ind_cluster, adj_mx_cluster = util.load_adj(args.adjdatacluster, args.adjtype)
    # load_adj 返回的邻接矩阵是归一化后的拉普拉斯矩阵 以及邻接矩阵转置后的拉普拉斯矩阵
    dataloader = util.load_dataset_cluster(args.data, args.batch_size, args.batch_size, args.batch_size)
    # scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]  # 将一个 Python 列表 adj_mx 中的每个元素转换为 PyTorch 的 Tensor
    supports_cluster = [torch.tensor(i).to(device) for i in adj_mx_cluster]
    transmit_np = np.float32(np.loadtxt(args.transmit, delimiter=','))
    transmit = torch.tensor(transmit_np).to(device)

    # 生成语义矩阵
    data = np.load('data/chengdu/original_data.npz')['data']
    data = data.squeeze()
    data = data.transpose()

    def z_score_normalization(data):
        mean = np.mean(data)
        std = np.std(data)
        normalized_data = (data - mean) / std
        return normalized_data

    data = z_score_normalization(data)
    data = torch.tensor(data).to(device)

    correlation_matrix = torch.zeros(args.num_nodes, args.num_nodes).to(device)
    for i in range(args.num_nodes):
        for j in range(args.num_nodes):
            # 获取传感器i和传感器j的数据
            x = data[i]
            y = data[j]

            # 计算皮尔逊相关系数
            correlation = torch.nn.functional.cosine_similarity(x, y, dim=0)

            # 将相关系数存入相关系数矩阵中
            correlation_matrix[i, j] = correlation

    value_1 = torch.tensor(1).to(correlation_matrix.device)
    value_0 = torch.tensor(0).to(correlation_matrix.device)

    correlation_matrix = torch.where(correlation_matrix > 0.8, value_1, value_0)
    # # 将相关系数矩阵展平为一维向量
    # flat_correlation = correlation_matrix.flatten()
    #
    # # 找到前20个最大值的索引
    # top_indices = torch.topk(flat_correlation, k=40).indices
    #
    # # 将相关系数矩阵中前20大的值置为1，其余置为0
    # correlation_matrix = torch.zeros(args.num_nodes, args.num_nodes).to(device)
    # for idx in top_indices:
    #     # 将一维索引转换为二维索引
    #     i, j = idx // args.num_nodes, idx % args.num_nodes
    #     correlation_matrix[i, j] = 1
    #     correlation_matrix[j, i] = 1  # 由于是对称矩阵，需要同时置为1

    # 生成区域层语义矩阵
    Rdata = np.load('data/chengdu/original_data_cluster.npz')['data']
    # Rdata = Rdata[:,:,0]
    Rdata = Rdata[:, :, 1]
    Rdata = Rdata.squeeze()
    Rdata = Rdata.transpose()

    def z_score_normalization(Rdata):
        mean = np.mean(Rdata)
        std = np.std(Rdata)
        normalized_data = (Rdata - mean) / std
        return normalized_data

    Rdata = z_score_normalization(Rdata)
    Rdata = torch.tensor(Rdata).to(device)
    Rcorrelation_matrix = torch.zeros(args.cluster_nodes, args.cluster_nodes).to(device)
    for i in range(args.cluster_nodes):
        for j in range(args.cluster_nodes):
            # 获取传感器i和传感器j的数据
            x = Rdata[i]
            y = Rdata[j]

            # 计算皮尔逊相关系数
            correlation = torch.nn.functional.cosine_similarity(x, y, dim=0)

            # 将相关系数存入相关系数矩阵中
            Rcorrelation_matrix[i, j] = correlation
        # semantic_adj = (correlation_matrix + 1) / 2

    value_1 = torch.tensor(1).to(Rcorrelation_matrix.device)
    value_0 = torch.tensor(0).to(Rcorrelation_matrix.device)

    Rcorrelation_matrix = torch.where(Rcorrelation_matrix > 0.8, value_1, value_0)

        # # 将相关系数矩阵展平为一维向量
        # flat_correlation = Rcorrelation_matrix.flatten()
        #
        # # 找到前20个最大值的索引
        # top_indices = torch.topk(flat_correlation, k=40).indices
        #
        # # 将相关系数矩阵中前20大的值置为1，其余置为0
        # Rcorrelation_matrix = torch.zeros(args.cluster_nodes, args.cluster_nodes).to(device)
        # for idx in top_indices:
        #     # 将一维索引转换为二维索引
        #     i, j = idx // args.cluster_nodes, idx % args.cluster_nodes
        #     Rcorrelation_matrix[i, j] = 1
        #     Rcorrelation_matrix[j, i] = 1  # 由于是对称矩阵，需要同时置为1

    print(args)

    if args.model == 'HSTSGCN':
        engine = trainer7(args.in_dim, args.in_dim_cluster, args.seq_length, args.num_nodes, args.cluster_nodes,
                          args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, supports_cluster, transmit,
                          args.decay, args.q, args.v, args.m, args.N,
                          args.gcn_true, args.buildA_true, args.gcn_depth, args.dilation_exponential,
                          args.conv_channels, args.residual_channels, args.skip_channels,
                          args.end_channels, args.layers, args.propalpha, args.tanhalpha, args.subgraph_size,
                          args.node_dim, correlation_matrix, Rcorrelation_matrix
                          )
    # elif args.model=='H_GCN_wdf':
    #     engine = trainer6( args.in_dim,args.in_dim_cluster, args.seq_length, args.num_nodes,args.cluster_nodes,args.nhid, args.dropout,
    #                      args.learning_rate, args.weight_decay, device, supports,supports_cluster,transmit,args.decay
    #                      )
    # check parameters file
    params_path = args.save + "/" + args.model
    if os.path.exists(params_path) and not args.force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()

        dataloader['train_loader_cluster'].shuffle()

        for iter, (x, y, x_cluster, y_cluster) in enumerate(dataloader['train_loader_cluster'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            trainx_cluster = torch.Tensor(x_cluster).to(device)
            trainx_cluster = trainx_cluster.transpose(1, 3)
            trainy_cluster = torch.Tensor(y_cluster).to(device)
            trainy_cluster = trainy_cluster.transpose(1, 3)
            metrics = engine.train(trainx, trainx_cluster, trainy[:, 0, :, :], trainy_cluster)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])

        # engine.scheduler.step()
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()

        for iter, (x, y, x_cluster, y_cluster) in enumerate(dataloader['val_loader_cluster'].get_iterator()):
            validx = torch.Tensor(x).to(device)
            validx = validx.transpose(1, 3)
            validy = torch.Tensor(y).to(device)
            validy = validy.transpose(1, 3)
            validx_cluster = torch.Tensor(x_cluster).to(device)
            validx_cluster = validx_cluster.transpose(1, 3)
            validy_cluster = torch.Tensor(y_cluster).to(device)
            validy_cluster = validy_cluster.transpose(1, 3)
            with torch.no_grad():
                metrics = engine.eval(validx, validx_cluster, validy[:, 0, :, :], validy_cluster)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                         mvalid_rmse, (t2 - t1)), flush=True)

        # save the best model for this run over epochs
        if mvalid_loss < minl:
            torch.save(engine.model.state_dict(),
                       params_path + "/" + args.model + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
            print("BEST MODEL UPDATE")
            minl = mvalid_loss
            early_stop = 0
        else:
            early_stop = early_stop + 1

        if early_stop > 10:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(
        params_path + "/" + args.model + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    engine.model.eval()

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)

    realy = realy.transpose(1, 3)[:, 0, :, :]
    # print(realy.shape)
    for iter, (x, y, x_cluster, y_cluster) in enumerate(dataloader['test_loader_cluster'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testx_cluster = torch.Tensor(x_cluster).to(device)
        testx_cluster = testx_cluster.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx, testx_cluster)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())
    for iter, (x, y, x_cluster, y_cluster) in enumerate(dataloader['test_loader_cluster'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testx_cluster = torch.Tensor(x_cluster).to(device)
        testx_cluster = testx_cluster.transpose(1, 3)
        with torch.no_grad():
            _ = engine.model(testx, testx_cluster)
        break

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    # print(yhat.shape)
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    prediction = yhat
    for i in range(12):
        pred = prediction[:, :, i]
        # pred = scaler.inverse_transform(yhat[:,:,i])
        # prediction.append(pred)
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(), params_path + "/" + args.model + "_exp" + str(args.expid) + "_best_" + str(
        round(his_loss[bestid], 2)) + ".pth")
    prediction_path = params_path + "/" + args.model + "_prediction_results"
    ground_truth = realy.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    # spatial_at=spatial_at.cpu().detach().numpy()
    # parameter_adj=parameter_adj.cpu().detach().numpy()
    np.savez_compressed(
        os.path.normpath(prediction_path),
        prediction=prediction,

        ground_truth=ground_truth
    )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))

