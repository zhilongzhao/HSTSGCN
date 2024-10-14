import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from layer import *

import numpy as np
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, InstanceNorm2d

# from utils import ST_BLOCK_0 #ASTGCN
# from utils import ST_BLOCK_1 #DGCN_Mask/DGCN_Res
# from utils import ST_BLOCK_2_r #DGCN_recent
#
# from utils import ST_BLOCK_4 #Gated-STGCN
# from utils import ST_BLOCK_5 #GRCN
# from utils import ST_BLOCK_6 #OTSGGCN
# from utils import multi_gcn #gwnet
# from utils import GCNPool #H_GCN
from utils import Transmit
from utils import gate
# from utils import GCNPool_dynamic
# from utils import GCNPool_h
# from utils import T_cheby_conv_ds_1
# from utils import dynamic_adj
# from utils import SATT_h_gcn
from sparse_activations import Sparsemax
from fuseAttetion import MultiHeadAttention

"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
"""


class HSTSGCN(nn.Module):
    def __init__(self, device, num_nodes, cluster_nodes, dropout=0.3, supports=None, supports_cluster=None,
                 in_dim=1, in_dim_cluster=2, out_dim=12, transmit=None, residual_channels=32,
                 skip_channels=64, end_channels=128, q=8, v=8, m=8, N=1, gcn_true=True, buildA_true=True, gcn_depth=2,
                 dilation_exponential=1, conv_channels=32, layers=3, propalpha=0.05, tanhalpha=3, subgraph_size=20,
                 node_dim=40, semantic_adj=None,Rsemantic_adj=None):
        super(HSTSGCN, self).__init__()
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.transmit = transmit
        self.cluster_nodes = cluster_nodes
        self.A = supports[0]
        self.A_cluster = supports_cluster[0]
        self.semantic_adj = semantic_adj
        self.Rsemantic_adj = Rsemantic_adj

        self.start_conv0 = nn.Conv2d(in_channels=in_dim, out_channels=conv_channels, kernel_size=(1, 1))
        self.start_conv1 = nn.Conv2d(in_channels=in_dim_cluster, out_channels=conv_channels, kernel_size=(1, 1))
        self.start_conv2 = nn.Conv2d(in_channels=conv_channels * 2, out_channels=in_dim, kernel_size=(1, 1))

        self.end_conv0 = nn.Conv2d(in_channels=in_dim, out_channels=conv_channels, kernel_size=(1, 1))
        self.end_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=conv_channels, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(in_channels=conv_channels * 2, out_channels=in_dim, kernel_size=(1, 1))

        self.end_conv_1 = nn.Conv2d(in_channels=conv_channels*2,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=in_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.stblock1 = stblock1(gcn_true=gcn_true, buildA_true=buildA_true, gcn_depth=gcn_depth, num_nodes=num_nodes,
                                 device=device, predefined_A=self.A, static_feat=None, dropout=dropout,
                                 subgraph_size=subgraph_size, node_dim=node_dim,
                                 dilation_exponential=dilation_exponential, conv_channels=conv_channels,
                                 residual_channels=residual_channels, skip_channels=skip_channels,
                                 end_channels=end_channels, seq_length=out_dim, in_dim=in_dim, out_dim=out_dim,
                                 layers=layers, propalpha=propalpha, tanhalpha=tanhalpha, layer_norm_affline=True, q=q,
                                 v=v, m=m)
        self.stblock2 = stblock2(gcn_true=gcn_true, buildA_true=buildA_true, gcn_depth=gcn_depth,
                                 num_nodes=cluster_nodes,
                                 device=device, predefined_A=self.A_cluster, static_feat=None, dropout=dropout,
                                 subgraph_size=subgraph_size, node_dim=node_dim,
                                 dilation_exponential=dilation_exponential,
                                 conv_channels=conv_channels, residual_channels=residual_channels,
                                 skip_channels=skip_channels, end_channels=end_channels, seq_length=out_dim,
                                 in_dim=in_dim_cluster,
                                 out_dim=out_dim, layers=layers, propalpha=propalpha, tanhalpha=tanhalpha,
                                 layer_norm_affline=True, q=8, v=8, m=8)

        self.matchconv = nn.Conv2d(in_channels=cluster_nodes, out_channels=num_nodes, kernel_size=(1, 1), stride=(1, 1))
        self.conv1x1 = nn.Conv2d(conv_channels * 2, conv_channels, 1)
        self.attention1 = MultiHeadAttention(conv_channels, q, v, m);
        self.atconv = nn.Conv2d(in_channels=conv_channels * 2, out_channels=1, kernel_size=(1, 1))
        self.Biatconv = nn.Conv2d(in_channels=1, out_channels=conv_channels * 2, kernel_size=(1, 1))


        self.bn = BatchNorm2d(in_dim, affine=False)
        self.bn2 = BatchNorm2d(conv_channels, affine=False)
        self.bn3 = BatchNorm2d(conv_channels * 2, affine=False)
        self.bn_cluster = BatchNorm2d(in_dim_cluster, affine=False)
        self.transmit1 = Transmit(conv_channels, out_dim, transmit, num_nodes, cluster_nodes)
        self.gate1 = gate(2 * conv_channels)

        self.idx1 = torch.arange(self.num_nodes).to(device)
        self.idx2 = torch.arange(self.cluster_nodes).to(device)

    def forward(self, input, input_cluster):
        x = self.bn(input)  # 归一化
        # shape=x.shape#64 1 792 12
        input_c = input_cluster
        x_c = self.bn_cluster(input_c)
        x_cluster = self.bn_cluster(input_c)  # 64 1 50 12

        # network
        transmit = self.transmit  # 792 50
        transmit = transmit.permute(1, 0)
        x = self.start_conv0(x)  # 64 32 792 12
        x_cluster = self.start_conv1(x_cluster)  # 64 32 50 12
        transmit1 = self.transmit1(x, x_cluster)  # 64 792 50
        x_1 = (torch.einsum('bmn,bcnl->bcml', transmit1, x_cluster))  # 64 792 50 *64 32 50 12 = 64 32 792 12
        x = self.gate1(x, x_1)  # 拼接x,x_att
        x = self.start_conv2(x)
        x_cluster = self.stblock2(x_c, self.idx2,self.Rsemantic_adj)
        x = self.stblock1(x, self.idx1, self.semantic_adj)
        x = x.transpose(1, 3)
        x_cluster = x_cluster.transpose(1, 3)
        x = self.end_conv0(x)
        x_cluster = self.end_conv1(x_cluster)
        x = self.bn2(x)
        transmit2 = self.transmit1(x, x_cluster)
        x_2 = (torch.einsum('bmn,bcnl->bcml', transmit2, x_cluster))
        x = self.gate1(x, x_2)  # 64 64 32 12

        # output
        x = F.relu(x)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x.permute(0, 3, 2, 1)
        return x


class stblock1(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32,
                 residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=1, out_dim=12,
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, q=8, v=8, m=8):
        super(stblock1, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)

        self.attention1 = MultiHeadAttention(conv_channels, q, v, m);

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, idx=None, semantic_adj=None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A * semantic_adj

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x)

        # mutihead attention
        x1 = x.permute(0, 3, 2, 1)
        x_tem = self.attention1(x1, x1, x1)
        x1 = x_tem.permute(0, 3, 2, 1)
        x = x + x1

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class stblock2(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32,
                 residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=1, out_dim=12,
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, q=8, v=8, m=8):
        super(stblock2, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)

        self.seq_length = seq_length
        self.attention1 = MultiHeadAttention(conv_channels, q, v, m);
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (
                            dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, idx=None,Rsemantic_adj = None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A * Rsemantic_adj

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x)

        # x1 = x.permute(0, 3, 2, 1)
        # x_tem = self.attention1(x1, x1, x1)
        # x1 = x_tem.permute(0, 3, 2, 1)
        # x = x + x1

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x








