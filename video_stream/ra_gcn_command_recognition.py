import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchvision import transforms

action_class = {'A001': 'Stop', 'A002': 'Go Right', 'A003': 'Go Left', 'A004': 'Come Here', 'A005': 'Follow me',
                'A006': 'Go Away', 'A007': 'Agree', 'A008': 'Disagree', 'A009': 'Go there', 'A010': 'Get Attention',
                'A011': 'Be Quiet', 'A012': 'Dont Know', 'A013': 'Turn Around', 'A014': 'Take This',
                'A015': 'Pick Up', 'A016': 'Standing Still', 'A017': 'Being Seated', 'A018': 'Walking Towards',
                'A019': 'Walking Away', 'A020': 'Talking on Phone'}
lab_to_num = {'Stop': 0, 'Go Right': 1, 'Go Left': 2, 'Come Here': 3, 'Follow me': 4, 'Go Away': 5, 'Agree': 6, 'Disagree': 7,
              'Go there': 8, 'Get Attention': 9, 'Be Quiet': 10, 'Dont Know': 11, 'Turn Around': 12, 'Take This': 13,
              'Pick Up': 14, 'Standing Still': 15, 'Being Seated': 16, 'Walking Towards': 17, 'Walking Away': 18,
              'Talking on Phone': 19}
joint_dict = {'Nose': 0, 'LEye': 1, 'REye': 2, 'LEar': 3, 'REar': 4, 'LShoulder': 5, 'RShoulder': 6, 'LElbow': 7,
              'RElbow': 8, 'LWrist': 9, 'RWrist': 10, 'LHip': 11, 'RHip': 12, 'LKnee': 13, 'RKnee': 14,
              'LAnkle': 15, 'RAnkle': 16}

labels_to_learn = ['Stop', 'Standing Still', 'Follow me']

class Graph:
    def __init__(self, max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.center = self._get_edge()

        # get adjacency matrix
        self.hop_dis = self._get_hop_distance()

        # normalization
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    @staticmethod
    def _get_edge():
        num_node = 17
        neighbor_1base = [(1, 2), (1, 3), (2, 4), (3, 5), (6, 1), (6, 8),
                          (7, 1), (7, 9), (8, 10), (9, 11), (12, 6), (12, 14),
                          (13, 7), (13, 15), (14, 16), (15, 17)]
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        center = 0
        return (num_node, edge, center)

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        return A

    @staticmethod
    def _normalize_digraph(A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD



class Data_transform():
    def __init__(self, data_transform=True):
        self.data_transform = data_transform

    def __call__(self, x):
        if self.data_transform:
            C, T, V, M = x.shape
            x_new = np.zeros((C*3, T, V, M))
            x_new[:C,:,:,:] = x
            for i in range(T-1):
                x_new[C:(2*C),i,:,:] = x[:,i+1,:,:] - x[:,i,:,:]
            for i in range(V):
                x_new[(2*C):,:,i,:] = x[:,:,i,:] - x[:,:,1,:]
            return x_new
        else:
            return x


class Occlusion_part():
    def __init__(self, occlusion_part=[]):
        self.occlusion_part = occlusion_part

        self.parts = dict()
        self.parts[1] = np.array([7, 9, 11])              # left arm
        self.parts[2] = np.array([6, 8, 10])           # right arm
        self.parts[3] = np.array([10, 11])                  # two hands
        self.parts[4] = np.array([12, 13, 14, 15, 16, 17])  # two legs
        self.parts[5] = np.array([0, 1, 2, 3, 4, 5])                  # head

    def __call__(self, x):
        for part in self.occlusion_part:
            x[:,:,self.parts[part],:] = 0
        return x


class Occlusion_time():
    def __init__(self, occlusion_time=0):
        self.occlusion_time = int(occlusion_time // 2)

    def __call__(self, x):
        if not self.occlusion_time == 0:
            x[:,(50-self.occlusion_time):(50+self.occlusion_time),:,:] = 0
        return x


class RA_GCN(nn.Module):
    def __init__(self, data_shape, num_class, A, drop_prob, gcn_kernel_size, model_stream):
        super().__init__()

        C, T, V, M = data_shape
        self.register_buffer('A', A)

        # baseline
        self.stgcn_stream = nn.ModuleList((
            ST_GCN(data_shape, num_class, A, drop_prob, gcn_kernel_size)
            for _ in range(model_stream)
        ))

        # mask
        self.mask_stream = nn.ParameterList([
            nn.Parameter(torch.ones(T * V))
            for _ in range(model_stream)
        ])


    def forward(self, inp):

        # multi stream
        out = 0
        feature = []
        for stgcn, mask in zip(self.stgcn_stream, self.mask_stream):
            x = inp

            # mask
            N, C, T, V, M = x.shape
            x = x.view(N, C, -1)
            x = x * mask[None,None,:]
            x = x.view(N, C, T, V, M)

            # baseline
            temp_out, temp_feature = stgcn(x)

            # output
            out += temp_out
            feature.append(temp_feature)
        return out, feature


class ST_GCN(nn.Module):
    def __init__(self, data_shape, num_class, A, drop_prob, gcn_kernel_size):
        super().__init__()

        C, T, V, M = data_shape
        self.register_buffer('A', A)

        # data normalization
        self.data_bn = nn.BatchNorm1d(C * V * M)

        # st-gcn networks
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_layer(C, 64, gcn_kernel_size, 1, A, drop_prob, residual=False),
            st_gcn_layer(64, 64, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(64, 64, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(64, 64, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(64, 128, gcn_kernel_size, 2, A, drop_prob),
            st_gcn_layer(128, 128, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(128, 128, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(128, 256, gcn_kernel_size, 2, A, drop_prob),
            st_gcn_layer(256, 256, gcn_kernel_size, 1, A, drop_prob),
            st_gcn_layer(256, 256, gcn_kernel_size, 1, A, drop_prob),
        ))

        # edge importance weights
        self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(A.shape)) for _ in self.st_gcn_networks])

        # fcn
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)

        # extract feature
        _, c, t, v = x.shape
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # global pooling
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(N, -1)

        return x, feature


class st_gcn_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, A, drop_prob=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        # spatial network
        self.gcn = SpatialGraphConv(in_channels, out_channels, kernel_size[1]+1)

        # temporal network
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0],1), (stride,1), padding),
            nn.BatchNorm2d(out_channels),
        )

        # residual
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(out_channels))

        # output
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        # residual
        res = self.residual(x)

        # spatial gcn
        x = self.gcn(x, A)

        # temporal 1d-cnn
        x = self.tcn(x)

        # output
        x = self.relu(x + res)
        return x


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super().__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = s_kernel_size

        # weights of different spatial classes
        self.conv = nn.Conv2d(in_channels, out_channels * s_kernel_size, kernel_size=1)

    def forward(self, x, A):

        # numbers in same class have same weight
        x = self.conv(x)

        # divide into different classes
        n, kc, t, v = x.shape
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()
        return x