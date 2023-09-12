import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler
import data_parser
from data_parser import convert_jaad_dict_to_df, get_JAAD_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from temporal_conv import ConvTemporalGraphical
from Graph import Graph

class st_gcn(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 time_dim=1,
                 joints_dim=25,
                 dropout=0,
                 bias=True):

        super(st_gcn, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)
        self.gcn = ConvTemporalGraphical(time_dim, joints_dim)  # the convolution layer
        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:

            self.residual = nn.Sequential(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )


        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res = self.residual(x)
        x = self.gcn(x, A)
        x = x.permute(0,1,3,2)
        x = self.tcn(x) + res
        x = self.prelu(x)
        return x, A


class CNN_layer(
    nn.Module):  # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        super(CNN_layer, self).__init__()
        self.kernel_size = kernel_size
        padding = (
        (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)  # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            , nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        output = self.block(x)
        return output

class PedestrianGraphNetwork(nn.Module):
    def __init__(self, kernel_size):
        super(PedestrianGraphNetwork,self).__init__()
        self.st_gcn = nn.ModuleList()
        self.txcnns = nn.ModuleList()
        self.st_gcn.append(st_gcn(1, 64, kernel_size))
        self.st_gcn.append(st_gcn(64,32, kernel_size))
        self.st_gcn.append(st_gcn(32,64,kernel_size))
        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        self.txcnns.append(CNN_layer(64,32,kernel_size,dropout=0.5)) # with kernel_size[3,3] the dimensinons of C,V will be maintained
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(32, 1)
        # self.fc = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(32)
        self.adjacency_matrix = torch.tensor(Graph().A, dtype=torch.float32)
    def forward(self, x):
        for gcn in (self.st_gcn):
            x,_ = gcn(x,self.adjacency_matrix)

        for tcnn in (self.txcnns):
            x = tcnn(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.bn(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x





# data_dir = ".../JAAD_per_person/JAAD_per_person/"
lst = get_JAAD_data((data_dir))
X, Y = convert_jaad_dict_to_df(get_JAAD_data(data_dir))
Y = Y.drop(columns=['cross'])

#————————————————————————————————————————————————————————————————————————————————————————————————————————
handwaving = "...\data\handwaving"
handclipping = "...\data\handclipping"
#
# drop_third_element = lambda lst: [x for i, x in enumerate(lst, start=1) if i % 3 != 0]
arr_handwaving = data_parser.read_all_folder(handwaving)
arr_handclipping = data_parser.read_all_folder(handclipping)
data_hw = [arr_handwaving[i].flatten() for i in range(arr_handwaving.shape[0])]
data_hc = [arr_handclipping[i].flatten() for i in range(arr_handclipping.shape[0])]
data_hand = data_hw + data_hc
df_hand = pd.DataFrame(data_hand)
scaler = MinMaxScaler()
scaler.fit(df_hand)
arr_hand = scaler.transform(df_hand)
label_array = np.array([1, 0, 1, 0])
label_df = pd.DataFrame(np.tile(label_array,(len(data_hand), 1)), columns=['look', 'action', 'hand_gesture', 'nod'])

X = pd.concat([pd.DataFrame(X), pd.DataFrame(arr_hand)], ignore_index=True)
X = X.values
Y = pd.concat([Y,label_df], ignore_index=True)

#————————————————————————————————————————————————————————————————————————————————————————————————————————
feature_matrix = X.reshape(-1, 25, 2)
feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
expanded_X = torch.unsqueeze(feature_matrix, 1)
labels = Y


X_train, X_test, y_train, y_test = train_test_split(expanded_X, labels, test_size=0.4, random_state=42)
X_train = torch.tensor(X_train.clone(), dtype=torch.float32)
X_test = torch.tensor(X_test.clone(), dtype=torch.float32)
# print(X_test.shape)
for col in y_train.columns:
    model = PedestrianGraphNetwork(tuple([3,3]))
    Y_train = torch.tensor(y_train[col].values, dtype=torch.float)
    # print("Y_train:",Y_train.shape)
    Y_test = torch.tensor(y_test[col].values, dtype=torch.float)
    print(col)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.001)
    num_epochs = 30
    batch_size = 64
    Xtrain_loader = DataLoader(X_train, batch_size=batch_size, shuffle=False)
    Ytrain_loader = DataLoader(Y_train, batch_size=batch_size, shuffle=False)
    model.train()
    for epoch in range(num_epochs):
        for batch1, batch2 in zip(Xtrain_loader,Ytrain_loader):
            optimizer.zero_grad()
            outputs = model(batch1)
            loss = F.binary_cross_entropy_with_logits(outputs, batch2[:, None], weight=None)
            loss.backward()
            optimizer.step()
    model.eval()
    Xtest_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)
    Ytest_loader = DataLoader(Y_test, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        test_accuracy = 0.0
        num_test_samples = 0
        y_true = []
        y_pred = []
        for batch1, batch2 in zip(Xtest_loader, Ytest_loader):
            outputs = model(batch1)
            outputs = torch.where(outputs >= 0.5, torch.tensor(1), torch.tensor(0))
            y_true.extend(batch2.tolist())
            y_pred.extend(outputs.squeeze().tolist())
            test_accuracy += (outputs == batch2[:, None]).sum().item()
            num_test_samples += batch2.size(0)
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        precision = TP / (TP+FP) if (TP+FP) != 0 else 0
        recall = TP / (TP+FN) if (TP+FN) != 0 else 0
        f1_score = (2*(precision*recall))/(precision+recall) if (precision+recall) != 0 else 0
        conf_mat = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        disp.plot()
        plt.title(col+':gcn')
        plt.show()
        test_accuracy /= num_test_samples
        print('Test Accuracy: {:.2f}%'.format(test_accuracy * 100))
        print('F1_score: {:.2f}%'.format(f1_score*100))
