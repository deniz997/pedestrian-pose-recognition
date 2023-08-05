import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from data_parser import convert_jaad_dict_to_df, get_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from temporal_conv import ConvTemporalGraphical
from Graph import Graph

class PedestrianGraphNetwork(nn.Module):
    def __init__(self, kernel_size):
        super(PedestrianGraphNetwork, self).__init__()
        self.st_gcn_block1 = st_gcn(1, 64, kernel_size)
        self.st_gcn_block2 = st_gcn(64, 128, kernel_size)
        self.st_gcn_block3 = st_gcn(128, 128, kernel_size)
        self.st_gcn_block4 = st_gcn(128, 256, kernel_size)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(256, 1)
        # self.fc = nn.Conv2d(256, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        self.adjacency_matrix = torch.tensor(Graph().A, dtype=torch.float32)

    def forward(self, x):
        x = torch.matmul(self.adjacency_matrix, x)
        x, _ = self.st_gcn_block1(x, self.adjacency_matrix)
        x, _ = self.st_gcn_block2(x, self.adjacency_matrix)
        x, _ = self.st_gcn_block3(x, self.adjacency_matrix)
        x, _ = self.st_gcn_block4(x, self.adjacency_matrix)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


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


if __name__ == '__main__':

    data_dir = "D:/APP-RAS/JAADLabels/"
    # data_dir = "D:/APP-RAS/JAAD_JSON_Labels_new/JAAD_JSON_Labels/"
    lst = get_data((data_dir))
    X, Y = convert_jaad_dict_to_df(get_data(data_dir))

feature_matrix = X.reshape(-1, 25, 2)
feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
expanded_X = torch.unsqueeze(feature_matrix, 1)
labels = Y

X_train, X_test, y_train, y_test = train_test_split(expanded_X, labels, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
# print(X_test.shape)
for col in y_train.columns:
    model = PedestrianGraphNetwork(tuple([3,1]))
    Y_train = torch.tensor(y_train[col].values, dtype=torch.float)
    # print("Y_train:",Y_train.shape)
    Y_test = torch.tensor(y_test[col].values, dtype=torch.float)
    print(col)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.001)
    num_epochs = 30
    batch_size = 32
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
        conf_mat = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        disp.plot()
        plt.title(col+':gcn')
        plt.show()
        test_accuracy /= num_test_samples
        print('Test Accuracy: {:.2f}%'.format(test_accuracy * 100))
