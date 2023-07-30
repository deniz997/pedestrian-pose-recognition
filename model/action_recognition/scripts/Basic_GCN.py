import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
import scipy.sparse as sp
import matplotlib.pyplot as plt
from torch.nn import functional as F
from data_parser import convert_jaad_dict_to_df, get_data
from torch.utils.data import Dataset, DataLoader
from scripts import st__gcn
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    data_dir = "D:/APP-RAS/JAAD_JSON_Labels_new/JAAD_JSON_Labels/"
    lst = get_data((data_dir))
    print(lst[:10])
    X, Y = convert_jaad_dict_to_df(get_data(data_dir))

# Define the adjacency matrix
connect_points = [[0, 1], [0, 15],[0, 16],[1, 2],[1, 5],[1, 8],[5, 6],[6, 7],[2, 3],[3, 4],[12, 13],[13, 14],[9, 10],[10, 11],[8, 9],[8, 12],[14, 21],[11, 24],[14, 20],[20, 19],[11, 22],[22, 23],[16, 18],[15, 17]]


def adjacency_matrix_construction(lst):
    ini_matrix = np.zeros((25, 25))
    identity_matrix = np.identity(25)
    for [i, j] in connect_points:
        ini_matrix[i, j] = 1
        ini_matrix[j, i] = 1
    A_matrix_hat = ini_matrix + identity_matrix
    return A_matrix_hat


adjacency_matrix_hat = adjacency_matrix_construction(connect_points)
# Degree matrix
degree_matrix = np.array(adjacency_matrix_hat.sum(1))
D_new = degree_matrix[:, None]
# print(D_new.shape)
D_ = np.power(D_new, -0.5).flatten()
D_diag = sp.diags(D_)
# Normalize the adjacency matrix
# normalized_adjacency_matrix_hat = np.linalg.inv(degree_matrix) @ adjacency_matrix_hat
adjacency_matrix = (D_diag@adjacency_matrix_hat)@D_diag
# Plot adjacency_matrix
fig = plt.figure(figsize=(20, 20))
ax = fig.gca()
alpha = ['Nose', 'Neck', 'RShoulder', 'RElbow','RWrist','LShoulder','LElbow','MidhHip',"RHip","RKnee","RAnkle","LHip",'Lknee','LAnkle','REye',"LEye","REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe","RSmallToe","RHeel"]
cax = ax.matshow(adjacency_matrix)
fig.colorbar(cax)
ax.set_xticks(np.arange(len(alpha)), labels=alpha)
ax.set_yticks(np.arange(len(alpha)), labels=alpha)
plt.show()


feature_matrix = X.reshape(-1, 25, 2)
feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
expanded_X = torch.unsqueeze(feature_matrix, 1)
labels = Y


class PedestrianGraphNetwork(nn.Module):
    def __init__(self, adjacency_matrix):
        super(PedestrianGraphNetwork, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True)
        )
        # self.conv_block1 = st__gcn.ConvTemporalGraphical()


        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

    def forward(self, x):
        x = torch.matmul(self.adjacency_matrix, x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# Create an instance of the PedestrianGraphNetwork
X_train, X_test, y_train, y_test = train_test_split(expanded_X, labels, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
print(X_test.shape)
for col in y_train.columns:
    model = PedestrianGraphNetwork(adjacency_matrix)
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

        for batch1, batch2 in zip(Xtest_loader, Ytest_loader):
            outputs = model(batch1)
            outputs = torch.where(outputs >= 0.5, torch.tensor(1), torch.tensor(0))
            test_accuracy += (outputs == batch2[:, None]).sum().item()
            num_test_samples += batch2.size(0)

        test_accuracy /= num_test_samples
        print('Test Accuracy: {:.2f}%'.format(test_accuracy * 100))
