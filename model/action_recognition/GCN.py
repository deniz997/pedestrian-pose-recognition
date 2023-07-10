import torch
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_parser import convert_jaad_dict_to_df, get_data
if __name__ == '__main__':

    data_dir = "D:/APP-RAS/JAADLabels/"

    X, Y = convert_jaad_dict_to_df(get_data(data_dir))
print(Y.shape)
print(type(Y))
# 1. Prepare your dataset
feature_matrix = X.reshape(-1, 25, 2)
labels = Y


# 2. Define the adjacency matrix
connect_points = [[0, 1], [0, 15],[0, 16],[1, 2],[1, 5],[1, 8],[5, 6],[6, 7],[2, 3],[3, 4],[12, 13],[13, 14],[9, 10],[10, 11],[8, 9],[8, 12],[14, 21],[11, 24],[14, 20],[20, 19],[11, 22],[22, 23],[16, 18],[15, 17]]


def adjacency_matrix_construction(lst):
    ini_matrix = np.zeros((25, 25))
    identity_matrix = np.identity(25)
    for [i, j] in connect_points:
        ini_matrix[i, j] = 1
        ini_matrix[j, i] = 1
        # identity_matrix[i, j] = 1
        # identity_matrix[j, i] = 1
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
# 3. Create the feature matrix
feature_matrix = feature_matrix
# 4. Build the GCN model


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.gc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = self.gc1(x)
        x = self.relu(x)
        x = torch.tensor(x)
        adj = torch.tensor(adj, dtype=torch.float)
        x = torch.matmul(adj, x)  # Graph convolution
        x = self.gc2(x)
        return x


# Define model parameters
input_dim = 2  # Number of 2D coordinate values
hidden_dim = 128
output_dim = 5  # Number of action categories

model = GCN(input_dim, hidden_dim, output_dim)

# 5. Train and evaluate the model
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.2, random_state=42)

# Convert the data to sparse matrix
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train, adjacency_matrix)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test, adjacency_matrix)
    _, predicted = torch.max(outputs.data, 1)
    predicted[torch.where(predicted != 1)] = 0
    accuracy = (predicted == y_test).sum().item() / (y_test.size(0)*y_test.size(1))
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
