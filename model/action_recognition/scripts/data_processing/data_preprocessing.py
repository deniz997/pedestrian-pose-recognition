import numpy as np

from sklearn.preprocessing import MinMaxScaler

def read_txt_file(txt_path):
    with open(txt_path, 'r') as f:
        filenames = f.readlines()
    return [name.strip() for name in filenames]

#train_files = read_txt_file('D:\APP-RAS\dataset\handwaving_out\TRAIN.txt')
#test_files = read_txt_file('D:\APP-RAS\dataset\handwaving_out\TEST.txt')
#base_path='D:\APP-RAS\dataset\handwaving_out'
# handwaving_tr = data_parser.read_selected_folder(base_path,train_files)
# handwaving_te = data_parser.read_selected_folder(base_path,test_files)

#Centering:8th point as original point
def centering_data(data):
    center_point = data[:,8,:]
    centered_dataset = data - center_point[:,np.newaxis,:]
    return centered_dataset

#Scaling
scaler = MinMaxScaler()

# def apply_scaler(array):
#     for i in range(array.shape[0]):
#         array[i] = scaler.fit_transform(array[i])
#     return array
def apply_scaler_to_dataset(dataset):
    reshaped_data = dataset.reshape(-1,2)
    scaler.fit(reshaped_data)
    for i in range(dataset.shape[0]):
        dataset[i] = scaler.fit_transform(dataset[i])
    return dataset


