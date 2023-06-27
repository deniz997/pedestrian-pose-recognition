#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import numpy as np


def get_data(file_dir):
    """
    Will read JSON files from the provided file directory and subdirectories and return a list of dicts
    :param file_dir: directory where to look for input JSON files
    """
#     print('Starting to read JAAD json files!')
    json_list = []

    for dirpath, dirnames, filenames in os.walk(file_dir):
        JAAD_list = [pos_json for pos_json in filenames if pos_json.endswith('.json')]
        for js in JAAD_list:
            with open(os.path.join(dirpath, js)) as json_file:
                json_list.append(json.load(json_file))

    return json_list


# In[2]:


#lst_new:all the filename
def file(folder_path):
    with open(folder_path) as file:
        lst=[]
        for eachline in file:
            eachline=eachline.rstrip('\n')
            lst.append(eachline)
        lst_new=lst[1:-1]
    return lst_new       


# In[3]:


json_list1=get_data('JAAD_output_JSON/video_0001')
json_list2=get_data('JAAD_output_JSON/video_0002')


# In[4]:


def get_index(filename):
    name=file(filename)
    index=[]
    for i in name:
        index.append(int(i[-4:]))
    return index


# In[5]:


# index of the videos
index=get_index('JAAD_output_JSON/LIST.TXT')


# In[6]:


#create a dictionary which could save variables from json_list1 to json_list346
def define_Dict(filename):
    D={}
    lst=file(filename)
    index=get_index(filename)
    for num,i in enumerate(lst):
        json_list_name = f"json_list{index[num]}"
        D[json_list_name]=get_data(f'JAAD_output_JSON/{lst[num]}')
    return D


# In[7]:


Dict=define_Dict('JAAD_output_JSON/LIST.TXT')


# In[8]:


#input:file content
# return:(25,2)data
#in this function the coordinate data would be extracted
def transform_to_arr(lst):
    lst_x=[]
    lst_y=[]
    for num,i in enumerate(lst):
        if num%3==0:
            lst_x.append(i)
        if num%3==1:
            lst_y.append(i)
    data=np.stack((np.array(lst_x),np.array(lst_y)),axis=-1)
    return data


# In[9]:


data0=transform_to_arr(json_list1[0]['people'][0]['pose_keypoints_2d'])


# In[10]:


# def stacked_data(lst):
#     arr=np.empty((0,2))
#     data=[]
#     for i in range(len(lst)):
#         if 'people' in lst[i]:
#             for j in range(len(lst[i]['people'])):
#                 arr=transform_to_arr(lst[i]['people'][j]['pose_keypoints_2d'])
#                 print(j)
#                 data.append(arr)
#         elif 'pose_keypoints_2d' in lst[i]:
#             arr=transform_to_arr(lst[i]['pose_keypoints_2d'])
#             data.append(arr)
#     stacked_data=np.stack(data,axis=0)
#     return stacked_data


# In[11]:


#only extract the first person's infomation
def stacked_data(lst):
    arr=np.empty((0,2))
    data=[]
    for i in range(len(lst)):
        if 'people' in lst[i] and lst[i]['people']!=[]:
            arr=transform_to_arr(lst[i]['people'][0]['pose_keypoints_2d'])
            data.append(arr)
        elif 'pose_keypoints_2d' in lst[i]:
            arr=transform_to_arr(lst[i]['pose_keypoints_2d'])
            data.append(arr)
    stacked_data=np.stack(data,axis=0)
    return stacked_data


# In[12]:


import matplotlib.pyplot as plt

def plot_skeleton2D(skeleton, connect_points=None):
    fig = plt.figure(figsize=(4,10))
    ax = fig.add_subplot(111)
    x = skeleton[:, 0]
    y = skeleton[:, 1]
    ax.scatter(x, y, c='b', marker='o')
    if connect_points:
        for points in connect_points:
            x_points = x[points]
            y_points = y[points]
            ax.plot(x_points, y_points, c='r')
    for i,(xi,yi) in enumerate(zip(x,y)):
        ax.annotate(str(i),(xi,yi), textcoords="offset points", xytext=(0,10), ha='center')
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
connect_points=[[0,1],[0,15],[0,16],[1,2],[1,5],[1,8],[5,6],[6,7],[2,3],[3,4],[12,13],[13,14],[9,10],[10,11],[8,9],[8,12],[14,21],[11,24],[14,20],[20,19],[11,22],[22,23],[16,18],[15,17]]


# In[13]:


plot_skeleton2D(data0,connect_points)


# In[21]:


#find outliers
#data:stacked data e.g(623,25,2)
def make_zero_to_mean(data):
    data_new=data
    for i in range(data.shape[0]):
        if np.any(np.all(data[i]==[0,0],axis=1)):
            outliers=np.where((data[i][:,0]==0)&(data[i][:,1]==0))
            #calculate the mean without outliers
            lst=list(outliers[0])
            for j in lst:
                data_new=np.delete(data[i],int(j),axis=0)
            new_points_mean=np.mean(data_new,axis=0)
            #replace outliers[0,0] with new__mean
            data[i][outliers] = new_points_mean
        else:
            data[i]=data[i]
    return data


# In[22]:


#Min-Max-Scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
def MinMaxScaling(data):
    data_new=make_zero_to_mean(data)
    for i in range(data_new.shape[0]):
        data_new[i]=scaler.fit_transform(data_new[i])
    return data_new


# In[23]:


# #Min-Max-Scaling
# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler()
# def MinMaxScaling(data):
#     for i in range(data.shape[0]):
#         data[i]=scaler.fit_transform(data[i])
#     return data


# In[24]:


def get_x_data_inGeneral(D):
    lst=[]
    for i in index:
        data=stacked_data(D[f'json_list{i}'])
        data_preprocessing=MinMaxScaling(data)
        lst.append(data_preprocessing)
    return lst


# In[25]:


stacked_data(Dict['json_list54']).shape


# In[26]:


# lst constains all of the 145 videos,each shape is (N,25,2)
lst=get_x_data_inGeneral(Dict)
len(lst)


# In[29]:


#video243:run
plot_skeleton2D(lst[100][51],connect_points)


# In[30]:


#video294:standing
plot_skeleton2D(lst[126][78],connect_points)


# In[31]:


#video2:walking
plot_skeleton2D(lst[1][140],connect_points)


# In[32]:


#video1:looking at Vehicles and show hands
plot_skeleton2D(lst[0][350],connect_points)


# ## Semi-supervised learning,make some labels manuelly firstly

# In[33]:


#stack all data together
X=np.empty((1,25,2))
for i in lst:
    X=np.concatenate((X,i),axis=0)


# In[34]:


#make labels,initial:all equal for -1
Y=-1*np.ones((23408))


# In[35]:


#input is the index of videos:for example:3 is video_0003
#return:where this video data begin in the whole dataset
def find_current_index(ind):
    num=index.index(ind)
    summ=0
    for i in range(num):
        summ+=lst[i].shape[0]
    return summ


# In[36]:


find_current_index(346),lst[-1].shape


# In[37]:


#standing:0
#walking:1
#running:2
#holding phone:3
#looking at AV:4
#showing hands to AV:5


# In[38]:


#make some labels manuelly
#video0:
Y[0:146]=0
Y[156:337]=0
Y[400:528]=1
Y[147:149]=5
#video2:
Y[find_current_index(2)+5]=0
Y[find_current_index(2)+17:find_current_index(2)+19]=0
Y[find_current_index(2)+20:find_current_index(2)+177]=1
#video3:
Y[find_current_index(3):find_current_index(4)]=1
#video9:
Y[find_current_index(9)+30:find_current_index(9)+37]=0
Y[find_current_index(9):find_current_index(9)+18]=1
Y[find_current_index(9)+43:find_current_index(9)+75]=1
#video10:
Y[find_current_index(10)+15:find_current_index(10)+19]=1
#video11:
Y[find_current_index(11)+15:find_current_index(11)+19]=0
Y[find_current_index(11)+51:find_current_index(11)+94]=1
#video12:
Y[find_current_index(12)]=1
#video14/17/21:
Y[find_current_index(14):find_current_index(23)]
#video42:
Y[find_current_index(42):find_current_index(44)]=1
#video90:
Y[find_current_index(90):find_current_index(90)+48]=0
Y[find_current_index(90)+48:find_current_index(90)+208]=1
#video102:
Y[find_current_index(102)+76:find_current_index(102)+249]=3
#video107:
Y[find_current_index(107)+7:find_current_index(107)+42]=0
Y[find_current_index(107)+43:find_current_index(107)+139]=1
#video123:
Y[find_current_index(123)+27:find_current_index(123)+116]=2
Y[find_current_index(123):find_current_index(123)+4]=4
#video140:
Y[find_current_index(140):find_current_index(140)+33]=0
Y[find_current_index(140)+33:find_current_index(140)+166]=1
#video158:
Y[find_current_index(158):find_current_index(158)+16]=0
Y[find_current_index(158)+38:find_current_index(158)+163]=1
#video243:
Y[find_current_index(243)+20:find_current_index(158)+91]=2
#video294:
Y[find_current_index(294)+29:find_current_index(294)+76]=0
Y[find_current_index(294)+80:find_current_index(294)+142]=1
#video295:
Y[find_current_index(295)+121]=3
Y[find_current_index(295)+219:find_current_index(295)+281]=3
#video304:
Y[find_current_index(304)+63:find_current_index(304)+91]=0
#video308:
Y[find_current_index(308):find_current_index(312)]=1
#video315
Y[find_current_index(315)+101:find_current_index(315)+184]=1
Y[find_current_index(315)+3]=4
Y[find_current_index(315)+7]=4
#video318
Y[find_current_index(318):find_current_index(318)+235]=1
#video329
Y[find_current_index(329):find_current_index(329)+91]=1
Y[find_current_index(329)+91:find_current_index(329)+131]=4
#video339
Y[find_current_index(339)+8:find_current_index(339)+21]=0
Y[find_current_index(339)+127:find_current_index(339)+177]=1


# In[39]:


#SK-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[40]:


# Label Spreading
label_prop_model = LabelSpreading()


# In[41]:


X_new=X.reshape(23408,50)
X_new_=X_new[:10000]
Y=np.array(Y,dtype=int)
Y_new_=Y[:10000]


# In[42]:


#fit to Label Spreading & reduce the array from 23408 to 20000
label_prop_model.fit(X_new_,Y_new_)

# Predict the Labels for Unlabeled Samples
pred_lb = label_prop_model.predict(X_new_)


# In[43]:


# Feature& Target  Dataset
X_ = X_new_
y_ = Y_new_  # labels predicted by Label Spreading

#Dataset Split  [train = 90%, test = 10%]
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.1, random_state=0) 

#Define Model
model = RandomForestClassifier(verbose = 0, max_depth=2, random_state=0)

#Fit
model.fit(X_train,y_train)

#Prediction
rf_pred = model.predict(X_test)
#Accuracy Score
acc = accuracy_score(y_test, rf_pred)
print("Random Forest Model Accuracy (after Label Spreading): ",'{:.2%}'.format(acc))


# In[44]:


X_train.shape


# In[46]:


#dataset suit for LSTM Model
X_train_new=X_train.reshape(9000,25,2)
Y_train_new=pred_lb[:int(0.9*pred_lb.shape[0])]
X_test_new=X_test.reshape(1000,25,2)
Y_test_new=pred_lb[int(0.9*pred_lb.shape[0]):]


# ### LSTM Model

# In[57]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Input

# define model
timesteps,features,coordinate_values=X_train_new.shape
model = Sequential()
model.add(LSTM(64,input_shape=(features, coordinate_values)))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))  # output layer，units is the unit number for output

# compile model
optimizer = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['accuracy'])

# train model
model.fit(X_train_new,Y_train_new,epochs=10,batch_size=32) 

# prediction
predictions = model.predict(X_test_new)

round_pred=np.round(predictions).astype(int)


# ### Attention+LSTM

# In[59]:


from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense,Input,Attention

# define model
timesteps,features,coordinate_values=X_train_new.shape
inputs=layers.Input(shape=(features,coordinate_values))
lstm_output=layers.LSTM(64,return_sequences=True)(inputs)
attention=layers.Attention()([lstm_output,lstm_output])
dense_output=layers.Dense(32,activation='tanh')(attention)#full connect layer
output=layers.Dense(1,activation='relu')(dense_output)

model=keras.Model(inputs=inputs,outputs=output)
# compile model
optimizer = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['accuracy'])

# train model
model.fit(X_train_new,Y_train_new, epochs=10, batch_size=32) 

# prediction
predictions = model.predict(X_test_new)


# ### RNN

# In[60]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense,Input

# define model
timesteps,features,coordinate_values=X_train_new.shape
model = Sequential()
model.add(SimpleRNN(64,input_shape=(features, coordinate_values)))
model.add(Dense(32,activation='tanh'))  #Hideen layer
model.add(Dense(1,activation='relu'))# output layer，units is the unit number for output

# compile model
optimizer = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['accuracy'])

# train model
model.fit(X_train_new,Y_train_new, epochs=10, batch_size=32) 

# prediction
predictions = model.predict(X_test_new)


# In[ ]:




