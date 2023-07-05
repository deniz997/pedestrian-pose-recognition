import numpy as np
import sklearn.metrics as metrics
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain

from model.data_parser import convert_jaad_dict_to_df, get_JAAD_data

if __name__ == '__main__':

    data_dir = "C:/Users/max00/Documents/PoseRecognition/pedestrian-pose-recognition/data/JAAD_JSON_Labels/"

    X, Y = convert_jaad_dict_to_df(get_JAAD_data(data_dir))

    # SVM Classifier

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    sp_X_train = sparse.lil_matrix(X_train.values)
    sp_Y_train = sparse.lil_matrix(Y_train.values)
    sp_X_test = sparse.csr_matrix(X_test.values)
    sp_Y_test = sparse.csr_matrix(Y_test.values)

    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense,Input

    # define model
    print(X.shape)
    features, coordinate_values = X_train.shape
    model = Sequential()
    model.add(LSTM(64, input_shape=(coordinate_values,1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # output layer，units is the unit number for output

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    for col in Y_train.columns:
        y_train = Y_train[col].to_numpy()
        y_test = Y_test[col].to_numpy()
        model.fit(X_train, y_train, epochs=11, batch_size=32)

        # prediction
        predictions = model.predict(X_test)

        print(predictions)
        print("Evaluate on test data for col: " + col)
        results = model.evaluate(X_test, y_test, batch_size=128)
        print(dict(zip(model.metrics_names, results)))

    # train model
    # model.fit(X_train, Y_train, epochs=11, batch_size=32)
    #
    # # prediction
    # predictions = model.predict(X_test)
    #
    # print("Evaluate on test data")
    # results = model.evaluate(X_test, Y_test, batch_size=128)
    # print(dict(zip(model.metrics_names, results)))
    # i = 0
    # round_pred = np.round(predictions).astype(int)
    # for col in Y_test.columns:
    #     print(col + " accury:")
    #     print(metrics.accuracy_score(Y_test[col], round_pred[:,1]))
    #     i += 1
        # Y_test_new = Y_test.to_numpy()
        # y_prediction = np.argmax(round_pred[:,1])
        # y_test = np.argmax(Y_test_new[:,1], axis=1)
        # result = confusion_matrix(y_test, y_prediction, normalize='pred')
        # print(result)