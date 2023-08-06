import numpy as np
import tensorflow as tf
from keras.layers import LSTM, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tcn import TCN, tcn_full_summary

from model.data_parser import convert_jaad_dict_to_df, get_JAAD_data

if __name__ == '__main__':
    data_dir = "C:/Users/max00/Documents/PoseRecognition/pedestrian-pose-recognition/data/JAAD_per_person/"

    X, Y = convert_jaad_dict_to_df(get_JAAD_data(data_dir))

    # SVM Classifier

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

    features, coordinate_values = X_train.shape

    batch_size, time_steps, input_dim = None, 150, coordinate_values
    tcn_layer = TCN(input_shape=(coordinate_values, 1))
    # The receptive field tells you how far the model can see in terms of timesteps.
    print('Receptive field size =', tcn_layer.receptive_field)

    m_2 = Sequential([
        tcn_layer,
        Dense(4, activation='sigmoid')
    ])

    m_2.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=[tf.keras.metrics.F1Score(
        average='weighted', threshold=0.5, name='f1_score', dtype=None)])

    tcn_full_summary(m_2, expand_residual_blocks=False)

    m_2.fit(X_train, Y_train, epochs=10, steps_per_epoch=1000)
    predictions = m_2.predict(X_test)
    print("Complete")
    results = m_2.evaluate(X_test, Y_test)
    print(dict(zip(m_2.metrics_names, results)))
    i = 0
    for col in Y_train.columns:

        cm = confusion_matrix((Y_test.to_numpy())[:, i], np.round(predictions[:, i]))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(col + ' tcn')
        plt.show()
        i += 1


    # define model
    features, coordinate_values = X_train.shape
    model = Sequential()
    model.add(LSTM(64, input_shape=(coordinate_values, 1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))  # output layer，units is the unit number for output

    # compile model
    model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.F1Score(
                      average='weighted', threshold=0.5, name='f1_score', dtype=None)])

    model.fit(X_train, Y_train, epochs=20, batch_size=32)

    # prediction
    predictions = model.predict(X_test)

    results = model.evaluate(X_test, Y_test, batch_size=128)
    print(dict(zip(model.metrics_names, results)))

    i = 0
    for col in Y_train.columns:

        cm = confusion_matrix((Y_test.to_numpy())[:, i], np.round(predictions[:, i]))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(col + 'lstm')
        plt.show()
        i += 1

    a = 0