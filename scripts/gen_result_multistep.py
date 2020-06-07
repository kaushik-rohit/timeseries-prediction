import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, GRU, Conv1D, MaxPooling1D
from keras.utils.vis_utils import plot_model
from operator import add
import json
#'ACC', 'HCLTECH', 'JSWSTEEL', 'MARUTI','AXISBANK', 'INFY', 
symbols = [ 'HDFC', 'INFY', 'BHARTIARTL', 'ULTRACEMCO',
           'CIPLA']

# (x, y) here x indicates the input size and y indicates the prediction size
# (30, 7) means we predict next 7 days stocks using previous 30 days prices
window_size = [(30, 7), (30, 14), (60, 7), (60, 14)]


def adj_r2_score(r2, n, k):
    return 1 - ((1 - r2) * ((n - 1) / (n - k - 1)))


def window_transform(series, input_size, output_size):
    X = []
    y = []

    slen = len(series)
    n_windows = slen - (input_size + output_size) + 1

    for i in range(n_windows):
        X.append(series[i:i + input_size])
        y.append(series[i+input_size: i+input_size+output_size])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1])
    y = y.reshape(y.shape[0], y.shape[1])
    return X, y


def test_window_transform(series, input_size, output_size):
    X = []
    y = []

    slen = len(series)

    j = input_size
    while (j+output_size) < slen:
        X.append(series[j - input_size:j])
        y.append(series[j:j+output_size])
        j = j + output_size

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1])
    y = y.reshape(y.shape[0], y.shape[1])
    return X, y


def make_ann_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(input_size,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def make_cnn_model(input_size, output_size):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', input_shape=(input_size, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(output_size, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def make_lstm_model(input_size, output_size):
    model_lstm = Sequential()
    model_lstm.add(LSTM(256, input_shape=(input_size, 1), activation='relu', kernel_initializer='lecun_uniform',
                        return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(128, input_shape=(input_size, 1), activation='relu', kernel_initializer='lecun_uniform',
                        return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(output_size, activation='linear'))
    opt = keras.optimizers.Adam(clipnorm=1.0)
    model_lstm.compile(loss='mean_squared_error', optimizer=opt)

    return model_lstm


def make_gru_model(input_size, output_size):
    model = Sequential()
    model.add(GRU(256, input_shape=(input_size, 1), activation='relu', kernel_initializer='lecun_uniform',
                  return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(128, input_shape=(input_size, 1), activation='relu', kernel_initializer='lecun_uniform',
                  return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_size, activation='linear'))
    opt = keras.optimizers.Adam(clipnorm=1.0)
    model.compile(loss='mean_squared_error', optimizer=opt)

    return model


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    true_ = [item for sublist in y_test for item in sublist]
    pred_ = [item for sublist in y_pred for item in sublist]

    error = mse(true_, pred_)

    return error, pred_


def plot_all_model():
    print('plotting_model')
    ann_model = make_ann_model(3)
    cnn_model = make_cnn_model(3)
    lstm_model = make_lstm_model(3)
    gru_model = make_gru_model(3)

    plot_model(ann_model, to_file='ann_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(cnn_model, to_file='cnn_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(lstm_model, to_file='lstm_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(gru_model, to_file='gru_plot.png', show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print('invalid number of arguments')
    elif len(sys.argv) == 2:
        if sys.argv[1] == 'plot':
            print('plotting')
            plot_all_model()
    else:
        result = []

        for sym in symbols:

            json_descriptor = {'stock': sym, 'ann': [], 'gru': [], 'lstm': [], 'cnn': []}

            data = pd.read_csv('../data/{}.csv'.format(sym), index_col=0)
            print('data loaded for {}!!'.format(sym))
            
            data.index = pd.to_datetime(data.index)
            data_frame = data.copy()
            df = data_frame[["Close"]]

            split_date = pd.Timestamp('01-01-2017')

            train = df.loc[:split_date]
            test = df.loc[split_date:]

            sc = MinMaxScaler()
            train_sc = sc.fit_transform(train)
            test_sc = sc.transform(test)

            ann = []
            gru = []
            lstm = []
            cnn = []

            for in_size, out_size in window_size:
                ann_result = []
                gru_result = []
                lstm_result = []
                cnn_result = []
                pred_ANN = []
                pred_LSTM = []
                pred_GRU = []
                pred_CNN = []
                for i in range(5):
                    # ''''''''ANN'''''''''''''''''''
                    X_train, y_train = window_transform(train_sc, in_size, out_size)
                    print('X train: ', X_train.shape)
                    print('y train: ', y_train.shape)
                    X_test, y_test = test_window_transform(test_sc, in_size, out_size)
                    print('X test: ', X_test.shape)
                    print('y test: ', y_test.shape)
                    model = make_ann_model(in_size, out_size)
                    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
                    history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, callbacks=[early_stop],
                                        shuffle=False)

                    error, pred_ann = test_model(model, X_test, y_test)

                    ann_result.append(error)
                    # ''''''''ANN''''''''''''''''''''

                    # ''''''''CNN''''''''''''''''''''
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
                    print('X train: ', X_train.shape)
                    print('y train: ', y_train.shape)
                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
                    print('X train: ', X_train.shape)
                    print('y train: ', y_train.shape)
                    model = make_cnn_model(in_size, out_size)
                    print(model.summary())
                    early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
                    history_model_cnn = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, shuffle=False,
                                                  callbacks=[early_stop])

                    error, pred_cnn = test_model(model, X_test, y_test)
                    cnn_result.append(error)
                    # ''''''''CNN''''''''''''''''''''

                    # ''''''''''LSTM''''''''''''''''''
                    model = make_lstm_model(in_size, out_size)
                    print(model.summary())
                    early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
                    history_model_lstm = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1,
                                                   shuffle=False, callbacks=[early_stop])

                    error, pred_lstm = test_model(model, X_test, y_test)
                    lstm_result.append(error)
                    # "''''''''LSTM'''''''''''''''''''

                    # ''''''''GRU'''''''''''''''''''''
                    model = make_gru_model(in_size, out_size)
                    early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
                    history_model_gru = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, shuffle=False,
                                                  callbacks=[early_stop])

                    error, pred_gru = test_model(model, X_test, y_test)
                    gru_result.append(error)
                    # ''''''''GRU'''''''''''''''''''''

                    pred_ANN.append(pred_ann)
                    pred_CNN.append(pred_cnn)
                    pred_LSTM.append(pred_lstm)
                    pred_GRU.append(pred_gru)
                # update optimal window size param
                ann.append([in_size, out_size, min(ann_result), np.mean(ann_result), np.std(ann_result)])
                gru.append([in_size, out_size, min(gru_result), np.mean(gru_result), np.std(gru_result)])
                lstm.append([in_size, out_size, min(lstm_result), np.mean(lstm_result), np.std(lstm_result)])
                cnn.append([in_size, out_size, min(cnn_result), np.mean(cnn_result), np.std(cnn_result)])

                plot_ann = [0] * len(pred_ANN[0])
                for pred in pred_ANN:
                    plot_ann = list(map(add, plot_ann, pred))

                for i in range(len(plot_ann)):
                    plot_ann[i] = plot_ann[i] / 5

                plot_lstm = [0] * len(pred_LSTM[0])
                for pred in pred_LSTM:
                    plot_lstm = list(map(add, plot_lstm, pred))

                for i in range(len(plot_lstm)):
                    plot_lstm[i] = plot_lstm[i] / 5

                plot_gru = [0] * len(pred_GRU[0])
                for pred in pred_GRU:
                    plot_gru = list(map(add, plot_gru, pred))

                for i in range(len(plot_gru)):
                    plot_gru[i] = plot_gru[i] / 5

                plot_cnn = [0] * len(pred_CNN[0])
                for pred in pred_CNN:
                    plot_cnn = list(map(add, plot_cnn, pred))

                for i in range(len(plot_cnn)):
                    plot_cnn[i] = plot_cnn[i] / 5

                # save prediction plots
                plt.plot(test_sc, '-', label='True Values', color='#1b9e77')
                plt.plot(plot_ann, label='MLP Prediction', color='#d95f02')
                plt.plot(plot_cnn, ':', label='CNN Prediction', color='#7570b3')
                plt.plot(plot_lstm, label='LSTM Prediction', color='#e7298a')
                plt.plot(plot_gru, label='GRU Prediction', color='#66a61e')
                plt.title("Prediction")
                plt.xlabel('Time')
                plt.ylabel('Normalized Stock Prices')
                plt.legend()
                plt.savefig('../plots/' + sym + ' ' + ' in_sz ' + str(in_size) + ' out_sz ' + str(out_size))
                plt.clf()

            json_descriptor['ann'] = ann
            json_descriptor['lstm'] = lstm
            json_descriptor['cnn'] = cnn
            json_descriptor['gru'] = gru
            result.append(json_descriptor)

            with open('data.json', 'w') as f:
                json.dump(result, f)
