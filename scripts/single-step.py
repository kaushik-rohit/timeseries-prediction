import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

from nsepy import get_history
from datetime import date

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM, GRU, Conv1D, MaxPooling1D
from operator import add
import json
#'ULTRACEMCO', 'CIPLA', 'ACC', 'HDFC',, 'HCLTECH', 'JSWSTEEL', 'MARUTI'
symbols = ['INFY', 'BHARTIARTL', 'AXISBANK' ]
window_size = [3, 5, 7, 9, 11, 13, 15]

def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))

def window_transform(time_series, window_size):
    X = []
    y = []
    for i in range(len(time_series) - window_size):
        X.append(time_series[i:i+window_size])
        y.append(time_series[i+window_size])
        
    return np.array(X), np.array(y)

def make_ann_model(window_size):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(16, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def make_lstm_model(window_size):
    model_lstm = Sequential()
    model_lstm.add(LSTM(256, input_shape=(window_size, 1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(128, input_shape=(window_size, 1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1, activation = 'linear'))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    
    return model_lstm
    
def make_cnn_model(window_size):
	model = Sequential()
	model.add(Conv1D(filters=64,kernel_size=2,activation='relu',input_shape=(window_size,1)))
	model.add(Conv1D(filters=32,kernel_size=1,activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	#model.add(Dense(32,activation='relu'))
	model.add(Flatten())
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam')
	
	return model
	
def make_gru_model(window_size):
    model = Sequential()
    model.add(GRU(256, input_shape=(window_size, 1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(128, input_shape=(window_size, 1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model


def test_model(model, X_train, X_test, y_train, y_test):
	y_pred_test_ann = model.predict(X_test)
	y_train_pred_ann = model.predict(X_train)
	r2_train = mse(y_train, y_train_pred_ann)
	r2_test = mse(y_test, y_pred_test_ann)
	
	return r2_train, r2_test
	
if __name__ == '__main__':

	result = []

	for sym in symbols:

		json_descriptor = {'stock':sym,'ann':[], 'gru':[], 'lstm':[], 'cnn':[]}

		try:
			data = get_history(symbol=sym, start=date(2002,1,1), end=date(2019,1,15))
		except:
			print('----------------Sleeping for 5 minutes. Too much Load-------------------------')
			time.sleep(300)
			data = get_history(symbol=sym, start=date(2002,1,1), end=date(2019,1,15))
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
		

		for win_sz in window_size:
			ann_result = []
			gru_result = []
			lstm_result = []
			cnn_result = []
			pred_ANN = []
			pred_LSTM = []
			pred_GRU = []
			pred_CNN = []
			for i in range(5):
				#''''''''ANN'''''''''''''''''''
				X_train, y_train = window_transform(train_sc, win_sz)
				X_test, y_test = window_transform(test_sc, win_sz)

				model = make_ann_model(win_sz)
				early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
				history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, callbacks=[early_stop], shuffle=False)
				
				train_acc, test_acc = test_model(model, X_train, X_test, y_train, y_test)
				y_pred_test_ANN = model.predict(X_test)
				
				ann_result.append(test_acc)
				#''''''''ANN''''''''''''''''''''
				
				
				#''''''''CNN''''''''''''''''''''
				model = make_cnn_model(win_sz)
				early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
				history_model_cnn = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, shuffle=False, callbacks=[early_stop])
				
				train_acc, test_acc = test_model(model, X_train, X_test, y_train, y_test)
				y_pred_test_CNN = model.predict(X_test)
				cnn_result.append(test_acc)
				#''''''''CNN''''''''''''''''''''
				
				
				X_tr_t = X_train.reshape(X_train.shape[0], win_sz, 1)
				X_tst_t = X_test.reshape(X_test.shape[0], win_sz, 1)
				
				#''''''''''LSTM''''''''''''''''''
				model = make_lstm_model(win_sz)
				early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
				history_model_lstm = model.fit(X_tr_t, y_train, epochs=200, batch_size=32, verbose=1, shuffle=False, callbacks=[early_stop])
				
				train_acc, test_acc = test_model(model, X_tr_t, X_tst_t, y_train, y_test)
				y_pred_test_LSTM = model.predict(X_tst_t)
				lstm_result.append(test_acc)
				#"''''''''LSTM'''''''''''''''''''
				
				#''''''''GRU'''''''''''''''''''''
				model = make_gru_model(win_sz)
				early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
				history_model_gru = model.fit(X_tr_t, y_train, epochs=200, batch_size=32, verbose=1, shuffle=False, callbacks=[early_stop])
				
				train_acc, test_acc = test_model(model, X_tr_t, X_tst_t, y_train, y_test)
				y_pred_test_GRU = model.predict(X_tst_t)
				gru_result.append(test_acc)
				#''''''''GRU'''''''''''''''''''''
				
				pred_ANN.append(y_pred_test_ANN)
				pred_CNN.append(y_pred_test_CNN)
				pred_LSTM.append(y_pred_test_LSTM)
				pred_GRU.append(y_pred_test_GRU)
			#update optimal window size param
			ann.append([win_sz, min(ann_result), np.mean(ann_result), np.std(ann_result)])
			gru.append([win_sz, min(gru_result), np.mean(gru_result), np.std(gru_result)])
			lstm.append([win_sz, min(lstm_result), np.mean(lstm_result), np.std(lstm_result)])
			cnn.append([win_sz, min(cnn_result), np.mean(cnn_result), np.std(cnn_result)])
			
			plot_ann = [0]*len(pred_ANN[0])
			for pred in pred_ANN:
				plot_ann = list(map(add, plot_ann, pred))
				
			for i in range(len(plot_ann)):
				plot_ann[i] = plot_ann[i]/5
			
			plot_lstm = [0]*len(pred_LSTM[0])
			for pred in pred_LSTM:
				plot_lstm = list(map(add, plot_lstm, pred))
				
			for i in range(len(plot_lstm)):
				plot_lstm[i] = plot_lstm[i]/5
				
			plot_gru = [0]*len(pred_GRU[0])
			for pred in pred_GRU:
				plot_gru = list(map(add, plot_gru, pred))
				
			for i in range(len(plot_gru)):
				plot_gru[i] = plot_gru[i]/5
				
			plot_cnn = [0]*len(pred_CNN[0])
			for pred in pred_CNN:
				plot_cnn = list(map(add, plot_cnn, pred))
				
			for i in range(len(plot_cnn)):
				plot_cnn[i] = plot_cnn[i]/5
				
				
			#save prediction plots
			plt.plot(y_test, label='True Values', color='black')
			plt.plot(plot_ann, label='ANN Prediction', color='red')
			plt.plot(plot_cnn, label='CNN Prediction', color='blue')
			plt.plot(plot_lstm, label='LSTM Prediction', color='green')
			plt.plot(plot_gru, label='GRU Prediction', color='yellow')
			plt.title("Prediction")
			plt.xlabel('Time')
			plt.ylabel('Normalized Stock Prices')
			plt.legend()
			plt.savefig('./plots/' + sym + ' ' + ' window_sz ' + str(win_sz))
			plt.clf()
			
			
		json_descriptor['ann'] = ann
		json_descriptor['lstm'] = lstm
		json_descriptor['cnn'] = cnn
		json_descriptor['gru'] = gru 
		result.append(json_descriptor)
		
		with open('data.json', 'w') as f:
			json.dump(result, f)
		#json.dumps(result)
