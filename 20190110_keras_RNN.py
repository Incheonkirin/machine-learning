import os

print (os.getcwd())
os.chdir("C://Users/Hyungi/Desktop/workplace/datasets")

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Conv1D, Conv2D, Flatten,MaxPooling1D, MaxPooling2D, Dropout

X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])
X

model = Sequential()
model.add(Dense(100, activation='relu', input_dim= 3))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


model.fit(X, y, epochs= 2000, verbose= 0)


x_input = array([50, 60, 70])
x_input = x_input.reshape((1, 3))
yhat = model.predict(x_input, verbose=0)
print(yhat)


# CNN분류
# 여기서는 reshape가 들어간 이유?
#   데이터 입력 포멧 : sampling, timestemps, features 로 구성 해야 함.
X = X.reshape((X.shape[0], X.shape[1], 1))
model = Sequential()
# Conv1D = text에서 특징 추출
# filter size가 '그냥' 2입니다. 2 by2가 아닙니다.
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
# 여기서 필요한 가중치 맵 사이즈 : 64x2 = 128+ 64 = 192
# Pooling하면 어떻게 되나요? 위에서 나가는 차수가 어떻게 되나요?
# 2, 64로 나가고, pooling하게 되어지면 ?  1, 64가 되어집니다.
model.add(MaxPooling1D(pool_size=2)) # 1, 64
model.add(Flatten()) # Flatten 하면? 그냥 64가 남겠죠.

#여기서 Dense를 왜 넣었죠?  Deep하게 하기 위해서 넣은 것 입니다. layer를 하나 추가 한 것이에요.
#여기서의 가중치 사이즈는 어떻게 되나요?  64가 Flatten되어서 들어오죠? 그리고, 이게 50으로 나와야 하지요. 64 x 50이 필요하지요.
# 64 x 50 그다음에 + 50 (바이어스 50이 되어야 하니까요.) 그래서 내려가는 차수가 어떻게 되지요?
model.add(Dense(50, activation='relu')) # 50

model.add(Dense(1)) # 50 + 1이 되겠지요.

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=1000, verbose=0)

x_input = array([50, 60, 70])
x_input = x_input.reshape((1, 3, 1))
y_hat = model.predict(x_input, verbose=0)
print( yhat)


from keras.layers import LSTM

X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential()

# input_shape 의 3은 셀 수 = > 셀 3
# 셀이 갖는 특징 수 50

# status = 2개 * 2개 = 4개
# output
# input ( 데이터 들어오는 곳)

model.add(LSTM(50, activation='relu', input_shape=(3, 1)))

model.add(Dense(1)) # 50 x1
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs= 1000, verbose=0)

x_input = array([50, 60, 70])
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose= 0)
print(yhat)




다음은 Timedistribute를 보도록 하겠습니다.
## CNN- LSTM을 섞은 것 입니다.
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


X = array([[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70]])
y = array([50, 60, 70, 80])
X = X.reshape((X.shape[0], 2, 2, 1))
# [10, 20]
# [30, 40] 이미지형태처럼 2개씩 끊어서 특징 추출

model = Sequential()
# (1 x 64) => 2 x 64
# TimeDistributed 는 시간차를 이용해서 데이터를 모아주는 역할을 합니다.
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
        input_shape=(None, 2, 1)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
# Convolution 거쳐서 특징 뽑고, 다시 LSTM으로 특징을 추출
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)
model.summary()

x_input = array([50, 60, 70, 80])
x_input = x_input.reshape((1, 2, 2, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)

# TimeDistributed 만 이해하면 됩니다 !
# 왜 2번만 묶어 주나요? input shape에서 2개 로 지정했기 때문입니다



# RNN에서 이해해 보도록 하겠습니다.
##
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import  RepeatVector
from keras.layers import TimeDistributed
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([[40, 50], [50, 60], [60, 70], [70, 80]])
X = X.reshape((X.shape[0], X.shape[1], 1))
y = y.reshape((y.shape[0], y.shape[1], 1))
model= Sequential()
model.add(LSTM(100, activation='relu', input_shape=(3, 1)))
# RepeatVector 와 TimeDistributed가 연속적으로 사용되고 있는데,
# RepeatVector는 왜 사용되고 있지요?
# 같은벡터를 반복한다는 이야기에요.
# 지금까지는 데이터가 1개씩 나왔는데, 여기서는 2개 (40, 50) (50, 60) 등과 같이.
# 그래서 (입력데이터가)두번 반복한 거에요.

#return sequence로 하면, 셀마다   ouput 전체 출력.
#TimeDistributed는 데이터를 묶어서, 묶은 다음에 Dense해서 하나로 내보내고 있어요.
model.add(RepeatVector(2))

#셀이 몇개인가요?
# 입력데이터는 셀이 3개입니다.
# 셀에 나온 값들을 TimeDistributed 기다렸다가 1로 만들고,

model.add(LSTM(100, activation='relu', return_sequences= True)) #return sequnces하면? 셀 마다 값을 배출해준다고 했지요.
model.add(TimeDistributed(Dense(1))) # 100 x 1 + 1
# 예측 : keras에서는 예측까지 define 해주고, 나머지는 compile에서 됩니다.
model.compile(optimizer= 'adam', loss='mse')
model.summary()
model.fit(X, y, epochs=100, verbose=0)


x_input = array([50, 60, 70])
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose= 0)
print(yhat)


##
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
numpy.random.seed(7)
# 날짜, 승객수
# usecols로 컬럼을 제외. 1번째만 읽어드림.  skipfooter는 ?
dataframe = pandas.read_csv('pass.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

train_size = int(len(dataset) * 0.67)       #왜 0.67을 곱했지
test_size = len(dataset) - train_size
train, test = dataset[0: train_size, :], dataset[train_size: len(dataset), :]
print(len(train), len(test))

# 예측을 한다고 하면, 이전데이터를 사용을 하는데..
# 그래서 한칸씩, 이전데이터와 현재 데이터 짝을 맞춰야해.
def create_dataset(dataset, look_back=1): #lookback이 뭐지? 윈도우 사이즈를 이야기 합니다. window size 앞 데이터를 고려하는 범위
    dataX, dataY = [], [] # 하나의 데이터를 변수값과 예측값으로 분리하기 위해서.
    for i in range(len(dataset)- look_back-1):
        a = dataset[i: (i+ look_back), 0] # 고려할 범위.  1- 작년데이터가 올해 . 2- 작년데이터가 올해 내년,
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


look_back = 2
# 변수 갯수 2개
# 맨 처음건 이전데이터가 없음, 두번째 건 이전이전데이터가 없음.
#  2 + 1 부터 시작
trainX, trainY = create_dataset(train, look_back) # 67 %
# 앞에 변수 두개  y
#  0  0          y
#  0  0          y
#  0  0          y
#  0  0          y

testX, testY = create_dataset(test, look_back)# 33 %

model = Sequential()

# Dense 8 의 의미는 무엇인가요? 8 로 나간다는 의미입니다.
# 케라스에서는  무엇을 기준으로 ? ' 출력 차수 를  기준으로 합니다.'
# 8 로 나갔다는 것은 '확대'
model.add(Dense(8, input_dim= look_back, activation = 'relu'))
model.add(Dense(1))
# define 은 예측 까지입니다.
# 원래 코스트 펑션 구해서 빼주고 하는 절차 등이 여기서 compile로 해결됩니다.
model.compile(loss = 'mean_squared_error', optimizer= 'adam')

# epoch는 몇번 돌릴 지 ..
# batchsize 를 전체 데이터 수로 나눕니다..
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)


trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose= 0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan

#                앞에 두개는 없으니까,
trainPredictPlot[look_back: len(trainPredict) + look_back, :] = trainPredict
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+ (look_back*2) + 1: len(dataset) - 1, :] = testPredict
#                                           뒤에를 채우고 있어요.
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
# 파란색이 실제 데이터


##
#전기 소비량 예측

import numpy as np
import matplotlib.pyplot as plt
import os
import re
# DATA_DIR = "./data"
# 15분 마다 전기 소비량을 측정.
fld = open(os.path.join( 'elec2011_2014.txt'), 'rb')
data=[]
line_num = 0
cid= 250
# cid가 250인 것만 뽑은 것입니다.

for line in fld:
    if line.startswith(b"\"\";"):
        continue
    if line_num % 100 == 0:
        print("{:d} lines read".format(line_num))
    cols = [float(re.sub(b",",b".", x)) for x in
            line.strip().split(b";")[1:]] # 첫번째 열에 있는 날짜를 제외하라는 것 입니다.
    data.append(cols[cid]) # 전체데이터를 다 하는 것이 아니라,
    line_num += 1
fld.close()
NUM_ENTRIES = 1000
plt.plot(range(NUM_ENTRIES), data[0:NUM_ENTRIES])
plt.ylabel("전기소비량")
plt.xlabel("시간 (1pt = 15mins)")
plt.show()

# 전처리 한 데이터는 저장해 두어라는 명령입니다.
# numpy 전처리 한 데이터 확장자 npy
np.save(os.path.join(DATA_DIR, "LD_250.npy"), np.array(data))





## 그다음에 이 데이터를 분석해보도록 하겠습니다.

from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import os

DATA_DIR = './data'
data = np.load(os.path.join(DATA_DIR, 'LD_250.npy')) # 한 가구에 대한 데이터 @

STATELESS = True # False하게 되면 어떻게 되나요? STATEFUL하게 작동하게 됩니다.

NUM_TIMESTEPS = 20 # 이전 데이터 20개를 고려하겠다는 의미입니다. Window size가 되는 것입니다.
HIDDEN_SIZE = 10 # 특징 값 추출 되는 사이즈
BATCH_SIZE = 96 # costfunc 차이를 내서 back propagation내서   ==> sdg 의 개념
NUM_EPOCHS = 5 # 데이터가 많지 않기 때문에, 같은 데이터로 5번 진행하라는 의미입니다.



data = data.reshape(-1, 1) # -1 은 무슨 의미이죠?  '-'는 모든 차수를 의미하는 것이고, '1' 이 들어온 것은  "행을 열"로 바꾸겠다. 시계열데이터...
scaler = MinMaxScaler(feature_range=(0, 1), copy=False) # 신경망 정규화
data = scaler.fit_transform(data)

# scaler 역으로 inverse_transform


X = np.zeros((data.shape[0], NUM_TIMESTEPS)) # 0이면 '행 수' , NUM_TIMESTEPS 20개의 데이터를 만들었다.
Y = np.zeros((data.shape[0], 1))  # y값은 하나로..
for i in range(len(data) - NUM_TIMESTEPS -1):
    X[i] = data[i: i + NUM_TIMESTEPS].T
    Y[i] = data[i + NUM_TIMESTEPS + 1]


X = np.expand_dims(X, axis= 2) # 3차원 , 열 에 차원확장을 하여라.  차원확장을 하면 내려가 버립니다.    하나씩 끊어주는 것입니다..
sp = int(0.7 * len(data))
Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]



if STATELESS : # stateless & stateful 로 구분        2개의 망을 하나로 했다는 이야기 입니다.   # 20 개만 고려하면서 예측..
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, input_shape=(NUM_TIMESTEPS, 1), # 20, 1 NUM TIMESTEPS의 의미? 수직으로 들어오는 것에 처리하는 단위를 이야기하는 것입니다. '1'은 셀 수.
            return_sequences=False)) # return sequence는 뭐할때 쓰죠? many to many 할 때 쓰죠. 셀에서 나오는 것을 다 받겠다는 의미이죠.
    model.add(Dense(1))
else: # 여기는 stateful이 되는 것입니다.
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, stateful= True,#stateful 이있으면, 다 학습된 다음에 , 다음 단계 처리시 state를 입력으로 사용. - 수직적. 망과 망을 연결하는 것은 수평적.
            batch_input_shape=(BATCH_SIZE, NUM_TIMESTEPS, 1), # BATCH 사이즈가 나오죠.
            return_sequences= False))
    model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam',
        metrics=['mean_squared_error'] )




if STATELESS:#망과 망을(좌우로) 연결할 때, state로 연결.
    model.fit(Xtrain, Ytrain, epochs = NUM_EPOCHS, batch_size=BATCH_SIZE,
            validation_data = (Xtest, Ytest),
            shuffle= False)
else:#수직적으로. 앞단에서 훈련된 state를 뒷단으로 연결           # 배치사이즈가 중요하기 때문에, 배치사이즈에 대한 계산을 하고 있는 것입니다.
    train_size = (Xtrain.shape[0] // BATCH_SIZE) * BATCH_SIZE  # 전체 데이터 사이즈에 배치사이즈 .. 전체 사이즈가 되는 것이죠. 배치사이즈가 안되는 것은 처리안됩니다.
    test_size = (Xtest.shape[0] // BATCH_SIZE) * BATCH_SIZE
    Xtrain, Ytrain = Xtrain[0:train_size], Ytrain[0:test_size]
    Xtest, Ytest = Xtest[0:test_size], Ytest[0:test_size]
    print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
    for i in range(NUM_EPOCHS): # EPOCH할 때 마다 초기화 시킵니다. EPOCH시 state를 reset. 전체 데이터 처리하는 동안, state를 계속 뒷단으로 넘기는 것입니다. 그러니까 state가 살아 있는 것.
        print("Epoch {:d}/{:d}".format(i+1, NUM_EPOCHS)) # 뒷단으로 넘길때는 어떤 의미이겠습니까?  # 시계열 데이터. 윈도우사이즈 뿐만 아니라, 그 이전부터 계속해서 state가 넘어가는 것.
        model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=1,
                validation_data = (Xtest, Ytest),
                shuffle=False)
        model.reset_states()


score, _ = model.evaluate(Xtest, Ytest, batch_size= BATCH_SIZE)
rmse = math.sqrt(score)
print("\nMSE: {:.3f}, RMSE: {:.3f}".format(score, rmse))
# print("\naccuracyy: {:.3f}".format(accuracy))
# 아직 선형대수가 나오면 맨붕이죠. 아직 감각이 안온거죠.
# 계속 하다보면 적응 됩니다.
# 전체 형태를 볼 줄 알아야 해요.





## 비트코인 예측
import pandas as pd
import time
import seaborn as sns
import datetime
import numpy as np
from math import sqrt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
import lxml

bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype( 'int64')


bitcoin_market_info.info()

bitcoin_market_info = bitcoin_market_info[[
        'Date', 'Close**', 'Open*', 'High', 'Low', 'Volume', 'Market Cap']]
bitcoin_market_info.head()

# 날짜는 훈련에 필요없어서,제외시킴.
# 하루 전의 변수데이터로 다음날의 종가를 예측.
bitcoin_market_info.drop(['Date'], inplace=True, axis=1)

#신경망이니까, 민맥스스케일
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(bitcoin_market_info)

#컬럼이름 추출
columns = bitcoin_market_info.columns

#시계열 데이터 => 변수,
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1): # 변수의 변화율 shift (차이) 미분 = 차이를 구하는 미분.
        cols.append(df.shift(-i))
        names += [(j) for j in columns]
    for i in range(0, n_out):
        cols.append(df.shift(i))
        if i == 0:
            names += [('output ' + j) for j in columns]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(
                    n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg





reframed = series_to_supervised (scaled, 1, 1)
ref = series_to_supervised(bitcoin_market_info.values, 1, 1)
ref.drop(ref.columns[[7, 8, 9, 10, 11]], axis=1, inplace=True)
print(ref.head())
reframed.drop(reframed.columns[[7,8,9,10,11]], axis=1, inplace=True)


# 시계열 데이터 =>지도학습 데이터로 변경 ( 전날의 변수의 변화로 다음날의 종가를 예측함)
print(reframed.head()) # 변수가 6 => 7개 (output close **) 종가 예측

values = reframed.values
train = values[9: 1500, :]
test = values[1500 :, :]


#              마지막 하나 전까지,  마지막에 있는 하나
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
type(train_X)

# 차수 증가 => LSTM의 입력형태와 맞추기 위해서.
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# 1491, 1(step수), 6(변수)
# LSTM에 들어가는 데이터 형태 samples, timestemps, feature => 28 by 28 이미지였다면 ?  feature 는 28 이 되겠지요.

model = Sequential()
model.add(LSTM(80, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')




history = model.fit(train_X, train_y, epochs=50, batch_size=32,
        validation_data=(test_X, test_y), verbose=2, shuffle=False)




# 문제 loss 그라프와 val_loss 그라프를 출력하시오

print(history.history.keys())

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

# acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')

# acc_ax.plot(history.history['acc'], 'b', label='train acc')
# acc_ax.plot(history.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')

plt.show()



##선생님 답

plt.plot(history.history['loss'], 'y', label='train loss')
plt.plot(history.history['val_loss'], 'r', label='val loss')
plt.legend()
plt.show()




# test를 predict해서 RMSE값을 출력해 보시오.

# 신경망 : scale
# inverse_transform

yhat = model.predict(test_X)
test_X.shape
X.shape
# inverse_transform ( dim 이 다릅니다 )
# transform 할때,  bitcoin_market_info 은 6개 변수가 있는데,
# test_X는 변수가 하나 뿐입니다.

# 예측된 것은 한 개 , output close 입니다.

# 스케일 복원하는 과정

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

#원본과 같은 모양으로 생성
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]




# test_y도 스케일시켰죠. 그래서 복원해야 해요
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))



지금까지는 있는 데이터로 하였고, 아래는 실제 데이터로 예측을 한 다음.
그 예측된 값과, 실제 데이터의 차이를 백분율로 구하시오.



                            입력 데이터                                    :         실제 데이터
          [[11171.4 ,11256.0 ,11656.7 ,10470.3 ,9746200000 ,189398000000] :           11440.7
           [11259.4 ,11421.7 ,11785.7 ,11057.4 ,8873170000 ,192163000000] :           11171.4
           [11359.4 ,10903.4 ,11501.4 ,10639.8 ,9940990000 ,183419000000] :           11259.4
           [10868.40,10944.50 ,11377.60,10129.70, 9660610000 ,184087000000] :         11359.4
           [10931.40,11633.10,11966.40,10240.20,10537400000 ,195645000000] :          10868.4
           [11600.10,12889.20,12895.90,11288.20,9935180000,216740000000]] :           10931.4

from numpy import array
X = array([[11171.4 ,11256.0 ,11656.7 ,10470.3 ,9746200000 ,189398000000], [11259.4 ,11421.7 ,11785.7 ,11057.4 ,8873170000 ,192163000000],
        [11359.4 ,10903.4 ,11501.4 ,10639.8 ,9940990000 ,183419000000],[10868.40,10944.50 ,11377.60,10129.70, 9660610000 ,184087000000],
        [10931.40,11633.10,11966.40,10240.20,10537400000 ,195645000000],[11600.10,12889.20,12895.90,11288.20,9935180000,216740000000]])

y = array([[11440.7, 11171.4, 11259.4, 11359.4, 10868.4, 10931.4]])

input_data


X = X.reshape((X.shape[0], 1, X.shape[1]))
X.shape
y = y.reshape((y.shape[0], 1, y.shape[1]))
y.shape

yhat = model.predict(X)
print(yhat)
print(y)
