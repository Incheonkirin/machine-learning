import tensorflow as tf
import keras
from keras.models import Sequential # functional 방식도 있다. graph, : Model
from keras.layers import Dense # 입력데이터와 가중치를 완전하게 해주는 Dense, 결과 값 activation(dot(input, kernel) + bias)

tensor - conda
keras - pip

#                                                                                                       가중치
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

X, Y = make_moons(noise= 0.2, random_state = 0, n_samples = 1000)
X = scale(X) #정규화
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= .5) #데이터 분리

fig, ax = plt.subplots()
ax.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')
ax.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')
ax.legend()
ax.set(xlabel='X', ylabel='Y', title='binary classification');

model = Sequential()
#model Define# activation(dot(input, kernel) + bias)
model.add(Dense(32, input_dim = 2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 0~ 1 사이의 확률값 분류

model.compile(optimizer='AdaDelta', # Adagrad의 문제점 : 시간의 문제점, 처음 learning rate크게, 나중에 작게. 그러나, 기울기가 0으로 가는 문제점 있다. 이것을 수정한 게, AdaDelta.
        loss='binary_crossentropy', # mean_squared_error L2, mean_absolute_error L1 데이터가 나왔으면 예측값과 실제값 재서, 평균을 내어라.
#   crossentropy는 두개로 나누었을 때 복잡도 를 말합니다. 복잡도에 의해서 loss_function을 정의하고 있습니다.
        metrics=['accuracy'])


# 콜백함수로 등록
# 그래프 출력
# TensorBoard
# LeaerningRateScheduler
# RemoteMonitor
# Earlystopping (비용함수가 최저점 인 곳에서 끊어라.)
from tensorflow.python.client import device_lib
device_lib.list_local_devices()



# 그래프 출력.                             출력되는 디렉터리              100번마다 1번 저장하고,
tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph/model_1/',
        histogram_freq=100, write_graph=True, write_images=False)


#콜백함수를 모델에 등록
tb_callback.set_model(model)
#전체데이터 / batch_size 만큼 돌아가는데, epochs 만큼 200번 실행하여라.
# epoch는 똑같은 데이터를 200번 돌리라는 의미
#verbose 는 상태를 계속 알려줌.
#
hist = model.fit(X_train, Y_train, batch_size=32, epochs=200, verbose=0,
        validation_data = (X_test, Y_test), callbacks=[tb_callback])

score = model.evaluate(X_test, Y_test, verbose=0)
#
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(hist.history['loss'])
print(hist.history['acc'])



#사각형 데이터 예측하는 함수, 무엇을 썼었죠? meshgrid를 썼었습니다. mgrid는 행과 열이 변경 된 것입니다.

grid = np.mgrid[-3:3:100j, -3:3:100j]
grid_2d = grid.reshape(2, -1).T
X, Y = grid
prediction_probs = model.predict_proba(grid_2d, batch_size=32, verbose=0)
# plot results
fig, ax = plt.subplots(figsize=(10, 6))
# 좌표값
contour = ax.contourf(X, Y, prediction_probs.reshape(100, 100))
ax.scatter(X_test[Y_test==0, 0], X_test[Y_test==0, 1])
ax.scatter(X_test[Y_test==1, 0], X_test[Y_test==1, 1], color='r')
cbar = plt.colorbar(contour, ax=ax)



from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

np.random.seed(3)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


X_val = X_train[50000:]
Y_val = Y_train[50000:]
X_train = X_train[:50000]
Y_train = Y_train[:50000]

# 255 로 나누어준 이유는 ?
# 이미지 정규화 하고 있음. 이미지 0 ~ 255 컬러값으로 표현되거나, 부동소수점으로 표시할 수 있다.
# 0에서 1사이로 바꾸기 위해서.
X_train = X_train.reshape(50000, 784).astype('float32') / 255.0
X_val = X_val.reshape(10000, 784).astype('float32')/ 255.0
X_test = X_test.reshape(10000, 784).astype('float32')/ 255.0
train_rand_idxs = np.random.choice(50000, 700)
val_rand_idxs = np.random.choice(10000, 300)

X_train = X_train[train_rand_idxs]
Y_train = Y_train[train_rand_idxs]
X_val = X_val[val_rand_idxs]
Y_val = Y_val[val_rand_idxs]

#분류 : 숫자
#CNN망을 사용하지 않고, FFNN을 사용하였다.


model = Sequential()
# unit 256, 2
model.add(Dense(units=2, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax')) # 분류 | 범주형

#categorical 이 나온 이유는?
#범주형데이터가 아닌, 숫자형태 로 들어왔기 때문에.  범주형이 아니고 숫자이면 회귀가 됨.
model.compile (loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#                                                    sgd = stachastic gradient descent 확률적 하강 경사법
print("작업중")
hist = model.fit(X_train, Y_train, epochs=1000, batch_size=10, validation_data=(X_val, Y_val), verbose=0)



import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')


loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

keras.utils.print_summary(model)
model.summary()





4) fit
5) evaluate


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train.shape
x_test.shape
num_features = np.prod(x_train.shape[1:])
num_features

model = Sequential()
model.add(Dense(1, input_dim= num_features, activation='linear'))
model.summary()
model.compile(optimizer='rmsprop', loss='mse', mertrics=['mae'])
model.fit(x_train, y_train, batch_size=1, epochs=10, verbose=1)
mse, mae = model.evaluate(x_test, y_test, verbose=False)

rmse = np.sqrt(mse)


mse, rmse, mae
model.predict(x_test[:3, :])
y_test[:3]



# wine data를 이용한 분류기

import pandas as pd
white = pd.read_csv("wine-white.csv", sep=",")
red = pd.read_csv("wine-red.csv", sep=",")


print(white.shape)
print(white.info())
print(red.info())

import seaborn as sns

red['type'] = 1
white[['type'] = 0

wines = red.append(white, ignore_index=True)
corr = wines.corr()
sns.heatmap(corr,
    xticklabels = corr.columns.values,
    yticklabels=corr.columns.values)

plt.show()



import numpy as np
from sklearn.model_selection import train_test_split
X = wines.iloc[:, 0:11]
y= np.ravel(wines.type)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
        random_state=42)

from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense


#FFNN분류기
# 1 x 11, 11 x 12 + bias(12) = 132 + 12 => 결과 1x 12
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(11, ))) # 12 --- 144 (가중치 사이즈)
# 1 x 12 가 입력이 되고, 12 x 8 + 8 => 결과 1 x 8
model.add(Dense(8, activation='relu'))
# 1 x 8 입력. 가중치 = 8 x 1 + 1 => 결과    1x 1             # 8  --- 104
model.add(Dense(1, activation='sigmoid'))                   # 1 ---- 9
model.summary()
model.compile(loss='binary_crossentropy',
        optimizer='adam', # momentum + adadelta +
        metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)
score = model.evaluate(X_test, y_test, verbose=1)
print(score)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred.round())


# pima - indians - diabetes.data 분류기 작성
rrom keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt('pima-indians-diabetes.data', delimiter=',')

X = dataset[:, 0: 8]
Y = dataset[:, 8]


# 1
model = Sequential()
model.add(Dense(12, input_dim= 8, kernel_initializer= ? , activation= ?))
model.add(Dense(8, kernel_initializer= ? , activation = ?))
model.add(Dense(1, kernel_initializer= ? , activation = ' sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = ? , metrics=['accuracy'])
history = model.fit(X, Y, validation_split = 0.33, epochs = 150, batch_size=10, verbos=0)

# 그라프로 출력하시오
print(history.history.keys()) # acc, loss



# kerasClassifier and KerasRegressor

import numpy
import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
X = dataset[:, 0:13]
Y = dataset[:, 13]


def baseline_model():
model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal')

model.compile(loss='mean_squared_error', optimizer='adam')
return model



seed = 7
numpy.random.seed(seed)
estimator = KerasRegressor(build_fn = baseline_model, nb_epoch=100, batch_size=5, verbose=0)

kfold = kFold(n_splits=10, random_state= seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f ( %.2f) MSE" % (results.mean(), results.std()))

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state = seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std() ))
pipeline.fit(X, Y)



##

import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import kerasClassifier
from keras.contraints import maxnorm

def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=8, kernel_initializer='uniform', activation = 'linear',
            kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    #Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


seed = 7
numpy.random.seed(seed)
dataset = numpy.loadtxt("./pima.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

model = kerasClassifier(build_fn = create_model, epochs=100, batch_size=10, verbose=0)
neurons=[1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator= model, param_gird = param_grid, n_jobs= -1)
grid_result = grid.fit(X, Y)



#

1) 최적의 파라미터 내용을 확인하시오
best_params_, best_score_

2)
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]




#숙제
# 대기 벤젠 수준 예측( 기온/상대습도, 일산화탄소, 일산화질소 => 벤젠의 농도 예측)
    # 데이터 : 5개의 금속산화물 센서 데이터로부터 벤젠의 농도 예측
    # 문제1 신경망은 scale에 민감함. scaling을 실시하시오
    # 문제2 train/test 데이터를 분리하시오 (7:3)
    # 문제3 회귀망으로 networks를 정의하시오
    # 문제4 테스트 데이터를 예측하고 평가하시오
    # 문제5 optimizer parameter tuning 실시하시오 (adam, sgd, rmsprop)

aqdf = pd.read_csv("./data/air.csv", sep=";" , decimal=",", header=0)
del aqdf["Date"]
del aqdf["Time"]
del dqdf["Unnamed: 15"]
del aqdf["Unnamed: 16"]
