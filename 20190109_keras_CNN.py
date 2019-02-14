import os

print (os.getcwd())
os.chdir("C://Users/Hyungi/Desktop/workplace")

from IPython.display import Image, SVG
import matplotlib.pyplot as plt
import pydot
import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils.vis_utils import model_to_dot

(X_train, y_train), (X_test, y_test) = mnist.load_data()
first_image = X_train[0, :, :]
plt.imshow(first_image, cmap = plt.cm.Greys)
num_classes= len(np.unique(y_train))
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')
input_shape= (28, 28, 1)

max_value = X_train.max()
X_train /= max_value
X_test /= max_value
(y_train, y_test)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


model = Sequential()
#kernel = filter
#나가는 필터 채널 수

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
        activation='relu',
        input_shape = input_shape)) # None(batch_size이니까, 128개)  input_shape = 28(-3+1), 28(-3+1), 1
# 26 x 26 x 32
model.add(MaxPooling2D(pool_size=(2, 2)))#가로 사이즈 2개를 1개로 줄이고, 세로 사이즈 2개를 1개로 줄입니다. # 13x 13x 32
# pooling이 4개가 되니까 줄어들죠, 상하좌우로.

#다시 필터처리하니까..
model.add(Conv2D(64, (3, 3), activation='relu')) # 11 x 11 x 64
model.add(Dropout(0.25)) #안에 계산회로를 없앰, 차수는 그대로.
# 또 pooling을 적용, 줄어듭니다.
model.add(MaxPooling2D(pool_size=(2, 2))) # 5 x 5 x 64

# Flatten이니까, 위를 모두 곱하면
model.add(Flatten()) # 1600
# 20480 + 128 bias

# Dense망이니까 완전망으로 들어가지요.
model.add(Dense(128, activation='relu')) # 1600 x 128의 가중치가 있어야 하지요. (입력 1600, 출력 128)
model.add(Dropout(0.5))# 절반만 계산하라
# 128 x 10 = 1280 + 10 => 1290

model.add(Dense(num_classes, activation='softmax')) # 계산해야하니 softmax , 0에서 1사이 확률값으로 분류.
# num class가 몇개 인가요? 10개이지요. 10개로 나가야 하잖아.
#들어오는 것은 몇 차 에요? 128 이 들어와야 하지요.

# 1600 x 128은 204800 인데, 왜 204928 이 dense_1인가요?
### 20480 + 128( bias)를 더해 줘야 하죠.

model.summary()



 from keras.layers import LSTM

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1)

model.evaluate(X_test, y_test)

first_test_image = X_test[0, :]
plt.imshow(first_test_image.reshape(28, 28), cmap=plt.cm.Greys)

second_test_image= X_test[1, :]
plt.imshow(second_test_image.reshape(28, 28), cmap=plt.cm.Greys)





# cifar10을 이용한 CNN분류
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', 'train samples')
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
plt.imshow(X_train[5])
plt.grid(False)
plt.show()


model = Sequential()
model.add(Conv2D(32, (3, 3), padding = 'same',
        input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer=OPTIM,
        metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE,
        epochs = NB_EPOCH, validation_split = VALIDATION_SPLIT,
        verbose=VERBOSE)



print('Testing...')
score = model.evaluate(X_test, Y_test,
        batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest loss: ", score[0])
print('Test accuracy: ', score[1])
#모델 저장과 가중치 저장
model_json = model.to_json()
open('cifar10_architecture.json', 'w').write(model_json) #모델
model.save_weights('cifar10_weights.h5', overwrite=True) #가중치

plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#모델을 로딩하고 예측해 보시오

from keras.models import model_from_json
from keras.optimizers import SGD
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)


#문제 cat truck 이미지를 다운받아서 예측해보시오

import numpy as np
from imageio import imread
from skimage.transform import resize
img_names =['cat2.jpg', 'truck.jpg']
imgs = [np.transpose(resize(imread(img_name), (32, 32)), (1, 0, 2)).astype('float32')
        for img_name in img_names]

imgs = np.array(imgs) / 255
model.compile (loss='categorical_crossentropy',
        optimizer=OPTIM, metrics=['accuracy'])
predictions = model.predict_classes(imgs)
print(predictions)



#ImageDataGenerator
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
datagen = ImageDataGenerator(rotation_range=90)
# datagen = ImageDataGenerator(horizontal_flip= True, vertical_flip= True)
datagen.fit(X_train)

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i].reshape(28, 28), cmap = pyplot.get_cmap('gray'))
    pyplot.show()
    break
import os
os.makedirs('images')



from IPython.display import display
from PIL import Image


for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='images',
        save_prefix='aug', save_format='png'):
        for i in range(0, 9):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
        pyplot.show()
        break


for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='images', save_prefix='aug', save_format='png'):
	# create a grid of 3x3 images
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	break




import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
# backend를 내포. backend를 컨트롤할때, k로 하면 됩니다.
from keras import backend as k


seed = 9
np.random.seed(seed= seed)
tf.set_random_seed(seed = seed)


nb_classes = 2
based_model_last_block_layer_number = 126
img_width, img_height = 299, 299
batch_size = 32
nb_epoch = 50
learn_rate = 1e-4
momentum = .9
transformation_ratio = .05

def train(train_data_dir, validation_data_dir, model_path):
    # applications : 미리 생성 모델.  Xception 같은 것들을 Keras에서는 Application이라고 합니다.
    # transition learning : 전이학습 : 기학습 모델(기반이 되는 모델 ) + 마지막 레이어 학습.
    base_model = Xception(input_shape = (img_width, img_height, 3), weights= 'imagenet',
            include_top = False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation = 'softmax')(x)

    model = Model(base_model.input, predictions)
    print(model.summary())
    train_datagen = ImageDataGenerator(rescale=1. / 255,
            rotation_range=transformation_ratio,
            shear_range = transformation_ratio,
            zoom_range = transformation_ratio,
            cval = transformation_ratio,
            horizontal_flip = True,
            vertical_flip = True)

    validation_datagen = ImageDataGenerator(rescale= 1. / 255)



    os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    # save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
    # save_prefix='aug',
    # save_format='jpeg')
    # use the above 3 commented lines if you want to save and look at how the data augmentations look like

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')



    model.compile(optimizer='nadam',
            loss = 'categorical_crossentropy',
            metrics=['accuracy'])
    top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')
    callbacks_list = [
            ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only= True),
            EarlyStopping(monitor='val_acc', patience=5, verbose=0)

    ]
    model.fit_generator(train_generator,
            samples_per_epoch = train_generator.nb_sample,
            nb_epoch= nb_epoch / 5,
            validation_data = validation_generator,
            nb_val_samples = validation_generator.nb_sample,
            callbacks = callbacks_list)

    model.load_weights(top_weights_path)
    for layer in model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False # Xception에 훈련에서 제외되어야 할 레이어
    for layer in model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True #

    model.compile( optimizer = 'nadam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])

    final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')
    callback_list = [
            ModelCheckpoint(final_weights_path, monitor='val_acc', verbose = 1, save_best_only = True),
            EarlyStopping(monitor='val_loss', patience= 5, verbose= 0)

    ]
    model.fit_generator(train_generator,
            samples_per_epoch = 12500/ 32,
            nb_epoch=nb_epoch,
            validation_data= validation_generator,
            nb_val_samples = (16*63+8)/ 32,
            callbacks= callbacks_list)
    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
        json_file.write(model_json)



data_dir = os.path.abspath("./data/")
train_dir = os.path.join(os.path.abspath(data_dir), 'train')
validation_dir = os.path.join(os.path.abspath(data_dir), 'valid')
model_dir = os.path.abspath("./data/model")

os.makedirs(os.path.join(os.path.abspath(data_dir), 'preview'), exist_ok = True)
os.makedirs(model_dir, exist_ok = True)


train(train_dir, validation_dir, model_dir)



k.clear_session()

## image generator  사용하는 이유, model 저장 하는 이유



import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# hyper parameters for model
nb_classes = 2  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 299, 299  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation


def train(train_data_dir, validation_data_dir, model_path):
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

    # Top Model Block
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)
    print(model.summary())

    # # let's visualize layer names and layer indices to see how many layers/blocks to re-train
    # # uncomment when choosing based_model_last_block_layer
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=transformation_ratio, #회전
                                       shear_range=transformation_ratio, # 절단
                                       zoom_range=transformation_ratio, #줌
                                       cval=transformation_ratio,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    # save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
    # save_prefix='aug',
    # save_format='jpeg')
    # use the above 3 commented lines if you want to save and look at how the data augmentations look like

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc

    top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')
    callbacks_list = [
        ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=5, verbose=0)
    ]

    # Train Simple CNN
    model.fit_generator(train_generator,
                        samples_per_epoch=12500/32,
                        nb_epoch=nb_epoch / 5,
                        validation_data=validation_generator,
                        nb_val_samples=(16*63+8)/32,
                        callbacks=callbacks_list)


    print("\nStarting to Fine Tune Model\n")


    model.load_weights(top_weights_path)

    for layer in model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False #Xception의 훈련에서 제외되어야 할 layer

    for layer in model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True


    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')
    callbacks_list = [
        ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    ]


    model.fit_generator(train_generator,
                        samples_per_epoch=12500/32,
                        nb_epoch=nb_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=(16*63+8)/32,
                        callbacks=callbacks_list)



    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
        json_file.write(model_json)


data_dir = os.path.abspath("./data/")
train_dir = os.path.join(os.path.abspath(data_dir), 'train')
# Inside, each class should have it's own folder
validation_dir = os.path.join(os.path.abspath(data_dir), 'valid')
# each class should have it's own folder
model_dir = os.path.abspath("./data/model")

os.makedirs(os.path.join(os.path.abspath(data_dir), 'preview'), exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

train(train_dir, validation_dir, model_dir)  # train model


k.clear_session()



# 숙제
# 문제: 실제 고양이/ 개 이미지(다양한 이미지) 를 로딩하여 예측하시오
# 개선 시킬 수 있는 방향 (parameter tuning)

-evaluate_generator
-predict_generator

score = model.evaluate_generator(validation_generator, (16 * 63 + 8) /32, workers= 4)
scores = model.predict_generator(validation_generator, (16 * 63 + 8)/ 32, workers= 4)

correct = 0
for i, n in enumerate(validation_generator.filenames):
    if n.startswith("cats") and scores[i][0] <= 0.5:
        correct += 1
    if n.startswith("dogs") and scores[i][0] > 0.5:
        correct += 1

print("Correct:", correct, " Total: ", len(validation_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])



# cifar10
# fashion_mnist.load_data() #데이터를 로딩하고 분류기를 만들어보시오
#모델을 저장하고 로딩하는 부분을 처리하시오

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras.utils.vis_utils import model_to_dot
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
print(train_X.shape, test_X.shape)


train_Y_one_hot = np_utils.to_categorical(train_Y)
test_Y_one_hot = np_utils.to_categorical(test_Y)
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

input_shape= (28, 28, 1)

train_X /= 255
test_X /= 255


from sklearn.model_selection import train_test_split
train_X, valid_X,  train_label, valid_label = train_test_split(train_X, train_Y_one_hot,
        test_size= 0.2, random_state= 13)


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
        activation='relu',
        input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Activation('softmax'))


model.summary()


from IPython.display import Image, SVG
import pydot


SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1)

model.evaluate(X_test, y_test)

first_test_image = X_test[0, :]
plt.imshow(first_test_image.reshape(28, 28), cmap=plt.cm.Greys)

second_test_image= X_test[1, :]
plt.imshow(second_test_image.reshape(28, 28), cmap=plt.cm.Greys)


history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE,
        epochs = NB_EPOCH, validation_split = VALIDATION_SPLIT,
        verbose=VERBOSE)

print('Testing...')
score = model.evaluate(X_test, Y_test,
        batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest loss: ", score[0])
print('Test accuracy: ', score[1])

model_json = model.to_json()
open('fashionmnist.json', 'w').write(model_json)
model.save_weights('fashionmnist_weights.h5', overwrite=True)

plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

from keras.models import model_from_json
from keras.optimizers import SGD
model_architecture = 'fashionmnist.json'
model_weights = 'fashionmnist_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
