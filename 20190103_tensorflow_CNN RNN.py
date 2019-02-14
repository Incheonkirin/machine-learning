
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

tf.test.is_gpu_available()


import os
os.chdir("C://Users/Hyungi/Desktop/workplace")
os.getcwd()


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256

# convolution은 다단계에 걸쳐서 학습합니다.
# 오늘은 CNN으로 해보는 것 입니다.
def init_weights(shape): #랜덤 초기화 함수
    return tf.Variable(tf.random_normal(shape, stddev=0.01)) #다단계에 걸쳐 공간을 만들어야 하기에, 이 구문을 계속 써줘야 합니다. 때문에 함수화 시켰습니다.

#CNN model을 구현
# Filter weight vectors: w, w2, w3, w4, w_0
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    #  X 몇차원이에요? 배치사이즈 , 28 , 28, 1 로 들어오겠죠?
    # 그 다음에 w값이 뭐에요? 필터죠. 필터 사이즈가 3 by 3 , 1  ,필터에 32장.   결국 1장이 32장이 나온다는 이야기입니다.
    # output 사이즈는? 입력사이즈와 같아야 하죠. padding 이 벌어졌다는 이야기입니다.
    # conv2d 만든다음에 relu값으로, 값을 제한 하고 있지요?

    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)  # activation function
                        strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=True))

    # 2, 2로 Pooling하면, 4칸이 1칸이 됩니다.
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)                                       # dropout이란    # 과적합을 방지하기위해서 100%연산을 진행하지않고, 줄여주는 것입니다. #감소시킴으로써 과적합을 방지합니다. #랜덤으로 줄입니다. 그래서 dropout을 중간중간 넣습니다. 몇%를 남길 것인지.

    #relu maxpool dropout을 계속해서 반복하는 것입니다.

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=True))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=True))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o) #실제 나가는 값 :  128, 10 # 배치사이즈 x 10으로 나가야 맞죠? # one-hot encoding
    #batch size = 128
    return pyx

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1) # reshape 시키는 이유는 무엇인가요? CNN은 입력데이터 포멧이 결정되어있습니다. 그 결정과 맞춰줘야하는데, input format이 이렇게 되어있습니다. [batch, in_height, in_width, inchannels] 지금은 28, 28 로 들어있습니다.
teX = teX.reshape(-1, 28, 28, 1) # -1은 배치사이즈가 들어옵니다. 그 다음에 데이터를 받으려면 (아래)
X = tf.placeholder("float", [None, 28, 28, 1]) # 변수 데이터
Y = tf.placeholder("float", [None, 10]) # 종속 변수 데이터
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64]) #앞단에서 32장 들어와서 64장으로 나갑니다.
w3 = init_weights([3, 3, 64, 128]) #앞단에서 64장 들어와서 128장으로 나갑니다.
w4 = init_weights([128 * 4 * 4, 625])  # reshape이 벌어지고있죠. output에 맞춰서, 전체 사이즈가 됩니다.
#                    ([2048,  625])
w_o = init_weights([625, 10]) #10이 뭐에요? output이죠? 여기에 맞춰서 위에 w4가 나옵니다.




p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost) #매개변수가 2개이죠? Adam은 디폴트로 하나썼죠? 그것은 learning rate이고, 여기서는 decay:감쇠입니다.    RMS는 뭐하는 거라고 했죠? Adagrad => 0으로 수렴한다고 했죠. 감쇠효과는 너무 줄어드는 것을 방지하기 위해서 decay.
predict_op = tf.argmax(py_x, 1) #가장 큰 값을 예측


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                range(batch_size, len(trX) + 1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end],
                    Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})
        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0: test_size]
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                sess.run(predict_op, feed_dict={X: teX[test_indices],
                Y: teY[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})))



import tensorflow as tf

s = tf.constant("Hello Tensor")
sess = tf.Session()

print(sess.run(s))



#
import os
FLOWERS_PATH = os.path.join("datasets", "flowers")
flowers_root_path = os.path.join(FLOWERS_PATH, "flower_photos")


flower_classes = sorted([dirname for dirname in os.listdir(
        flowers_root_path)
                if os.path.isdir(
                        os.path.join(flowers_root_path, dirname))])


flower_classes

from collections import defaultdict
image_paths = defaultdict(list)
for flower_class in flower_classes:
    image_dir = os.path.join(flowers_root_path, flower_class)
    for filepath in os.listdir(image_dir):
        if filepath.endswith(".jpg"):
            image_paths[flower_class].append(os.path.join(image_dir, filepath))

width = 299
height = 299
channels = 3




import matplotlib.pyplot as plt
import matplotlib.image as mpimg

n_examples_per_class = 2

for flower_class in flower_classes:
    print("Class: ", flower_class)
    plt.figure(figsize=(10, 5))
    for index, example_image_path in enumerate(
            image_paths[flower_class][:n_examples_per_class]):
        example_image = mpimg.imread(example_image_path)[:, :, :channels]
        plt.subplot(100 + n_examples_per_class * 10 + index + 1)
        plt.title("{} x{}". format(example_image.shape[1],
                example_image.shape[0]))
        plt.imshow(example_image)
        plt.axis("off")
    plt.show()



# 훈련용 이미지 : 사이즈 고려, 같은 위치만 자르면 없어지는 부분, 랜덤으로 뒤집고
from scipy.misc import imresize
import numpy as np
def prepare_image(image, target_width= 299, target_height= 299, max_zoom = 0.2):
    height = image.shape[0] # 행은 이미지로 보면 높이
    width = image.shape[1]
    image_ratio = width/ height #원본 이미지
    target_image_ratio = target_width / target_height #원하는 이미지
    crop_vertically = image_ratio < target_image_ratio # 세로로 길쭉하다면,
    crop_width = width if crop_vertically else int (height * target_image_ratio)
    crop_height = int(width/ target_image_ratio) if crop_vertically else height
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int (crop_width/ resize_factor)
    crop_height = int(crop_height/ resize_factor)
    x0 = np.random.randint(0, width - crop_width) # 시작점 : 랜덤
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    image = image[y0: y1, x0: x1]
    if np.random.rand() < 0.5:
        image = np.fliplr(image)

    image = imresize (image, (target_width, target_height))
    return image.astype(np.float32) / 255

plt.figure(figsize = (6, 8))
plt.imshow(example_image)
plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
plt.axis("off")
plt.show()




import numpy as np

prepared_image = prepare_image(example_image)

plt.figure(figsize=(8, 8))
plt.imshow(prepared_image)
plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
plt.axis("off")
plt.show()




rows, cols = 2, 3
plt.figure(figsize=(14, 8))
for row in range(rows):
    for col in range(cols):
        prepared_image = prepare_image(example_image)
        plt.subplot(rows, cols, row*cols+ col+1)
        plt.title("{}x{}".format(prepared_image.shape[1],prepared_image.shape[0]))
        plt.imshow(prepared_image)
        plt.axis("off")

plt.show()



# tensorflow에서 이미지 처리하는 함수

import tensorflow as tf
def prepare_image_with_tensorflow(image, target_width =299,
                target_height= 299, max_zoom = 0.2):
    image_shape = tf.cast(tf.shape(image), tf.float32)
    height = image_shape[0]
    width = image_shape[1]
    image_ratio = width/ height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = tf.cond(crop_vertically, #tf.cond 3항 연산자를
            lambda: width, #참 이면
            lambda : height * target_image_ratio) # 거짓이면
    crop_height = tf.cond(crop_vertically,
            lambda: width /target_image_ratio, #참이면
            lambda: height) #거짓이면
    resize_factor = tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + max_zoom)
    crop_width = tf.cast ( crop_width / resize_factor, tf.int32)
    crop_height = tf.cast(crop_height / resize_factor, tf.int32)

    box_size = tf.stack([crop_height, crop_width, 3])
    image = tf.random_crop(image, box_size) #이미지 절단
    image = tf.image.random_flip_left_right(image) #좌우 반전
    image_batch = tf.expand_dims(image, 0) #차원 확장
    # 선형으로 사이즈
    image_batch = tf.image.resize_bilinear(image_batch,
            [target_height, target_width])
    image = image_batch[0] / 255
    return image


input_image = tf.placeholder(tf.uint8, shape=[None, None, 3])
prepared_image_op = prepare_image_with_tensorflow(input_image)


with tf.Session():
    prepared_image = prepared_image_op.eval(feed_dict = {input_image : example_image})



plt.figure(figsize=(6, 6))
plt.imshow(prepared_image)
plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
plt.axis("off")
plt.show()




# CNN 모델 => deep Learning 가능한 모델


from tensorflow.contrib.slim.nets import inception # 가장 정밀하게 분류해주는 모델
import tensorflow.contrib.slim as slim
#                                 slim은 케라스 할 때 다룹니다. slim == 모델을 간편하고 쉽게 사용할 수 있도록 만들어 놓음.
from tensorflow.python.framework import ops
ops.reset_default_graph()


X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
training = tf.placeholder_with_default(False, shape=[])

#arg_scope 그래프에서 변수를 공유하기 위하여 지정
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(
            X, num_classes=1001, is_training=training)
inception_saver = tf.train.Saver()
# end_points["PreLogits"] inception_v3에서 리턴되어지는 가중치
prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])



n_outputs = len(flower_classes)

# CNN 역할 = 특성추출 => Fullyconnected망으로 보내야 함.
# Fullyconnected망에서 하는 일 뭐에요? 가중치 tensor (몇 by 몇으로 할지)잡고. 출력되어진 tensor 리턴
# 결국.. input으로 잡을 만큼,

# CNN RNN GAN의 개념만 완벽히 익히고 가면 됩니다.
# 내부에서 무슨일이 일어나고 있느냐 만 알고 있으면 됩니다.

# dense가 뭐에요?  뭐가 없어요 여기에? 가중치 곱해주고, 리턴하고 하는 형식(이 없지요?)을 줄여준 것이다. 'dense가'

with tf.name_scope("new_output_layer"): #왜 잡아주나요? 텐서보드에서 압축적으로 표현하기 위해서.

    flower_logits = tf.layers.dense(prelogits, n_outputs, name="flower_logits")
    Y_proba = tf.nn.softmax(flower_logits, name="Y_proba")



y = tf.placeholder(tf.int32, shape=[None])
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flower_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()

    flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="flower_logits")
    training_op = optimizer.minimize(loss, var_list=flower_vars)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(flower_logits, y, 1)#원래는 argmax로 했었죠? in_top_k는 순위를 찾을 수 있게끔 해주는 것입니다.
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

[v.name for v in flower_vars]
flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}
flower_class_ids

flower_paths_and_classes = []
for flower_class, paths in image_paths.items():
    for path in paths:
        flower_paths_and_classes.append((path, flower_class_ids[flower_class]))


test_ratio = 0.2
train_size = int(len(flower_paths_and_classes) * (1 - test_ratio))

np.random.shuffle(flower_paths_and_classes)

flower_paths_and_classes_train = flower_paths_and_classes[:train_size]
flower_paths_and_classes_test = flower_paths_and_classes[train_size:]

flower_paths_and_classes_train[:3]

from random import sample

def prepare_batch(flower_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(flower_paths_and_classes, batch_size)
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = np.stack(prepared_images)
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch
X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size=4)
X_batch.shape

X_test, y_test = prepare_batch(flower_paths_and_classes_test, batch_size=len(flower_paths_and_classes_test))
X_test.shape


INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")#inception은 어느 정도 훈련이 된 기반이 필요로합니다. 그 위에 훈련을 하기위해 inception_v3.ckpt를 사용합니다.
#ckpt == check point : 레이어가 깊어
#                      기반을 다져서, 가중치 학습된 것.
n_epochs = 10
batch_size = 50
n_iterations_per_epoch = len(flower_paths_and_classes_train) // batch_size

with tf.Session() as sess:
    init.run()
    # model => 변수들을 constant(정해져있죠), variables(얘만 저장하면 돼..), placeholder(외부주입변수니까 날라가는거고.)
    # 변수들이 다 올라왔다는 말은, 가중치를 로딩한다는 말 => random
    inception_saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)

    for epoch in range(n_epochs):
        print("Epoch", epoch, end="")
        for iteration in range(n_iterations_per_epoch):
            print(".", end="")
            X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("  Train accuracy:", acc_train)

        save_path = saver.save(sess, "./my_flowers_model")

    print("Computing final accuracy on the test set (this will take a while)...")
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Test accuracy:", acc_test)
#이런 함수이름들 보기만 하면 하나도 안외워져요. 계속 쳐봐야 해요. 쳐봐야 머릿속에 남아요.




#꽃을 5개 중 하나로 분류



# 그래서 나중에 이미지처리하고싶다. 분류기만들고 싶다.
# 얼굴 찍어서 웃고있는지 울고있는지 분류 하는 것을 만들었음.




#
# 이번에는 RNN을 해보겠습니다.
# Recurent neural network : 2개 이상(= 시간차)
# RNN 특성이 뭐에요? FFNN는 앞단과 '만',  관계를 맺습니다.( layer간에만 영향을 미칩니다.) 단, RNN는, 수직 수평적 관계를 맺습니다(레이어를 전진하면서 학습).
# 그런 특성적 관계를 맺을 수 있게끔 하는 것은, 'cell'
# 'cell'은 입력을 가지고, (입력된 것에)가중치를 곱하고, activation function도 달려있습니다. / 출력이 2개나 있죠. 하나는 전진해서 나가는것, 다른 하나는 옆으로 나가는 것

#결국 'cell '다 합치면 FFNN임,
# 이걸 관계를 가지고 시간차를 가지면 RNN이 되는 것.

import numpy as np
import tensorflow as tf
tf.reset_default_graph()
values = tf.constant(np.array([
        [   [1],
            [2],
            [3]
        ],
        [
            [2],
            [3],
            [4]]]), dtype=tf.float32)

lstm_cell = tf.contrib.rnn.LSTMCell(100) # LSTMCell 나온 이후로, 바닐라~ 그건 안씁니다. # 100개는 무슨 의미가 있나요? hidden status를 저장하는 공간입니다.
# 그래서 static_rnn이 있고, dynamic_rnn이  있어요
#   정해진 사이즈-패딩
outputs , state = tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, inputs = values)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_run, state_run = sess.run([ outputs, state]) #2, 3, 100 =>#원래 2, 3, 1인데 #1이 100으로 되었음 *가중치*!
    print(output_run.shape)
    print(output_run)


print(output_run.shape)
np.all(output_run[:, -1] == state_run.h) #마지막으로 들어 간 것이, state_run.h와 같다.      true라고 나오죠?

# 흐름 => 오른쪽으로 진행
# bidirectional_dynamic_rnn

tf.reset_default_graph()


values = tf.constant(np.array([
        [[1], [2], [3]],
        [[2], [3], [4]]]), dtype = tf.float32)


lstm_cell_fw = tf.contrib.rnn.LSTMCell(100) # tf.contrib.rnn.GRUCell(100) ** GRU셀이 뭐냐. 이런걸 신경써야 한다는 말이에요 . forget cell하고 input cell하고 합쳐 놓은 것.
lstm_cell_bw = tf.contrib.rnn.LSTMCell(105)

(output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_cell_fw,
        cell_bw=lstm_cell_bw,
        inputs=values,
        dtype=tf.float32)
print(output_fw.shape)
print(output_fw)



##
# 다음은 multi RNN입니다
#
# RNN을 조금 더 정확히 하기 위해서, 만든 것?
# bidirectional_dynamic_rnn이죠
#
# multi layer rnn을 이야기하는 건데..


lstm_cell = lambda:tf.contrib.rnn.LSTMCell(100)
multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(3)]) #수직으로 셀을 3개 쌓아.

# 수평으로 ..
outputs, state = tf.nn.dynamic_rnn(cell=multi_cell, dtype=tf.float32, inputs=values)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_run, state_run = sess.run([outputs, state])
    print("출력된 특성", output_run.shape)




###
t_min, t_max = 0, 30
resolution = 0.5

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps) :
    t0 = np.random.rand(batch_size, 1) * ( t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arrange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)


import matplotlib.pylab as plt
import numpy as np

t = np.linspace(t_min, t_max, int((t_max - t_min)/ resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps+ 1)
plt.figure(figsize=(11, 4))
plt.subplot(121)
plt.title("All generated data", fontsize=14)
plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label= "trained part")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")




t = np.linspace(t_min, t_max, int((t_max - t_min)/ resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps+ 1)



import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_iterations = 1500
batch_size = 50

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})

    saver.save(sess, "./my_time_series_model")

y_pred


plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

plt.show()


with tf.Session() as sess:
    saver.restore(sess, "./my_time_series_model")

    sequence = [0.] * n_steps
    for iteration in range(300):
        X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence.append(y_pred[0, -1, 0])

plt.figure(figsize=(8,4))
plt.plot(np.arange(len(sequence)), sequence, "b-")
plt.plot(t[:n_steps], sequence[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

with tf.Session() as sess:
    saver.restore(sess, "./my_time_series_model")

    sequence1 = [0. for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence1[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence1.append(y_pred[0, -1, 0])

    sequence2 = [time_series(i * resolution + t_min + (t_max-t_min/3)) for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence2.append(y_pred[0, -1, 0])

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.plot(t, sequence1, "b-")
plt.plot(t[:n_steps], sequence1[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.plot(t, sequence2, "b-")
plt.plot(t[:n_steps], sequence2[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
save_fig("creative_sequence_plot")
plt.show()
