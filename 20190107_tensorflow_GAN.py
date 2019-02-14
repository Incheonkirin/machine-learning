# GAN

import os
os.chdir("C://Users/Hyungi/Desktop/workplace")
os.getcwd()

import tensorflow as tf
import numpy.random as rnd
import numpy as np
#차원축소( deep-learning을 이용한) # autoencoder를 사용
rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2  - 0.5
data = np.empty((m, 3)) # 200 by 3의 데이터가 만들어집니다.
data[:, 0] = np.cos(angles) + np.sin(angles) /2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)
#원형으로 데이터 형태가 되도록, sin cos을 적당히 사용하였음.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # 3차원으로 찍을 때 사용.
ax = plt.axes(projection='3d')
ax.scatter3D(data[:, 0], data[:, 1], data[:, 2]) # x축, y축, z축
ax.view_init(azim=70, elev=50) # azim 은 카메라의 좌우각 을 이야기하고, elev은 상하각을 이야기함.


# 딥러닝을 통해서 차원축소를 해 보도록 하겠습니다.
import tensorflow as tf
tf.reset_default_graph()
n_inputs = 3
n_hidden = 2
n_outputs = n_inputs
learning_rate = 0.01
#autoencoder 구현

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# dense : 자동으로 가중치 공간을 만들고 결과는 입력차수의 행 수, n_hidden= 열 수.
# hidden , 100 by 2
# 100 by 3    =>    100 by 2
#       가중치 3 by 2로 자동으로 만들어진다.

#                  dense는 완전연결망
hidden = tf.layers.dense(X, n_hidden) # 완전연결망, hidden = 특성이 추출된.
# 2 by 3 의 가중치로 계산
outputs = tf.layers.dense(hidden, n_outputs) # hidden = 100 by 2 =>  (output) 100 by 3
# 입력데이터 나 출력데이터가 동일하도록 만들어라.

#계산된 데이터가 입력 데이터와 동일 해 질때까지 학습하라.
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # 입출력차수가 동일 # reduce_mean == 미니배치를 만들어서, 한번만 바꿔주는 것. st - GradientDescent : 시간 절약, 지역해극복
optimizer = tf.train.AdamOptimizer(learning_rate) # adagrad 요소와 optimizer요소를 결합한 AdamOptimizer
training_op = optimizer.minimize(reconstruction_loss)
init = tf.global_variables_initializer()
n_iterations = 1000
codings = hidden

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



with tf.Session() :
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: X_train}) # 100 x 3
    codings_val = codings.eval(feed_dict={X: X_test})  # 100 x 3
    # 100 x 2

fig = plt.figure(figsize=(4, 3))
plt.plot(codings_val[:, 0], codings_val[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.savefig("선형 AE")
plt.show()




## 비지도학습과 지도학습 => 결합학습
from tensorflow.examples.tutorials.mnist import input_data
import math
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

import sys
import tensorflow as tf
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = 50
n_outputs = 10


learning_rate = 0.01
l2_reg = 0.0005
activation = tf.nn.relu
#규제, 초기화
# scikits 규제를 사용 : 과적합을 방지하고 일반화하기 위해서 사용함.

regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
# 가중치 초기화 : 초기에 분산값이 일정해야 결과 좋아짐.
# random보다 variance_scaling_initializer사용하면 결과가 더 좋아진다.
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
y = tf.placeholder(tf.int32, shape=[None])

#layer 가 3개 존재함.
# 입력 (1 x 784) 784 x 300 => 특징 추출(== noise를 제거한다)
# deep layer는 특성추출.
weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2]) # 300 x 150
weights3_init = initializer([n_hidden2, n_hidden3]) # 150 x 50

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name='weights1')
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name='weights2')
weights3 = tf.Variable(weights3_init, dtype=tf.float32, name='weights3')

# 배치사이즈가 150이니까, 784로 들어오는 데이터가 150개로 들어왔다는 이야기에요.
# placeholder도 150 x 784로 들어올 수 있도록 준비되어있어야 한다는 것입니다.
#       hidden1 으로 나가는 것 차수 ==>  784 x 300
# hidden 2  ?
# y값은? int32, 150개
# hidden 3 ?

biases1 = tf.Variable(tf.zeros(n_hidden1), name='biases1')
biases2 = tf.Variable(tf.zeros(n_hidden2), name='biases2')
biases3 = tf.Variable(tf.zeros(n_hidden3), name='biases3')


hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
logits = tf.matmul(hidden2, weights3) + biases3





cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
# 복잡도

reg_loss = regularizer(weights1 ) + regularizer(weights2) + regularizer(weights3)
loss = cross_entropy + reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k (logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # 배치 데이터의 평균으로 backpropagation하는 것
init= tf.global_variables_initializer()
saver = tf.train.Saver() # 모델 저장, 모델은 뭐만 저장한다고 했어요? variables. 여긴 뭐가 있냐, 가중치가 있지요.
n_epochs = 4
batch_size = 150
n_labeled_instances = 20000

# 150 * 20000 * 4
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = n_labeled_instances
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            indices = rnd.permutation(n_labeled_instances)[:batch_size]
            X_batch, y_batch = mnist.train.images[indices], mnist.train.labels[indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict = {X: X_batch, y: y_batch})
        print("\r{}".format(epoch), "훈련정확도:", accuracy_val, end=" ")
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y:mnist.test.labels})
        saver.save(sess, "./auto.ckpt")
        print("테스트 정확도: ", accuracy_val)




#비지도학습

learning_rate = 0.01
num_steps = 30000
batch_size = 256
display_step = 1000
examples_to_show = 10
num_hidden_1 = 256
num_hidden_2 = 128
num_input = 784

X = tf.placeholder("float", [None, num_input]) # None = batch_size    256 x 784
# 특징추출
weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_input, #784 x 256
                num_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, # 256 x 128
                num_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, # 128 x 256
                num_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, # 256 x 784
                num_input]))
        }
print(weights['encoder_h1'])
print(weights['encoder_h2'])
print(weights['decoder_h1'])
print(weights['decoder_h2'])

biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),# 256
        'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),# 128
        'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),# 256
        'decoder_b2': tf.Variable(tf.random_normal([num_input]))# 784
        }
print(biases['encoder_b1'])
print(biases['encoder_b2'])
print(biases['decoder_b1'])
print(biases['decoder_b2'])


#이렇게 묶는 이유는, 재사용하기 때문에 그렇습니다.
# 가중치를 공유합니다.


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, #256 x 784 # 784 x 256
            weights['encoder_h1']), biases['encoder_b1'])) # ==> 256 x 256
    # print(weights['encoder_h1']) #현재 784, 256
    # print(biases['encoder_b1']) #현재 256,
    # print(layer_1) #현재  ?, 256
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, # 256 x 256 # 256 x 128
            weights['encoder_h2']), biases['encoder_b2'])) # => 256 x 128
    # print(weights['encoder_h2'])
    # print(biases['encoder_b2'])
    # print(layer_2)
    return layer_2 # 256 x 128 # 784개 변수가 있던것을 128 변수.


def decoder(x): # x == 256 x 128
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), # 128 x 256
            biases['decoder_b1'])) # 256 x 256
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, # 256 x 256
            weights['decoder_h2']), biases['decoder_b2'])) # 256 x 784
    return layer_2 # 256 x 784


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)


#지도학습 앞단에 비지도학습인 AE( auto encoder)




y_pred = decoder_op
y_true = X
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1, num_steps+1):
    batch_x, _ = mnist.train.next_batch(batch_size)
    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
    if i % display_step == 0 or i == 1:
        print("Step %i Minibatch Loss: %f " % (i, l))



n = 4
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))
for i in range(n): # i 가 맨처음에 0 이고, j 가 0 이면.. 아래 아무리 곱해도 0이 되니, 0
    batch_x, _ = mnist.test.next_batch(n)
    # 4 x 784
    g = sess.run(decoder_op, feed_dict={X: batch_x}) # 복원된 이미지
    for j in range(n):
        canvas_orig[i * 28:(i+1)*28, j * 28:(j+1)*28] = batch_x[j].reshape([28, 28])

    for j in range(n):
        canvas_recon[i *28:(i+1 )*28, j * 28:(j+1) *28] = g[j].reshape([28, 28])





print("Original Images")
#이미지를 한 장으로 찍기 위해서,

0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0

plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin='upper', cmap='gray')
plt.show()



print('Reconstruction Images')
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin='upper', cmap='gray')
plt.show()




### AE : 입력과 출력이 동일한 회로 ( 특성- 분포추출 )
# VAE variational Auto encoder : 노이즈 , 분포의 평균 / 분산

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
mnist = input_data.read_data_sets("/tmp/data/")
reset_graph()
from functools import partial # 함수의 매개변수의 일부를 고정하기위해서 사용함.


n_inputs = 28*28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.001
initializer = tf.contrib.layers.variance_scaling_initializer()
# 원래는 dense layer를 사용해야하는데,  activation을 모두 쓰기 곤란하기 때문에, my_dense_layer
my_dense_layer = partial(#함수의 매개변수를 고정한 함수를 정의
        tf.layers.dense,
        activation=tf.nn.relu,
        kernel_initializer=initializer)
X = tf.placeholder(tf.float32, [None, n_inputs])
hidden1 = my_dense_layer (X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)

# 분포의 평균과 분산, + 노이즈
hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)
hidden3_sigma = my_dense_layer(hidden2, n_hidden3, activation=None)
# 노이즈 생성
noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)

hidden3 = hidden3_mean + hidden3_sigma * noise # Latent variables 잠재변수를 정의
hidden4 = my_dense_layer(hidden3, n_hidden4)

hidden5 = my_dense_layer(hidden4, n_hidden5)
logits = my_dense_layer(hidden5, n_outputs, activation= None)
outputs = tf.sigmoid(logits)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
reconstruction_loss = tf.reduce_sum(xentropy)


eps = 1e-10
#Log(0) 로 가지 못하게 하는 상태입니다. Log(0)을 방지하는 상수
#잠재변수 loss를 정의
latent_loss = 0.5 * tf.reduce_sum(
        tf.square(hidden3_sigma) + tf.square(hidden3_mean)
        - 1 - tf.log(eps + tf.square(hidden3_sigma)))

loss = reconstruction_loss  + latent_loss #잠재변수의 loss까지도 고려해줘야 합니다.
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
import sys
n_digits = 60
n_epochs = 50
batch_size = 150





with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_val, reconstruction_loss_val, latent_loss_val =  sess.run([loss, reconstruction_loss, latent_loss], feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train total loss:",   loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)
        saver.save(sess, "./my_model_variational.ckpt")

    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])

    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})


def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

plt.figure(figsize=(8,50))
for iteration in range(n_digits):
    plt.subplot(n_digits, 10, iteration + 1)
    plot_image(outputs_val[iteration])
plt.show()





### GAN
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
total_epoch = 100
batch_size = 100
learning_rate = 0.0002

n_hidden = 256
n_input = 28 * 28
n_noise = 128
X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

#생성기 변수
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01)) #128 x 256
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01)) #245 x 783
G_b2 = tf.Variable(tf.zeros([n_input]))

#분별기 변수
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev = 0.01)) #784 x 256
D_b1 = tf.Variable(tf.zeros([n_hidden]))

D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01)) #256 => 1개 값으로
D_b2 = tf.Variable(tf.zeros([1]))


def generator(noise_z): #100x128 의 노이즈를 100 x 784
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1) # 100 x 128, 128 x 256 => 100 x 256
    output = tf.nn.sigmoid( tf.matmul(hidden, G_W2) + G_b2) # 100 x 256, 256 x 784 =>
    return output

def discriminator(inputs):
    hidden = tf.nn.relu (tf.matmul(inputs, D_W1) + D_b1) # 100 x 784, 784 x 256 => 100 x 256
    output = tf.nn.sigmoid( tf.matmul(hidden, D_W2) + D_b2) # 100 x256 , 256 x 1 => 100 x 1
    return output

def get_noise (batch_size, n_noise):
    return np.random.normal(size = (batch_size, n_noise))

G = generator(Z) #이미지
D_gene = discriminator(G) # 생성된 이미지 G. -> 변경된 이미지
D_real = discriminator(X) # 원래 이미지 (변경되면서 입력)

#분포비교 = Contrast Divergency : Gradient log-likelyhood
# 분별기
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
# 생성기의 로스 값
loss_G = tf.reduce_mean(tf.log(D_gene))

D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# 최소화가 아닌 최대화를 최적화 시킴. (마이너스를 최소화)
train_D = tf.train.AdamOptimizer(learning_rate).minimize(
        -loss_D, var_list = D_var_list) # 수정한 변수리스트를 제한
train_G = tf.train.AdamOptimizer(learning_rate).minimize(
        -loss_G, var_list = G_var_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise) # 잡음 생성
        _, loss_val_D = sess.run([train_D, loss_D],
                feed_dict = {X: batch_xs, Z: noise})
        #생성기 망도 실행, 분별기망도 실행 => 역전파되는 것은 생성기만 실행.
        _, loss_val_G = sess.run([train_G, loss_G],
                feed_dict = {Z: noise})

    print('Epoch:', '%04d' % epoch,
            'D loss: {:.4}'.format(loss_val_D),
            'G loss: {:.4}'.format(loss_val_G))
    if epoch == 0 or (epoch + 1 ) % 10 == 0:
        sample_size = 10
        noise = get_noise (sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise}) #생성기함수의 결과값
        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))
        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28) ))
        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)),
                bbox_inches='tight')
        plt.close(fig)

print("최적화 완료")
