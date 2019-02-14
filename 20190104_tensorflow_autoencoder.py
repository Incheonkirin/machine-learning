import os
os.chdir("C://Users/Hyungi/Desktop/workplace")
os.getcwd()

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
# 이미지 => RNN
# 28 * 28: RNN적 특징.
# RNN적 특징이 뭔가요? 예를 들어, 어떤 픽셀 다음에 어떤 픽셀이 온다는 특징이지. 논리적으로 보면
mnist = input_data.read_data_sets("./mnist/data/", one_hot=False)

X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()


n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
n_layers = 3
learning_rate = 0.001

# 28 step, 28 cell
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#                               None으로 들어오는 건 무슨 사이즈? 배치사이즈 죠. 여기선 150
y = tf.placeholder(tf.int32, [None])

lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units = n_neurons)
        for layer in range(n_layers)]

multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

top_layer_h_state = states[-1][1]
print(top_layer_h_state)
# 다 구해졌으니까, 여기서부터는 뭐에요? FC죠. fully connected neural network
# matrix를 자동으로 계산해서 설정
# 특징 차수, 출력 차수

logits = tf.layers.dense(top_layer_h_state, n_outputs, name='softmax')
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(xentropy, name='loss')
# SGD : 가 뭐라고했죠? 확률적으로하니까 배치사이즈만큼해서, 가중치 역전파한다고 했죠?
# 가중치 역전파해서 평균치를 잡아주는 거에요
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

n_epochs = 10
batch_size = 150
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Epoch", epoch, "Train accuracy =", acc_train, "Test accuracy =", acc_test)


# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for iteration in range(mnist.train.num_examples // batch_size):
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             print(y_batch)
#             print(y_batch.shape)
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_val = accuracy.eval(feed_dict={X: mnist.validation.images,
#                                             y: mnist.validation.labels})
#         print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)



from tensorflow.examples.tutorials.mnist import input_data
# 이미지 => RNN
# 28 * 28: RNN적 특징.
# RNN적 특징이 뭔가요? 예를 들어, 어떤 픽셀 다음에 어떤 픽셀이 온다는 특징이지. 논리적으로 보면
mnist = input_data.read_data_sets("MNIST_data/")


# hyperparameters
n_neurons = 128
learning_rate = 0.001
batch_size = 128
n_epochs = 10
# parameters
n_steps = 28 # 28 rows
n_inputs = 28 # 28 cols
n_outputs = 10 # 10 classes


# build a rnn model

# 28 step, 28 cell
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#                               None으로 들어오는 건 무슨 사이즈? 배치사이즈 죠. 여기선 150
y = tf.placeholder(tf.int32, [None])
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
logits = tf.layers.dense(state, n_outputs)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


# input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=False)

X_test = mnist.test.images # X_test shape: [num_test, 28*28]
X_test = X_test.reshape([-1, n_steps, n_inputs])
y_test = mnist.test.labels


# initialize the variables
init = tf.global_variables_initializer()
# train the model
with tf.Session() as sess:
    sess.run(init)
    n_batches = mnist.train.num_examples // batch_size
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            X_train, y_train = mnist.train.next_batch(batch_size)
            X_train = X_train.reshape([-1, n_steps, n_inputs])
            sess.run(optimizer, feed_dict={X: X_train, y: y_train})
        loss_train, acc_train = sess.run(
            [loss, accuracy], feed_dict={X: X_train, y: y_train})
        print('Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.3f}'.format(
            epoch + 1, loss_train, acc_train))
    loss_test, acc_test = sess.run(
        [loss, accuracy], feed_dict={X: X_test, y: y_test})
    print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss_test, acc_test))


#

import pprint
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()
h = [1, 0, 0, 0]# 4차원 벡터
e = [0, 1, 0 ,0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

with tf.variable_scope('five_sequences') as scope:
    hidden_size = 2
    # 4차원이 2차원으로 줄어든다는 이야기입니다. 2차원 벡터로 embedding한다.
    cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_size) # 2개
    x_data = np.array([[h, e, l, l, o]], dtype=np.float32)
    print(x_data.shape) # 1, 5, 4
    pp.pprint(x_data)
    outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval()) # 5 x 2 # embedding



# 끝 단어 예측 ( 4자 중 3자를 입력하면, 1자를 예측합니다.)
# 전체 알파벳 26자
ops.reset_default_graph()

char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
        'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u',
        'v', 'w', 'x', 'y', 'z']
num_dic = {n: i for i, n in enumerate(char_arr)} #dictionary로 나옴.
dic_len = len(num_dic)
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love',
        'kiss', 'kind']


def make_batch(seq_data):
    input_batch = []
    target_batch = []
    for seq in seq_data: # 22, 14, 17
        input = [num_dic[n] for n in seq[: -1]] # word의 wor 가 들어가고, w 가 나가고, 22번이 호출. 22, 14, 17
        target = num_dic[seq[-1]] #타겟은 d 3번이됨.
        #인코딩

        #원핫인코딩
        input_batch.append(np.eye(dic_len)[input])  # eye는 단위행렬을 만들어줍니다. 대각선으로 1111111 인데, w 22번째가 1인 , 14번째가 1인 17번째가 1인..
        target_batch.append(target)
    return input_batch, target_batch



learning_rate = 0.01
n_hidden = 128 # 특성
total_epoch = 30
n_step = 3 #왜 step이 3개인가요? 'wor' 3개죠?
n_input = n_class = dic_len # 26개  # n_input이 몇개에요? 셀 수니까 26개

X = tf.placeholder(tf.float32, [None, n_step, n_input])
# 배치사이즈 10개가 들었으니, 10개고. 3이고, 26이고.

Y = tf.placeholder(tf.int32, [None])
#

W = tf.Variable(tf.random_normal([n_hidden,n_class])) #128 by 26행렬이 만들어지죠.
b = tf.Variable(tf.random_normal([n_class]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden) # 128개로 이루어진 특성값이 나옴. # 128 x 26
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5, seed=10)  # dropout은 과적합을 방지하기 위해서 ,
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden) #여기는 dropout안했죠
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
# 10 ,  3 , 128 아웃풋이 이렇게 나오는 거에요.
outputs = tf.transpose(outputs, [1, 0, 2]) # 면 과 행을 바꾸는 거지요? 면/행  3x 10 x 128 로 되는 거지요? 10의 의미가 뭔가요?  배치사이즈니까 10개의 단어쌍이라는 것이고, 3개는? 단어수지. 단어수.  128은 128의 특성값을 만들어서 내보냈다는 것이고, 뭐가 RNN이 뽑아준 특성이란 말이죠. RNN이 뽑아준 특성을 이렇게 바꾼이유는 아웃풋-1해주고있죠. 요 3개중에 마지막걸 쓰겠다는 거야. # wor => 특성수를 맨 마지막것을 쓰겠다. 수평적으로 연관되지만, 수직적으로 연관. 수평수직적 특징이 추출
# 그림으로 보여주심.

# ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~
#
# w = ~~~~~~~~~~~~~~
# o = ~~~~~~~~~~~~~~
# r = ~~~~~~~~~~~~~~



outputs = outputs[-1]
# 1 x 10 x 128 이 결과적으로 나온 것이에요.
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, target_batch = make_batch(seq_data) #원핫인코딩 데이터와 타겟값.
for epoch in range(total_epoch): #같은 데이터로 30번. 가중치 조절
    _, loss = sess.run([optimizer, cost], # optimizer cost의 그래프 꼭지를 돌리고..
                       feed_dict={X: input_batch, Y: target_batch}) #그러면 모델이 만들어졌겠죠? 모델에 데이터를 넣으면? 예측이 되지요.
    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

prediction = tf.cast(tf.argmax(model, 1), tf.int32) # model이 예측 된 값. argmax 열방향으로 한개를 찾아서, prediction 되고.
prediction_check = tf.equal(prediction, Y) # Y와 같은 지 보고
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy],
                                 feed_dict={X: input_batch, Y: target_batch})

# 인덱스값으로 출력 => 단어장에서 글자확인
predict_words = []
for idx, val in enumerate(seq_data): # seq_data ==원래 입력데이터
    last_char = char_arr[predict[idx]] #predict단어를 char arr에서 찾으니까. predict 해서 나온 결과값이 뭐에요? 알파벳 수치가 되는거지? 그게 a= 1 b =2  # 마지막 알파벳 예측
    predict_words.append(val[:3] + last_char)

print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)




이미지 소리 based knowledge


##
#translate

ops.reset_default_graph()
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']


num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)
seq_data = [['word', '단어'], ['wood', '나무'],
        ['game', '놀이'], ['girl', '소녀'],
        ['word', '단어'], ['love', '사랑']]

def make_batch(seq_data):
    input_batch = []
    output_batch = []
    for seq in seq_data :
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]
        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)
    return input_batch, output_batch, target_batch




learning_rate = 0.01
n_hidden = 128
total_epoch = 100
n_class = n_input = dic_len

enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets   = tf.placeholder(tf.int64, [None, None])

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5,seed=100)
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                             dtype=tf.float32)
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)
cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

def translate(word):
    seq_data = [word, 'P' * len(word)]
    input_batch, output_batch, target_batch = make_batch([seq_data])
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})
    decoded = [char_arr[i] for i in result[0]]
    end = decoded.index('E')
    translated = ''.join(decoded[:end])
    return translated

print('word ->', translate('word'))
print('love ->', translate('love'))
print('abcd ->', translate('abcd'))
print('wodr ->', translate('wodr'))
print('loev ->', translate('loev'))




import sys
import tensorflow as tf
import numpy as np

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']

num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = [['word', '단어'], ['wood', '나무'], ['game', '놀이'],
            ['girl', '소녀'], ['kiss', '키스'], ['love', '사랑']]

def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []
    for seq in seq_data:
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]
        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)
    return input_batch, output_batch, target_batch

def translate(sess, model, word):
    seq_data = [word, 'P' * len(word)]
    input_batch, output_batch, target_batch = make_batch([seq_data])
    prediction = tf.argmax(model, 2)
    result = sess.run(prediction, feed_dict={enc_input: input_batch,
        dec_input: output_batch,
        targets: target_batch})
    decoded = [char_arr[i] for i in result[0]]
    try:
        end = decoded.index('E')
        translated = ''.join(decoded[:end])
        return translated
    except:
        return ''.join(decoded)

learning_rate = 0.01
n_hidden = 128
total_epoch = 100

n_input = n_class = dic_len

enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
        initial_state=enc_states, dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs', sess.graph)
    input_batch, output_batch, target_batch = make_batch(seq_data)
    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost],
            feed_dict={enc_input: input_batch,
                dec_input: output_batch,
                targets: target_batch})
        print('Epoch:', '%04d' % (epoch + 1),
            'cost =', '{:.6f}'.format(loss))
        sys.stdout.flush()
    print('완료!')
    print('\n=== 테스트 ===')
    print('word ->', translate(sess, model, 'word'))
    print('wodr ->', translate(sess, model, 'wodr'))
    print('love ->', translate(sess, model, 'love'))
    print('loev ->', translate(sess, model, 'loev'))
    print('abcd ->', translate(sess, model, 'abcd'))







# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.
# 영어 단어를 한국어 단어로 번역하는 프로그램을 만들어봅니다.
import tensorflow as tf
import numpy as np

# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
# 중복을 방지하고, 원핫인코딩을 하기 위해서 사용함.


num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 영어를 한글로 번역하기 위한 학습 데이터
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑']]

#여기에 input, output, target 3개가 있는 이유는 무엇인가요?
# 번역망은 어떻게 생겼나요? 왼쪽에 입력이 들어오고, 오른쪽에 번역이 들어오고, 그러면 상단에 목표(타겟)이 있어야하죠? 그러면 상단에 있는게 오른쪽에 있는 것과 같아야 하죠
# 그래서 2개는 같아요. output 과 target , 단지 분류하기 위해 S 와 E를 붙였습니다.

def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
        input = [num_dic[n] for n in seq[0]]
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        output = [num_dic[n] for n in ('S' + seq[1])]
        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return input_batch, output_batch, target_batch
#망이 몇개인가요? 망이 2개입니다.
# 아래보면, encode decode 2개 있어요.
# 2개가 서로 연결되어야 해요. 어떻게 연결 되어있나요?
# (망이 두개 별도로 만들어져있어요. 관계를 만들어야 하죠.)


#########
# 옵션 설정
######
learning_rate = 0.01
n_hidden = 128
total_epoch = 100
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_class = n_input = dic_len


#########
# 신경망 모델 0
######
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
# [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])

# NMT 모델 (seq2seq)
# NMT 모델을 찾아보세요. 이 내용들을 이해한다면, NMT 모델을 사용할 수 있고, NMT모델은 더욱 복잡한 문제를 해결할 수 있습니다.
#번역망 : 2개의 망으로 구성되어있고, 좌우 망은 initial state을 통해서 연결되어 있습니다.
tf.reset_default_graph()

# 인코더 셀을 구성한다.
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)


# 디코더 셀을 구성한다.
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을
    # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)


# 셀과 셀로 넘어가는 것 state
# initial state .가 달라요. 앞에서 모두 학습한 특성이 넘어가는 거죠.
# 두개의 망을 만들고, (encode, decode) 두개의 망을 initial state를 통해서 연결시킵니다.
# 앞에 단어에서 들어오는 특징만 만들어서,


model = tf.layers.dense(outputs, n_class, activation=None)


cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')


#########
# 번역 테스트
######
# 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
def translate(word):
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    # ['word', 'PPPP']
    seq_data = [word, 'P' * len(word)]

    input_batch, output_batch, target_batch = make_batch([seq_data])

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_arr[i] for i in result[0]]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated


print('\n=== 번역 테스트 ===')

print('word ->', translate('word'))
print('wodr ->', translate('wodr'))
print('love ->', translate('love'))
print('loev ->', translate('loev'))
print('abcd ->', translate('abcd'))





import csv
import random
import numpy as np
import random
from tensorflow.python.framework import ops
ops.reset_default_graph()

response = 6
batch_size = 50
symmetry = ['rotate180', 'rotate90', 'rotate270', 'flip_v', 'flip_h']

def print_board(board):
    symbols = ['O', ' ', 'X']
    board_plus1 = [int(x) + 1 for x in board]
    print(' ' + symbols[board_plus1[0]] + ' | ' + symbols[board_plus1[1]]
            + ' | ' + symbols[board_plus1[2]])
    print('____________')
    print(' ' + symbols[board_plus1[3]] + ' | ' + symbols[board_plus1[4]]
            + ' | ' + symbols[board_plus1[5]])
    print('____________')
    print(' ' + symbols[board_plus1[6]] + ' | ' + symbols[board_plus1[7]]
            + ' | ' + symbols[board_plus1[8]])



def get_symmetry(board, response, transformation):
    if transformation == 'rotate180':
        new_response = 8 - response
        return(board[:: -1], new_response)
    elif transformation == 'rotate90' :
        new_response = [6, 3, 0, 7, 4, 1, 8, 5, 2].index(response)
        tuple_board = list(zip(*[board[6:9], board[3:6], board[0:3]]))
        return ([value for item in tuple_board for value in item], new_response)
    elif transformation == 'rotate270':
        new_response = [2, 5, 8, 1, 4, 7, 0, 3, 6].index(response)
        tuple_board = list(zip(*[board[0:3], board[3:6], board[6:9]]))[::-1]
        return ([value for item in tuple_board for value in item], new_response)
    elif transformation == 'flip_v' : # 0 1 2 , 3 4 5, 6 7 8
        new_response = [6, 7, 8, 3, 4, 5, 0, 1, 2].index(response)
        return (board[6:9] + board[3: 6] + board[0:3], new_response)
    elif transformation == 'flip_h':
        new_response = [2, 1, 0, 5, 4, 3, 8, 7, 6].index(response)
        new_board = board[::-1]
        return(new_board[6:9] + new_board[3:6] + new_board[0:3], new_response)
    else:
        raise ValueError('해당하는 경우가 없습니다')



def get_moves_from_csv(csv_file):
    moves = []
    with open(csv_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            moves.append(([int(x) for x in row[0:9]], int(row[9])))
    return(moves)




# 데이터를 생성 - 경우가 많기 때문에 일일이 만들어서 훈련할 수 없으니까
# 일부 데이터를 만들어서, 뒤집고, 90도 돌리고 270 돌리고 해서 경우의 수를 만들고 있습니다.
# 왜냐면 원점을 중심으로 대칭하는 경우의수가 같기 때문에, 같은 결과가 나오기 때문에..
# 이렇게 해서 경우의 수를 늘리고 있는 것이죠.
def get_rand_move(moves, n=1, rand_transforms=2):
    (board, response) = random.choice(moves)
    possible_transforms = ['rotate90', 'rotate180', 'rotate270',
            'flip_v', 'flip_h']
    for i in range(rand_transforms):
        random_transform = random.choice(possible_transforms)
        (board, response) = get_symmetry(board, response, random_transform)
    return (board, response)




moves = get_moves_from_csv('datasets/tictactoe_moves.csv')
train_length = 500
train_set = []
print(train_set)


for t in range(train_length):
    train_set.append(get_rand_move(moves))
print(len(train_set))
print(train_set)
test_board = [-1, 0, 0, 1, -1, -1, 0, 0, 1]
train_set = [x for x in train_set if x[0] != test_board]


def init_weights(shape):
    return(tf.Variable(tf.random_normal(shape)))



def model(X, A1, A2, bias1, bias2):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, A1), bias1))
    layer2 = tf.add(tf.matmul(layer1, A2), bias2)
    return (layer2)




X = tf.placeholder(dtype=tf.float32, shape=[None, 9])
Y = tf.placeholder(dtype=tf.int32, shape=[None])
A1 = init_weights([9, 81])
bias1 = init_weights([81])
A2 = init_weights([81, 9])
bias2 = init_weights([9])
model_output = model(X, A1, A2, bias1, bias2)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model_output, labels=Y))

train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
prediction = tf.argmax(model_output, 1)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


loss_vec = []
for i in range(10000):
    rand_indices = np.random.choice(range(len(train_set)), batch_size,
            replace=False)
    batch_data = [train_set[i] for i in rand_indices]
    x_input = [x[0] for x in batch_data]
    y_target = np.array([y[1] for y in batch_data])
    sess.run(train_step, feed_dict = {X: x_input, Y: y_target})
    temp_loss = sess.run(loss, feed_dict={X: x_input, Y: y_target})
    loss_vec.append(temp_loss)
    if i%500==0:
        print('iteration' + str(i) + 'Loss: ' + str(temp_loss))


import matplotlib.pyplot as plt
plt.plot(loss_vec, 'k-', label='Loss')
plt.title('Loss (MES) ')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

test_boards = [test_board]
feed_dict = {X: test_boards}
logits = sess.run(model_output, feed_dict = feed_dict)
predictions = sess.run(prediction, feed_dict=feed_dict)
print(predictions)




def check(board):
    wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]]
    for i in range(len(wins)):
        if board[wins[i][0]] == board[wins[i][1]]==board[wins[i][2]]==1.:
            return(1)
        elif board[wins[i][0]] == board[wins[i][1]]==board[wins[i][2]]==-1.:
            return(-1)
    return(0)


game_tracker = [0., 0., 0., 0., 0., 0., 0., 0., 0.] #말이 결정
win_logical = False
num_moves = 0



while not win_logical:
    player_index = input("이동하고 싶은 인덱스를 입력하시오 (0-8): ")
    num_moves += 1
    game_tracker[int(player_index)] = 1.

    [potential_moves] = sess.run(model_output,
            feed_dict={X: [game_tracker]})
    allowed_moves = [ix for ix, x in enumerate (game_tracker) if x ==0.0]
    model_move = np.argmax([x if ix in allowed_moves else -999.0
            for ix, x in enumerate(potential_moves)])
    game_tracker[int(model_move)] = -1.
    print("모델이 이동하였습니다.")
    print_board(game_tracker)
    if check(game_tracker)==1 or num_moves>= 20:
        print("게임 종료 ! 승리하셨습니다. ")
        win_logical = True
    elif check(game_tracker) == -1 :
        print("게임 종료 ! 게임에 지셨습니다. ")
        win_logical = True



# turn around game : 장기 같은 게임. (무한 루프로 진행함) 중간에 승패를 확인하여, 승패를 확인 - 종료
# check 함수 -> 이겼는지 졌는지를 확인해줌. 20회를 넘어가지 않도록 제한.
# win logical 게임의 승패가 결정되면 종료되는 .
# 인공지능을 게임에 도입한다 == 몬스터에 '지능'을 달아줍니다
# 잘 조작하면, 게임의 상태에 따라서 승률이 일정하게 유지될 수 있도록, (adaptive하게 작동하는 것이 요즘 추세)







# check #print #get_symmetry함수들의 원리를 잘 보세요




g1 = tf.Graph()
with g1.as_default():
    c1 = tf.constant(1, name="c1")

type(c1)


c1

c1.op.node_def


g1.as_graph_def()

with tf.Session(graph=g1) as sess:
    print(sess.run(c1))

c1

g2 = tf.Graph()
with g2.as_default():
    v1 = tf.Variable(initial_value=1, name="v1")

type(v1)
v1
v1.op.node_def
v1._variable

v1._variable.op.node_def


g2.as_graph_def()




with g2.as_default():
    v2 = tf.Variable(initial_value=2, name="v2")
with tf.Session(graph=g2) as sess:
    init = tf.global_variables_initializer()
init.node_def

v1.initializer



with tf.Session(graph=g2 ) as sess:
    sess.run(v1.initializer)
    sess.run(v2.initializer)
    #위 두 라인과 동일한 효과를 냅니다.
    sess.run(tf.global_variables_initializer())
    #변수를 실행한다는 것은 변수안의 텐서 연산을 실행하는 것입니다.
    print(sess.run([v1, v2]))
    print(sess.run([v1._variable, v2._variable]))




import tensorflow as tf
from tensorflow.contrib import learn as skflow

classifier = skflow.TensorFlowDNNClassifier (
        hidden_unitts=[10, 20, 10],
        n_classes= 2,
        batch_size= 128,
        steps= 500,
        learning_rate= 0.05)



import tensorflow as tf

x = tf.constant([[1.0, 2.0, 3.0]])
w = tf.constant([[2.0], [2.0], [2.,]])
y = tf.matmul(x, w)
print(x.get_shape())

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y)

print(result)



import tensorflow as tf

x = tf.Variable([[1., 2., 3.]], dtype = tf.float32)
w = tf.constant([[2.],[2.],[2.]], dtype=tf.float32)
y = tf.matmul(x, w)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y)

print(result)




import tensorflow as tf

input_data = [[1., 2., 3.],[1., 2., 3.], [2., 3., 4.]] # 3x 3 matrix
x = tf.placeholder(dtype = tf.float32) # 3x 1 matrix
w = tf.Variable([[2.],[2.],[2.]], dtype = tf.float32) # 3x 1 matrix
y = tf.matmul(x, w)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y, feed_dict={x:input_data})

print(result)
