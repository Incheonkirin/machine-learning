# conda update --all
# GPU설치시는 graphic 카드 확인 (지원여부)
# cuda , cuda -dnn을 설치
# conda install tensorflow-gpu

# conda install tensorflow





#여러개의 버전을 동시에 사용하기 위한 설치 방법
# conda cerate --name tensorflow python=3.6.3 anaconda


# activate tensorflow
# conda update pip
# conda install numpy
# conda install pandas
# conda install jupyter
# conda install scipy


# conda install scikit- learn
# conda install bokeh
# conda install seaborn
# conda install tensorflow


# jupyter kernelspec install-self
# python -m ipykernel install --name tensorflow
# deactivate


# Anaconda 폴더 아래 괄호하고 tensorflow라고 원래 파일과 같은 파일이 생긴다.





#
import tensorflow as tf
print(tf.__version__)

# 앞으로 추가 설치는 tensorflow의 프롬프트에 들어가서 해야 합니다.

# pipenv라는 것이 있음.


# >>> import tensorflow as tf
# >>> hello = tf.constant(‘Hello, TensorFlow!’)
# >>> sess = tf.Session()
# >>> sess.run(hello)
# b’Hello, TensorFlow!’
# >>> a = tf.constant(10)
# >>> b = tf.constant(32)
# >>> sess.run(a + b)
# 42
# >>>



#constant , variable,  placeholder

hello = tf.constant('Hello, Tensorflow')

#device 분리
sess = tf.Session()
print(sess.run(hello)) #변수 확인
# run을 해야, numpy와 호환되는 포멧으로 변경
print(hello) # tensorflow에서 사용하는 데이터 Tensor : 직접 확인이 불가능함.


x = 35
y = x + 5
print(y)

x = tf.constant(35, name='x') # tensorboard에서 사용하기 위한 이름 , tensor의 이름.
y = tf.Variable(x + 5, name='y') #
print(y)
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    print (session.run(y))


a = tf.constant([1, 2, 3], name='a')
b = tf.constant([4, 5, 6], name='b')
add_op = a + b

with tf.Session() as session:
    print(session.run(add_op))



# 문제 :
[[1, 2, 3], [4, 5, 6]] + 4

# 결과 :
=[[5, 6, 7]
[8, 9, 10]]

# 풀이
i = tf.constant(4, name='i')
j = tf.constant([[1, 2, 3],[4, 5, 6]], name='j')

add_op = i+j

with tf.Session() as session:
    print(session.run(add_op))


# 주입변수
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
total = tf.multiply(add, mul)
with tf.Session() as sess:
    print("feed 변수 합: %i" % sess.run(add, feed_dict={a: 2, b:3}))
    print("feed 변수 곱: %i" % sess.run(mul, feed_dict={a: 2, b:3}))
    print("feed 변수 곱: %i" % sess.run(total, feed_dict={a: 2, b:3}))

#
matrix1 = tf.constant([[3., 3.]]) #2행 1열
matrix2 = tf.constant([[2.], [2.]]) #1행 2열
product = tf.matmul(matrix1, matrix2) #행렬 곱/ 행렬 내적 matmul = matrix multiply
with tf.Session() as sess:
    result = sess.run(product)
    print(result)


#numpy / tensorflow
import numpy as np
a = np.zeros((3, 3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))

#convert to tensor를 해 주어야, 계산에 참여할 수 있습니다



##
my_var = tf.Variable(3, name="var1")

add = tf.add(5, my_var)
add

num = tf.multiply(8, my_var)
num

my_var = num
zeros = tf.zeros([2,2])
ones = tf.ones([6])
uniform = tf.random_uniform([3, 3, 3], minval = 0.0, maxval =10)
normal = tf.random_normal([ 3, 3, 3], mean=0.0 , stddev=2.0)

#표준편차 2배수 내로 제한
trunc = tf.truncated_normal([2, 2], mean=5.0, stddev=1.0)


#신경망에서는 랜덤으로 초기화 하는 경우가 많다.
random_var = tf.Variable(tf.truncated_normal([2, 2])) #가중치 랜덤으로 초기화
random_var

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(normal))
print(sess.run(trunc))
print(sess.run(random_var))
print(sess.run(add))
print(sess.run(num))

print("변수 초기화", sess.run(my_var))


sess = tf.Session()
diagonal = [1, 2, 3, 4]
print(tf.diag(diagonal))
sess.run(tf.diag(diagonal))
dig = tf.diag(diagonal)
sess.run(tf.diag_part(dig))
#주입변수와 variable 혼용
#행 열 면에서, 마지막 차수는 미지정 가능,
# ** 그러나 상황에 따라 다르다고 한다. ==> 책보고 익히기
#단, '면'  '행'  '열'.
#가장 기본이 되는 '열' 이 지정 되지 않는다고 하면 안된다.
#행 과 열이 있다고 하면 (2차원이라고 하면) '열'을 지정하지 않으면 안된다.
p = tf.placeholder(tf.float32, shape=[], name="p")
v2 = tf.Variable(2., name="v2")
a = tf.add(p, v2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v2))
    print(sess.run(a, feed_dict={p: 3.0}))

#주입변수로 이루어진 연산자에, 주입 없이 연산자를 호출하게 되면 에러가 발생한다.
    try:
        sess.run(a)
    except Exception as e:
        print(e)



import tensorflow as tf
my_tensor = tf.constant(0, shape=[6, 2])
my_static_shape = my_tensor.get_shape()
print(type(my_static_shape))
print(my_static_shape)

print(my_static_shape.as_list())
#내부의 값 확인할 때는 run이나 aslist 통해 확인해야함.
#python core 데이터 타입으로 변경.
my_tensor_transposed = tf.transpose(my_tensor)
#                          전치
print(my_tensor_transposed.get_shape())



my_placeholder = tf.placeholder(tf.float32, shape=[None, 0])
print('맨 선두 생략', my_placeholder.get_shape())

#에러 가 남. 차수가 맞지 않음.
my_placeholder.set_shape([8, 2])


# as list 나 run을 써야 하는 불편함 때문에 InteractiveSession()이라고 하는 것을 지원합니다.

tf.InteractiveSession()
tf.zeros(2)
a = tf.zeros(2)
print(a.eval())
print(tf.size(a))
# size하면 size의 객체가 나오죠

# 그래서 이 것을 이용해서 함수를 하나 만들어주고,
# 그러면 InteractiveSession()으로 바로 확인 할 수 있도록 만든 다음에

def showvalue(t):
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print(t.eval())
    sess.close()

c2, c3 = tf.constant([1.2, 5.6]), tf.constant([-4, -1, 7])
v2, v3 = tf.constant([2.3, 4.5]), tf.constant([-2, 3, 5])
showvalue(tf.add_n([c2, v2]))
showvalue(tf.add_n([c3, v3, v3]))
showvalue(tf.abs(c2))
showvalue(tf.negative(c2))



#방정식의 해를 구해봅니다.
# 2x + y + z = 1 #검산
# 4x + 3y + 4z = 2
# -4x + 2y + 2z = -6
sess = tf.Session()
x = tf.constant([2.0, 1.0, 1.0, 4, 3, 4, -4, 2, 2], shape=[3, 3])
print(sess.run(x))
y = tf.constant([1.0, 2, -6], shape=[3, 1])
print(sess.run(y))

z = tf.matrix_solve(x, y) #matrix_solve란, matmul(내적)과 같은데, 역행렬을 곱해서 구하는 것
result = sess.run(z)
print("계수: ", result)

# 직접 쓰면,
x_1 = tf.matrix_inverse(x)
z = tf.matmul(x_1, y)
print("해:", sess.run(z))

# 즉, matrix_solve == matrix_inverse + matmul



# 행렬 거듭제곱할 때는
x = tf.constant([1, 2, 3, 4, 5, 6], shape = [2, 3])
x = tf.transpose(x) #행렬 거듭제곱을 하기 위해서는 #전치행렬을 구해서 곱해줘야 합니다.
print(sess.run(x))

# 그러면 2차원 이상의 행렬은 어떻게 됩니까?

#   3차원일 때는 행렬 곱, 면은 상관없이. 행-열만 전치 되어 있으면 됩니다.
a = tf.constant(np.arange(1, 13, dtype=np.int32),
        shape=[2, 2, 3])
print(sess.run(a))

b = tf.constant(np.arange(13, 25, dtype=np.int32),
        shape=[2, 3, 2])

print(sess.run(b))

c = tf.matmul(a, b)
sess.run(c)


#문제 : 4차원의 거듭 제곱
a_1 = tf.range(1, 25, 1, dtype= tf.float32)
a = tf.reshape (a_1, (2, 2, 2, 3))
# 공식은 똑같다.             ---- 이 부분만 전치.

# 정답
d = tf.matmul(a, tf.transpose(a, perm=[0, 1, 3, 2]))
#                              **transpose하는데, 2와 3만 바꿔라
sess.run(d)




# 더 쉽게하려면
e = tf.matmul(a, a, transpose_b=True)
# 같은 행렬을 주고, transpose_b에 True를 주거나
sess.run(e)



# reshape에 대해서 배워보겠습니다.

tensor = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
print(sess.run(tf.reshape(tensor, [2, -1]))) #2행 (6열)로 만들어라 #reshape에서 '-1'은 계산해서 채워라
tf.reshape(tensor, [-1, 6])
print(sess.run(tf.reshape(tensor, [2, -1, 3]))) # -1 부분은 2가 될 것이다.



#CNN에서 차수 일치를 위해서 공간을 채우는데,  Filter 생성시 사용.

t = [[1, 2, 3], [4, 5, 6]] #데이터 가 있을때
paddings = [[1, 1,], [2, 2]] #상하로 행이 1개씩 들어가 있고, 좌우로 행이 2개씩 늘어나 있어요
print(sess.run(tf.pad(t, paddings, "CONSTANT")))


paddings = [[1, 1,], [1, 1]]
print(sess.run(tf.pad(t, paddings, "REFLECT"))) #반사형태로 "거울에 비춘것 처럼 반대쪽에 데이터가 온다."


# 그 다음에 씨메트릭이 있습니다. 중앙을 중심으로 하고, 똑같은 것 끼리
#
print(sess.run(tf.pad(t, paddings, "SYMMETRIC")))




####
# 이제 단순 한 것 말고, 재미있는 것을 해 보겠습니다.

import numpy
rng = numpy.random

learning_rate = 0.01 #학습률(오차 수정 정도) #작으면 오래걸리고, 크면 점프가 크고
training_epochs = 4000 #몇 번을 반복/회전하는가
display_step = 50 #에포크를 디스플레이 스텝으로 나눕니다. ==0 50의 배수 에 점을 찍습니다. 4천개가 될 때까지 돌아갑니다.


train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = train_X.shape[0] #1차원이니까 갯수가 됨.
#GPU 병렬처리
X = tf.placeholder("float") #주입. 들어올 수 있도록
Y = tf.placeholder("float")
#
#가중치와 바이어스를 random으로 초기화 시킴
W = tf.Variable(rng.randn(), name='weight') #가중치
b = tf.Variable(rng.randn(), name='bias') #바이어스 = 0으로 가는 것을 방지하고, 절편과 같은 효과

activation = tf.add(tf.multiply(X, W), b) #예측

# L2 loss를 정의
cost = tf.reduce_sum(tf.pow(activation-Y, 2)/(2*n_samples)) # 비용함수를 정의
#예측 값에 Y를 빼주고 제곱을 해주고, n_sample로나누고

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #비용을 최소화, solver를 지정:GradientDescentOptimizer 하는데 learning rate를 매개변수로 받고.
# minimize를 찾는데, (cost)의 minimize를 찾아라

init = tf.initialize_all_variables()




import matplotlib.pylab as plt

with tf.Session() as sess: #세션을 열고, 초기화 시킵니다.
    sess.run(init) #초기화 시키면, placeholder와 Variable 공간을 확보합니다. #

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y}) # X데이터와 Y데이터 받아서 ZIP해서 집어 넣어줍니다. dictionary로 짝을 맞춰서 들어갑니다.

        if epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(sess.run(
            fetches=cost, feed_dict={X: train_X, Y:train_Y})),
            "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished! ")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}),
    "W=", sess.run(W), "b=", sess.run(b))

    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()



##
# sklearn과 연결해서

from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph() #변수 재지정 에러 발생
# reset_default_graph를 호출하지 않고, 똑같은 변수를 호출하면 에러가 나기 때문에, reset_default_graph를 이용해서 지워주고 다시 시작해야 한다.
sess = tf.Session()

iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data]) # x[3]은 무슨 의미인가요? #변수가 4개가 있죠? 그 중 3번째 변수를 사용하고,
y_vals = np.array([y[0] for y in iris.data]) # y[0] 0번째 것을 y_vals로

batch_size = 25 # 한번에 25개 씩 들어가는 거죠. 아까는 어떻게 했죠? epoch를 4천개 씩 줬죠. 이번에는 데이터 입력 사이즈로 batch_size를 사용합니다.
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32) #placeholder 몇개씩 들어올 지 모르겠어요. 그래서 None으로 줬어요.
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1, 1])) # Variable은 1,1  == 1행 1열
b = tf.Variable(tf.random_normal(shape=[1, 1]))


model_output = tf.add(tf.matmul(x_data, A), b) #예측값 # 텐서플로우에서는 model_ouput만 해주면 예측이 됩니다.
loss = tf.reduce_mean(tf.square(y_target - model_output)) # y_target이 실제값 (종속변수) , 에서 -model_output빼주고, 루트 평균을 해줌 ==> loss function은 그때 그때 다를 수 밖에 없기 때문에 어려운 부분입니다.

my_opt = tf.train.GradientDescentOptimizer(0.05) #optimizer 를 2줄로 해준겁니다.
train_step = my_opt.minimize(loss)

init = tf.initialize_all_variables()
sess.run(init)
loss_vec = [] #loss값은 점점 줄어들기 때문에 우하향하는 값을 가지고 있습니다.



# 앞으로 하는 것은 모두 위 같은 형태입니다
# 데이터를 준비하고
# 변수를 마련하고
# 예측과 로스함수
# 옵티마이져 최적화기
# 그리고 훈련 하면 됩니다.


for i in range(100): #epoch를 100 , 100번 돌아가라.
    rand_index = np.random.choice(len(x_vals), size= batch_size) # 1 Epoch마다 (iris데이터가 150개임) 150개 데이터 중에 (배치 사이즈만큼)15개씩 받아오라.
    rand_x = np.transpose([x_vals[rand_index]]) # 행으로 되어있는 데이터를 하나씩 점프하기 위해서 트랜포즈해줌.
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y}) #run할때 호출은 optimize하는 것을 넣어주면 되요. train_step을 실행 => 로스를 실행 => 모델 아웃풋 => 변수가 있어야함 => 플레이스 홀드에 데이터 들어와야 함.
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%25 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A) ) + ' b = '
                + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))



# 여기까지가 훈련 끝이고, 구해진 절편 슬롶 을 출력하면 됩니다.

[slope] = sess.run(A) #최종 기울기
[y_intercept] = sess.run(b)#최종 바이어스
best_fit = [] # 예측 데이터
for i in x_vals:
    best_fit.append(slope*i + y_intercept) # 각 데이터에 대한 예측값

plt.plot(x_vals, y_vals, 'o', label= 'Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')

plt.show()
plt.plot(loss_vec, 'k-')

plt.show()




#이번에는 그래프를 한번 출력해보겠습니다.


import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

a = tf.constant(5, name='input_a')
b = tf.constant(3, name='input_b')

c = tf.multiply(a, b, name='mul_c')
d = tf.add(a, b, name='add_d')
e = tf.add(c, d, name='add_e')

shape = tf.shape(a , name= 'a_shape')


print(shape)

sess=tf.Session()

print(sess.run(shape))
print(sess.run(e))

sess.run(c)
writer = tf.summary.FileWriter("c://tmp/my2_graph", sess.graph)


## tensorboard --logdir=c://tmp/my2_graph 라고 "아나콘다 프롬프트"에 입력


### 그래프 새롭게 생성하기

graph = tf.Graph()

with graph.as_default():
    in_1 = tf.placeholder(tf.float32, shape=[1], name="input_a")
    in_2 = tf.placeholder(tf.float32, shape=[1], name="input_b")
    const = tf.constant(3, dtype=tf.float32, name="static_value")
    with tf.name_scope("Transformation"):
        with tf.name_scope("A"):
            A_mul = tf.multiply(in_1, const)
            A_out = tf.subtract(A_mul, in_1)
        with tf.name_scope("B"):
            B_mul = tf.multiply(in_2, const)
            B_out = tf.subtract(B_mul, in_2)
        with tf.name_scope("C"):
            C_div = tf.divide(A_out, B_out)
            C_out = tf.add(C_div, const)
        with tf.name_scope("D"):
            D_div = tf.divide(B_out, A_out)
            D_out = tf.add(D_div, const)
        out = tf.maximum(C_out, D_out)

#그래프로 확인해 볼 것 - 값을 따라가면서 계산해보도록 유도할 것

writer = tf.summary.FileWriter('/tmp2/name_scope_2', graph=graph)
writer.close
