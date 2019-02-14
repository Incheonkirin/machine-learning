manifold

import os
os.chdir("C://Users/Hyungi/Desktop/workplace")
os.getcwd()


import matplotlib.pyplot as plot_tree
import seaborn as sns; sns.set()
import numpy as np

def make_hello(N=1000, rseed=42):
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('hello.png')
    plt.close(fig)

    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T
    print("이미지차원", data.shape)
    print(data)
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    # 4000 * 2 (188 * 72)
    i, j = (X * data.shape).astype(int).T

    mask = (data[i, j] < 1)
    X = X[mask]
    print("새로운X갯수", X.shape)
    print("원래이미지의 차수", data.shape)
    X[:,0] *= (data.shape[0] / data.shape[1])
    X = X[:N]

    return X[np.argsort(X[:, 0])]



X = make_hello(1000)
colorize = dict (c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis('equal');
plt.show()

#rainbow색으로 HELLO 가 출력되었죠? 이제, HELLO를 회전시켜보겠습니다.
#2차원 회전
print(X.shape)
def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],  #cos(theta)와 sin(theta)들로 이뤄진 rotate 행렬. (1000, 2)행렬이 됩니다.
    [-np.sin(theta), np.cos(theta)]] #R은 선형변환이다.
    return np.dot(X, R) #벡터 * 행렬이 되었죠? 행렬이 하는 역할이 뭐라고 했죠? (선형)변환 한다고 했지요. 앞에 있는 벡터를 변환한다고 했습니다.
    # 앞에 있는 벡터가 회전 변환을 하게 되면, 내적을 구해서.. 회전값이 나오겠죠?
# rotate함수에 X가 들어왔는데, X는 몇차원입니까? 2차원이죠. (1000, 2) 가 들어오고
# Degree Radian은 각도를 라디안으로 바꿔주는거죠? 컴퓨터에서 각도는 라디안을 사용하는 것.

# 원래행렬에 내적을 했더니, 20도 돌아갔다.

X2 = rotate(X, 20) + 5#회전값이 나오고, +5 만큼 띄우라는 말이에요.
plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis('equal');
plt.show()

#이걸 검증할 일은 없고, sin cos 을 사용하는 이유는 원을 표현하기 위함.
# sin, cos를 사용하면 회전한다고 알면 됩니다.


# 이거를 MDS를 해보겠습니다.
# Multi-dimensional scaling
# MDS는 원본 모양을 그대로 유지하고 있지요? 물론 여기서는 2차원 했으니 차원감소는 아니지만,
# MDS 적용해 보았더니 원래 모양 유지하고 있지요?
#STRESS 유사도
# 2차원으로 해보니, 원본 모양을 유지하는 것을 볼 수 있다.

from sklearn.metrics import pairwise_distances
D = pairwise_distances(X) #원래 1000, 1000 거리 행렬 : 상관행렬, 거리행렬 == 정방행렬 # 정방행렬이면서 대칭행렬
print(D.shape)#                                       전부1     전부0
D[:5, :5]
D2 = pairwise_distances(X2) #선형변환을 했기 때문에, 거리값은 그대로. D == D2

np.allclose(D, D2)
plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.show()
from sklearn.manifold import MDS
# STRESS 유사도
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out= model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal')
print(out)
plt.show()



#3차원으로 변환해보겠습니다
축을 중심으로 하고 변환하려면, 축이 직교해야 한다. (정직교해야함)
아래 그런데 C가 만들어지는 것은 랜덤한 축입니다.
3*3의 랜덤한 행렬을 만들었습니다.
그러니까 90도 축을 이룬다는 보장이 없습니다. 90도 축이 안되요.
그러니까 90도 축을 만들어야 하죠.
행렬 규칙 중에 하나가, 행렬을 거듭제곱하면, 대각요소만 제외하고 대칭인 행렬이 됩니다.
# 그 정방행렬이면서 대칭행렬인 행렬이 됩니다.          #자기자신과 전치한 것을 곱하면, 정방행렬& 대칭행렬이 됩니다.
# 그 행렬을 eigen분해 고유값분해하게 되면, 정직교하는 축을 생성. eigen vector와 value로 나누어지는데, eigen vector는 정직교
# X, Y 축이 정확히 90도가 아닌 상태에서 변환(정직교 아닌 상태에서) 하면 데이터가 찌그러집니다.

import numpy as np
rng = np.random.RandomState(10)
C = rng.randn(3, 3)
print(np.dot(C, C.T))

e, V = np.linalg.eigh(np.dot(C, C.T))
print("eigenvector", V)
print("eigenvalue", ?? )

def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension) # 3*3 의 랜덤한 행렬 90도 축이 안됨
    print("C는", C.shape)
    print(np.dot(C, C.T))
    e, V = np.linalg.eigh(np.dot(C, C.T)) # 정직교하는 축
    print("V는", V.shape)
    print("차원은", V[:X.shape[1]])
    return np.dot(X, V[:X.shape[1]]) # 2개의 축으로만 내적

print(X.shape)
print(X.shape[1])
print("데이터의 차원은",X.shape)
X3 = random_projection(X,3)
X3.shape


from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2],
        **colorize)
ax.view_init(azim=60, elev=30)
#            좌우각    상하각


\

# 다시 MDS로 2차원으로 축소해보겠습니다.
model = MDS(n_components = 2, random_state =1)
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal')
# 원래 데이터 거리값을 유지한 채 차원감소가 이루어진다.
# 선형데이터에서는 MDS가 아주 강하다.
에러 !




# 원래 데이터에 sin, cos를 이용해서 bending을 해보겠습니다.
#bending
def make_hello_s_curve(X): #사인코사인 원형
    t = (X[:, 0] -2) *0.75 *np.pi
    x = np.sin(t)
    y = X[:, 1] #y값은 그대로 사용합니다.
    z = np.sign(t) * (np.cos(t)-1) #Z값은 변화에 따라서 결정하기 위해서
    return np.vstack((x, y, z)).T

XS = make_hello_s_curve(X)
from mpl_toolkits import mplot3d
ax = plt.axes(projection = '3d')
ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2],
        **colorize);
plt.show()
#여기에 MDS를 하면 아까처럼 안 돌아간다는 말이지.


from sklearn.manifold import MDS
model = MDS(n_components = 2, random_state=2)
outS = model.fit_transform(XS)
plt.scatter(outS[:,0], outS[:,1], **colorize)
plt.axis('equal')
plt.show()
#원형을 되살리지 못합니다.



아까 원형에 비선형적 변환을 가한게 뭐라고 했지요?
# 비선형데이터에 강한 LLE. Localy Linear embedding

from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors=100, n_components=2, method='modified',
        eigen_solver='dense')
out = model.fit_transform(XS)

fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15)
plt.show()


# iris데이터 4차원 데이터를 mds를 이용해서 2차원과 3차원으로 시각화해보시오.
