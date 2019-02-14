import os
os.chdir("C://Users/Hyungi/Desktop/workplace")
os.getcwd()


# Kmeans :
# 거리척도 :
import numpy as np
import scipy
import sklearn.metrics.pairwise
#https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise


# np.array(함수, 매개변수 2개받음)는 ndarray 를 리턴합니다. 여기서 나오는 array는 뭐에요?
a_64 = np.array([61.22, 71.60, -65.755], dtype=np.float64)
b_64 = np.array([61.22, 71.608, -65.72], dtype=np.float64)
a_32 = a_64.astype(np.float32)
b_32 = b_64.astype(np.float32)

#거리값을 알 기 위해서는 np.linalgd의 norm을 사용해야합니다.
#                                norm은 언제 쓰나요? 벡터의 크기값을 말합니다.
dist_64_np = np.array([np.linalg.norm(a_64 - b_64)], dtype=np.float64)
dist_32_np = np.array([np.linalg.norm(a_32 - b_32)], dtype=np.float32)
dist_64_sklearn = sklearn.metrics.pairwise.pairwise_distances([a_64],[b_64])
dist_64_sklearn_l1 = sklearn.metrics.pairwise.pairwise_distances([a_64],[b_64], metric="l1") # l1 규제는 절대값입니다. l1을 넣은 것은 거리값을 '절대값''으로 내라는 말입니다.
dist_64_sklearn_l2 = sklearn.metrics.pairwise.pairwise_distances([a_64],[b_64], metric="l2") # l2 는? 제곱값으로 거리값을 재주라는 말입니다.
from sklearn.metrics.pairwise import euclidean_distances #여기서는 유클라디안디스턴스만 뽑아왔지요? 위에는 쓰지도 않는 것들을 모두 로딩하는데, 여기는 유.디 만 가져옵니다. 메모리를 줄여/속도를 빠르게 합니다.
print("euclidean_distance", euclidean_distances([a_64],[b_64])) #유클라디안 디스턴스를 구하는 공식이 뭐에요?
print(dist_64_np)



## 쓰레드 = 동시에 처리되는 단위.
import time, threading
from threading import Lock # 변수를 보호하는 Lock
class MyExample(threading.Thread): #매개변수가 아니고 ! 상속이다 !
    def __init__(self, name): # 생성자
        threading.Thread.__init__(self, name=name) #이렇게 호출하는 이유는 ?   상속을 받아서 초기화 할 때는, 부모의 초기화를 같이 해줘야 합니다.
        self.__suspend = False
        self.__exit = False
        self.lock = Lock()
    def run(self) :
        while True : #무한정 반복
            while self.__suspend:
                time.sleep(0.5)
            print(threading.currentThread().getName())
            time.sleep(0.2)
            if self.__exit:
                break
    def mySuspend(self): # 잠깐 중지
        self.__suspend = True
    def myResume(self): # 다시 시작
        self.__suspend = False
    def myExit(self): # 오버라이딩
        self._exit = True
lock = threading.Lock()
th = MyExample("thread_1")
th.start()
time.sleep(1)
th.mySuspend()
with lock:
    print('Suspend Thread.....')
time.sleep(1)
th.myResume()
with.lock:
    print('Resume Thread.....')
time.sleep(1)
th.myExit()



Thread라고 하는 건.. 컴퓨터는 속도가 빨라요. 그래서 컴퓨터는 시분할로 처리해요. 너무 빠르기 때문에. 데이터베이스를 하게 되어지면, 시분할 DB-> 8~30개로 돌아가요
#데이터를 처리하는 회로
그럼 Thread를 어떻게 처리하느냐. 데이터가 우선 주기억장치에 저장되어지고, (속도가 빠른)레지스터로 올라갑니다.
여기서 부동소수점 연산기 가 필요해요. 이게 누산기에 저장되어서 나와요.

Thread라고 하는건,
Thread   Thread
사용하고,   사용하고 하는데.

레지스터를 저장해야해. 그것을 swapping 한다고 이야기 한다.


단일 프로그램인데, 여러개의 Thread가 돌아가는거야.
변수를 건드리는 거야.

'threading'

변수+10*100...

'Lock 개념' 이 변수를 끝까지 처리하기 전까지는 이 변수를 건드리지 말아라.



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
X = np.array([[7,5], [5,7], [7,7], [4,4], [4,6], [1,4],
            [0,0], [2,2], [8,7], [6,8], [5,5], [3,7]])
plt.scatter(X[:,0], X[:,1], s=100)
plt.show()
from sklearn.cluster import KMeans
# 초기값 중심값을 지정, 중심값을 몇번이나 변경하면서 작업하겠는가.
model = KMeans(n_clusters=2, init="random", n_init=1, max_iter=1, #max_iter는 전체
        random_state=1).fit(X)
c0, c1 = model.cluster_centers_
print(c0, c1)
print(model.labels_)
print(len(model.labels_))
print(model.inertia_) # 각 포인트가 중심값에서 얼마나 떨어져 있느냐
plt.scatter(X[model.labels_==0,0], X[model.labels_==0,1], s=100, marker='v', c='r') #s는 사이즈를 의미
plt.scatter(X[model.labels_==1,0], X[model.labels_==1,1], s=100, marker='^', c='b')
plt.scatter(c0[0], c0[1], s=100, c='r')
plt.scatter(c1[0], c1[1], s=100, c='b')
plt.show()



import numpy as np
a = np.array([1., 2., 3., 4.])
print(a)
print(a.shape)
a_4_1  = a[:, np.newaxis]
print(a_4_1) # 2차원
print(a_4_1.shape)
a = np.array([1,1])
b = np.array([2,2])
np.linalg.norm(a -b) #피타고라스 정리값이 출력
# 왜 1.4142.. 나와요? 루트씌운거지. norm 하면 뭐한거다? 피타고라스를 한거다.



def kmeans_df(c0, c1):
# hstack 이 뭐라고 했지요? R로 치면, cbind 컬럼바인드.
    df = pd.DataFrame(np.hstack([X,
# X에는 무슨 데이터가 들어있나요?
    np.linalg.norm(X - c0, axis=1)[:, np.newaxis],
# X에서 c0까지의 거리값            행이니까 열로 바꾸기위해서
    np.linalg.norm(X - c1, axis=1)[:, np.newaxis],
    model.labels_[:, np.newaxis]]),
    columns = ['x0', 'x1', 'd0', 'd1', 'c'])
    return df

kmeans_df(c0, c1)


print(X[model.labels_==0,0].mean(),X[model.labels_==0,1].mean())
print(X[model.labels_==1,0].mean(),X[model.labels_==1,1].mean())
model.score(X)
c0_score = (np.linalg.norm(X[model.labels_==0] - c0, axis=1) **2).sum()
c1_score = (np.linalg.norm(X[model.labels_==1] - c1, axis=1) **2).sum()
c0_score +c1_score

c0 #중심값





from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)# fit하면 중심값, 훈련데이터에 대한 레이블이 만들어지고.   predict을 하게되면, 이미 들어온 데이터가 어느 군집에 속하는지.
    ax = ax or plt.gca() #plot에는 두개있다고 했지. figure(전체셋팅)하는 것과  axes(도화지)
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)  #zorder는, 포토샵에서보면 layer(이미지에서 레이어는 투명용지) . 밑바닥이 보여. zorder=2 는 도화지를 2장마련했다.

    centers = kmeans.cluster_centers_
    #반지름을 구하고 있습니다.

    radii = [cdist(X[labels == i], [center]). max() # 두개의 값을 받아 거리값을 재주는 cdist. 왜 norm을 안쓰고 cdist를 쓰나요? 차이점이 뭐에요? norm이라고 쓸때는 항상 원점을 가정(하나의 데이터를 가지고 재 주는것). cdist는 두 점에 대한 거리를 말합니다.
     # 그런데 kmeans에서 거리값은 무슨 의미가 있어요? 반지름.   원형.kmeans를 말하면 원형을 추상해야해요.
     # max 제일 거리값이 먼 것을 찾아요


        for i , center in enumerate(centers)]

    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5,
        zorder=1))
#add_patch는 이 이미지를 레이어에 더해주는데 zorder=1에 넣어주어라. 이미지를 합칠 때 씁니다.


kmeans= KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)
plt.show()

# kmeans의 값들을 어떻게 활용할 것인가. kmeans를 봤을 때 원형으로 생각하는 습관!!




X = np.loadtxt('data_quality.txt', delimiter=',')


plt.figure()
plt.scatter(X[:,0], X[:,1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X[:, 0].min() -1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() -1, X[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
#\화면에 나와야 할 거리값을 재주기 위해서 min max를 재 주었다.

plt.xticks(())
plt.yticks(())
plt.show()

scores = []
values = np.arange(2,10) #군집수
from sklearn import metrics
for num_clusters in values:
    #k-means : random보다는 효율적으로 중심을 잡아라
    #max iter가 안들어와 있으면, 300번 => 결과는 가장 좋은 값을 돌려줍니다. # 중심값을 10번 잡고
    kmeans = KMeans(init = 'k-means++', n_clusters=num_clusters, n_init=10) #init를 10번 하라
    kmeans.fit(X)
    score = metrics.silhouette_score(X, kmeans.labels_,
#                  식이 어떻게 된다고? max a, b 에 b-ab 이때 a는? 내부 군집의 평균거리.  그리고 b값은? 한 군집과 제일 가까운 군집의 거리값
#                 클러스터 개수를 모르고 있을 때 군집분석의 평가 지표는 silhouette_score이다 !

    metric='euclidean', sample_size=len(X))
# euclidean 뿐만이 아니라, metrics.pairwise.pairwise_distances에 있는 어느 값이라도 줄 수 있다.

    print("\n클러스터의 수=", num_clusters)
    print("실루엣지수 = ", score)
    scores.append(score)



plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('실루엣지수와 클러스터개수')
plt.show()

num_clusters = np.argmax(scores) + values[0]
print('\n최적의 클러스터 수=', num_clusters)




## 3차원 시각화 => kmeans cluster는 축을 줄여주지 못한다. 열변수 4개.  그래서 3개만 찍었죠 ax.scatter(X[:, 3], X[:, 0], X[:, 2]
# md차원변환

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
np.random.seed(5)
centers = [[1, 1], [-1, 1], [1, -1]]
iris = datasets.load_iris() #열 변수 4개
X = iris.data

y = iris.target
estimators = {'_3': KMeans(n_clusters=3),
            '_8':KMeans(n_clusters=8)}
fignum = 1
for name, est in estimators.items():
    fig = plt.figure(fignum)
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # elev는 상하각, azim은 좌우각입니다.
    plt.cla()
    est.fit(X)
    labels = est.labels_
    #kmeans 데이터를 3차원으로 찍을때는, '주축'을 결정해야 합니다.
    # 하지만, 차원축소가 되지 않아서, '제외' 시키는 방법을로 해야합니다.
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float), s=100)
    fignum = fignum + 1
plt.show()




from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
digits = load_digits() #숫자정보 (0~9까지 숫자 정보)  #원래 load_digits은 지도학습인데, 여기서는 kmeans로 해보는 겁니다.
data = scale(digits.data)

data = scale(digits.data)
def print_digits(images, labels):
    f = plt.figure(figsize=(10, 2)) #출력 사이즈를  결정

    plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
    i = 0
    while (i < 10 and i < images.shape[0]):
        ax = f.add_subplot(1, 10, i + 1) # 1행 10열, 1번부터 10번까지 이미지를 찍고 있습니다.
        ax.imshow(images[i], cmap=plt.cm.bone)
        ax.grid(False)
        ax.set_title(labels[i])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
        i += 1

print_digits(digits.images, range(10))

plt.show()





#학습을 시키겠습니다.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, images_train, images_test =  train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)
from sklearn.cluster import KMeans
clf = KMeans(init="k-means++", n_clusters=10, random_state=42)
clf.fit(X_train)
print_digits(images_train, clf.labels_)
plt.show()
#잘 못맞추고있네요


#모델에 예측까지 해보겠습니다.
y_pred = clf.predict(X_test)
def print_cluster(images, y_pred, cluster_number):
    images = images[y_pred == cluster_number]
    y_pred = y_pred[y_pred == cluster_number]
    print_digits(images, y_pred)
for i in range(10):
    print_cluster(images_test, y_pred, i)
# number조심 : label링은 수작업으로 표현할 필요가 있다.

plt.show()



#여기(지도학습)서 왜 비지도학습을 하나요?
# 패턴을 찾기위해서 입니다. kmeans가 가지고 있는 논리가 잘 들어가는지 확인.  원형으로 분류되어진 그룹에 잘 먹힌다는 것입니다.





#벡터양자화
import matplotlib.pyplot as plt
from sklearn import cluster
image = plt.imread("donald.jpg")

plt.figure(figsize = (15, 8))
plt.imshow(image)
image.shape #행 ,열 , 컬러값
# 전체 이미지 사이즈를 계산하려면 어떻게 해야하나요?
image.shape[0] * image.shape[1] * image.shape[2]

x, y, z = image.shape
image_2d = image.reshape(x*y, z) # x,y를 없애서 , 2차원으로 바꿔줍니다. 왜 바꿨느냐 =>
image_2d.shape
# 컬러값 RGB
# 벡터 양자화 == 압축한다.
kmeans_cluster = cluster.KMeans(n_clusters=255)   # 컬러값을 16개로 바꾼다는 말이죠?
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_ #16개의 컬러값이 되겠지요?
cluster_centers #대표 컬러값이 되어짐
cluster_centers.shape[0] * cluster_centers.shape[1] * 8
cluster_labels = kmeans_cluster.labels_ #컬러값에 대한 (군집에따라서) 레이블
cluster_labels
len(cluster_labels)
plt.figure(figsize=(15,8))
plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z))
# 0 ~255 컬러 : unsigned int는 -127부터 + 128까지 표현이 되어진다. => 0 ~255
# 컬러값은 0~1 정규화된 값으로 출력할 수 있고, 0~255 칼라값으로 출력할 수 있음.
plt.imshow((cluster_centers *255).astype(np.uint8)[cluster_labels].reshape(x, y, z)) #0부터 255까지 컬러값을..
#                                                 cluster labels는 무엇만큼있어야 정상입니까? 데이터갯수만큼 있어야 정상입니다.
#                                      16개        데이터만큼
#          ------------------------------------------------------- 하면 데이터 갯수만큼 만들어집니다.
len(cluster_centers[cluster_labels])
# plt.imshow((out * 255).astype(np.uint8))
plt.show()

# cluster_centers와 cluster_labels 만 다른 사람에게 보내면 된다. 그렇게 만든게 GIF
# 컬러팔레트 - 헤더

# GIF 압축법


#지금까지는 사각이상치를 제거했습니다.


#원형이상치 제거
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X, label = make_blobs(100, centers = 1)
import numpy as np
kmeans = KMeans(n_clusters=1)
kmeans.fit(X)
f, ax = plt.subplots(figsize=(7, 5))
ax.set_title('원형 이상치 제거')
ax.scatter(X[:, 0], X[:, 1], label='Points')
ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], label='Centroid', color='r')

ax.legend(loc='best')
#이상치 제거 부분

# ravel : 데이터를 1차원으로  -1 : 역순으로
distances = kmeans.transform(X)
sorted_idx  = np.argsort(distances.ravel())[::-1][:5]
f, ax = plt.subplots(figsize=(7, 5))
ax.set_title('원형 이상치 제거')
ax.scatter(X[:, 0], X[:, 1], label='Points')
ax.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], label='Centroid',
            color='r')

ax.scatter(X[sorted_idx][:, 0],
            X[sorted_idx][:, 1],
            label='이상치', edgecolors='g',
            facecolors='none', s=100)
ax.legend(loc='best')



## 다음은 여러가지 군집을 비교분석해보겠습니다.
##군집분석 모델 비교
from sklearn.metrics.cluster import silhouette_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler # (관측치 - 평균)/ 표준편차
from sklearn.cluster import KMeans # KMeans하면 뭐가 생각나야 해요? 중심점, 클러스터개수, 그리고 원형
from sklearn.cluster import AgglomerativeClustering # 이건 뭐라고 했지요? 병합군집. 병합군집은 모든 요소가 군집이 되죠? 그래서 가까운 군집끼리 묶어갑니다. 언제까지? 주어진 클러스터까지.

#그래서 DBSCAN은 항상 핵심, 경계, 잡음 (기본적으로 이상치를 제거하는 효과가 있습니다.)으로 구분합니다.
from sklearn.cluster import DBSCAN # eps 거리값하고, min_samples (군집의 최소 요소수) 에 따라서 이웃하는 것끼리 묶어가는 것 입니다.
import mglearn #출력이 잘 될 수 있도록 만든 패키지입니다.

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler = StandardScaler() #이것을 정규화해서..        StandardScaler는 클래스이죠? 인스턴스 되어가지고, 아래 fit을 하면 데이터가 구해지죠? 그럼 뭐가 들어와야해요? 이 데이터의 평균하고 표준편차를 구해줘야죠. 그래야 뭘 할 수 있어요? 그래야 정규화를 할 수 있게 되지. 그게 transform이란 말이에요. 그 역할을 뭐가 해주는거에요? 그걸 함수가 해주는거에요. 아예 기본함수로 다 만들어 놓은 거에요.  처음에는 복잡해보이죠. 그런데 다 만들어져 있으니까 나중가면 편해지죠.
scaler.fit(X) #fit을 통해서 평균과 표준편차를 구하고 아래 transform에서 찍으면 되요.
X_scaled = scaler.transform(X)
fig, axes = plt.subplots(1, 4, figsize=(15, 3),
                        subplot_kw={'xticks': (), 'yticks':()})
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
        cmap=mglearn.cm3, s=60, edgecolors='black')

axes[0].set_title("무작위 할당: {:.2f}".format(
#찍고나서 실루엣 스코어로..
        silhouette_score(X_scaled, random_clusters)))
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),#2개 그룹이 남을 때까지 계속 모아갑니다.
        DBSCAN()]


for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3,
            s= 60, edgecolors='black')
    ax.set_title("{} :{:.2f}".format(algorithm.__class__.__name__,
            silhouette_score(X_scaled, clusters)))
plt.show()



## DBSCAN을 조금 더 정확하게

from sklearn.cluster import DBSCAN
X, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN()
clusters  = dbscan.fit_predict(X)
print("클러스터 레이블:\n{}".format(clusters))

mglearn.plots.plot_dbscan()
plt.show()
#eps값이 뭐라고 했지요? 거리값.
# min samples와 eps를 잘 조정해야..



#병합군집법 하고 가겠습니다.
import mglearn
from sklearn.cluster import AgglomerativeClustering
X, y = make_blobs(random_state= 1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)

plt.legend(["클러스터 0", "클러스터 1", "클러스터 2"], loc="best")
plt.xlabel("특성 0")
plt.ylabel("특성 1")

plt.show()





#한글 깨질때
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)
#이걸 위에쓰고 출력하세요



## GaussianMixture - 원형으로 범위를 만듭니다. 타원을 결합 - 음성 분리할 때 많이 썼었음.
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,
        cluster_std=0.60, random_state=0)

X = X[:, ::-1]

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()



# tslearn : sensor data = 항상 불규칙
# GlobalAlignment 시계열 데이터를 범위값 비교 # DTW ( Dynamic Time Wraping)
# sigma 표준편차.
# kmeans + DTW + timeseries 결합해서.

from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.metrics import sigma_gak, cdist_gak
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# tslearn:
seed = 0                                 #물리적데이터를 직접 넣을 수 없으니까, '양자화'하여서 촘촘하게하면 정밀=양이 많아짐. 듬성듬성=양 줄어듬,정확도 떨어짐.
np.random.seed(seed)                     #타임시리즈 데이터, 이산적데이터 ..
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = X_train[y_train <4]
np.random.shuffle(X_train)

#          TimeSeriesScalerMeanVariance의 매개변수가 mu(평균)와 std(표준편차) => 정규분포 정규화.
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])
sz = X_train.shape[1]
gak_km = GlobalAlignmentKernelKMeans(n_clusters=3, sigma=sigma_gak(X_train),    #왜 GlobalAlignmentKernelKMeans를 붙였을까요? 시계열 데이터를 범윅값을 가지고 전 범위 비교. 시계열 데이터를 잴때 정확히 재려면 뭐를 쓴다고 했죠? DTW(Dynamic Time Wraping)을 써서 거리값을 재서 쓴다고 했죠?
        n_init=20, verbose=True, random_state=seed)     #매개변술로 sigma가 들어가고 있어요. 왜 시그마가 들어갔죠? 시그마는 표준편차. X_train이 가지고있는 sigma gak
        #n_init는 클러스터의 중심값을 몇번 바꾸냐는거에요. verbose는 조용히 하라고 하는 것이다.
y_pred = gak_km.fit_predict(X_train)
plt.figure()
for yi in range(3):
    plt.subplot(3, 1, 1+ yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()

plt.show()
