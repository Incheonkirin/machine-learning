import os
os.chdir("C://Users/Hyungi/Desktop/workplace")
os.getcwd()

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

mglearn.plots.plot_scaling()
# original Data 는 원래 데이터, StandaradScaler하면, 마이너스 값도 존재하게 된다. RobustScaler(견고하게), MinMaxScaler, Normalizer.
plt.show()



import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
X
pca = PCA(n_components = 2) #n_components 2개 => 데이터가 6개인데, 주성분수 2개
pca.fit(X) #내부에서 상관계수 구하고, 상관계수 구한데다가 고유값분해하고. 고유값분해 하면, 성분을 2개만 취하라.
# 크기 1.0
print(pca.explained_variance_ratio_) #분산에 대해서 얼마나 설명을 해주느냐.
#주성분 2개에 대해서, 두개의 값이 나옴. [0.99244289, 0.00755711]

print(pca.mean_)
print(pca.n_components_)
print(pca.noise_variance_)



#고유값분해 => 여기서는 SVD로 하고있어요. 특이값 분해 Singular Value Decomposition
pca = PCA(n_components=2, svd_solver='full') # full 방식이 있고, arpack 방식이 있고, randomize방식이 있다.
# full 방식은 모두

# arpack방식
# 분해되어진 요소 중에, truncated
# truncated : 0을 제거해서 희소행렬을 밀집행렬로 변환해서 사용합니다. 쓸데없는 것을 제거하여 계산속도를 빠르게.
# SVD는 비정방행렬에 가능하고, 고유값분해는 정방행렬만 가능합니다.
# SVD가 더 넓은 개념이기 때문에, SVD로 고유값분해를 할 수 있습니다.
# SVD로 하는 대부분의 것들이 희소행렬입니다. 희소행렬은 언제 생기나요? 연관분석의 행렬.. 그때는 정방행렬일 수가 없어요. 상품행렬은 비정방행렬일 수 밖에 없어요. 행과 열 수 가 달라요. 다를땐 SVM으로 하는데, scikit은 특이값 분해를 통해서 하도록 되어있고.

# randomize는 자기 멋대로.. 500 * 500 . 즉 변수가 많은 경우에는 randomize를 사용하는게 더 낫다고 한다. randomize는 변수를 자기가 임의로 정해서 한다.

pca.fit(X)
print(pca.explained_variance_ratio_)



from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
print(cancer.feature_names) #변수이름들을 출력할 수 있다
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=1)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("스케일 조정 후 특성별 최소값:\n{}".format(X_test_scaled.min(axis=0)))
print("스케일 조정 후 특성별 최대값:\n{}".format(X_test_scaled.max(axis=0)))

from sklearn.svm import SVC  # Support vector classifier 서포트 벡터 분류기
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("테스트 세트 정확도: {:.2f}".format(svm.score(X_test, y_test)))
svm.fit(X_train_scaled, y_train)
print("스케일 조정된 테스트 세트의 정확도: {:.2f}".format(
        svm.score(X_test_scaled, y_test)))

# minmaxscaler하면 값이 0에서 1사이로 나옵니다.
# standardscaler로 한번 바꿔보세요. 똑같죠?


# scaled 한 다음에 PCA를 해보겠습니다.

from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
scaler = StandardScaler() #전처리 모델
scaler.fit(cancer.data) # 값을 구하는 과정 : 평균/ 표준편차
X_scaled = scaler.transform(cancer.data) #일반 머신러닝에서는 predict를 하는데, 여기서는 transform을 하지요
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  #원래 30개를 2개로 표현합니다. 가장 중요한 성분 2개를 출력해줍니다. 가장 주성분-정렬
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print(pca.explained_variance_ratio_)
print("원본 데이터 형태: {}".format(str(X_scaled.shape)))
print("축소된 데이터 형태: {}".format(str(X_pca.shape)))


import matplotlib.pylab as plt
plt.plot(pca.explained_variance_ratio_, 'bo-')


import mglearn
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter    (X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(["악성", "양성"], loc="best")
plt.gca().set_aspect('equal')
plt.xlabel("첫 번째 주성분")
plt.ylabel("두 번째 주성분")
plt.show()


#이번에는 30개를 다 해보겠습니다.

#주성분 기여도 시각화
from sklearn.decomposition import PCA
pca = PCA(n_components=30)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

#이번에는 누적으로 안찍고 그대로 찍어보겠습니다.
plt.plot(pca.explained_variance_ratio_, 'bo-')
plt.show()



#문제 : X_train을 scaler하고 PCA한 후 SVC한 모델에 테스트데이터를 적용했을때 정확도가 얼마인지 출력하시오.
#9개 94%,  5~8개 92%,  4개 90%,  20개 95% ,30개 97%

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.decomposition import PCA
# pca = PCA(n_components=20)
pca = PCA(n_components=6)
pca.fit(X_train_scaled)

X_t_train = pca.transform(X_train_scaled)
X_t_test = pca.transform(X_test_scaled)

svm.fit(X_t_train, y_train)
print("SVM 테스트 정확도: {:.2f}".format(svm.score(X_t_test, y_test)))


#penalty : 규제 cost function 변수가 30개 주성분 6
print("PCA 주성분 형태 : {}".format(pca.components_.shape))
print("PCA 주성분 형태 : {}".format(pca.components_))



#matshow에 대한 설명
import matplotlib.pyplot as plt
import numpy as np
def samplemat(dims):
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = 1
    return aa
plt.matshow(samplemat((15, 15)))
plt.show()
plt.matshow(pca.components_, cmap='viridis')
plt.show()


from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)


# print("PCA 주성분 형태 : {}".format(pca.components_.shape))
# print("PCA 주성분 형태 : {}".format(pca.components_))
#위와 같이 표현하니까 잘 모르겠죠? 그래프로 찍어보겠습니다.
#
plt.yticks([0, 1, 2, 3, 4, 5, 6], ["첫 번째 주성분", "두 번째 주성분", "세 번째 주성분",
        "네번째 주성분", "다섯번째 주성분", "여섯번째 주성분"])

plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
        cancer.feature_names, rotation=60, ha='left')
plt.xlabel('특성')
plt.ylabel('주성분')
plt.show()
# 매트릭스에 있는 숫자를 이미지화 했다고 생각하면 됩니다.



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA # matlab
def biplot(score, coeff, labels = None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min()) # min-max scale로 보고
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs* scalex, ys* scaley)
    for i in range(n): # arrow 화살표를 그리는 함수
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color= 'r', alpha= 0.5)
        if labels is None:
            plt.text(coeff[i, 0]*1.15, coeff[i, 1] * 1.15, "Var"+str(i+1),
            color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0]* 1.15, coeff[i, 1]* 1.15, labels[i],
            color='g', ha='center', va='center')

pca_mlab = PCA(X_train_scaled)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

#          Y는 원래 데이터값,  Wt 주성분축에 대한 기여도 : 행렬이 변경되어 있음. -전치해서 여기에 넣어야함.
biplot(pca_mlab.Y[:,0:2], pca_mlab.Wt[:, 0:2])
plt.show()

print(dir(pca))


matlab을 따라서 만든, mlab



# 문제 :  PCA
#          원래데이터
X_train_t = X_train.transpose() #전치
R_train = pca.components_.transpose()

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()
biplot(X_train_t[:, 0:2], R_train[:, 0:2]) # scaled 된 데이터를 복원해서 출력
                                           # 위에서    scalex = 1.0/(xs.max() - xs.min()) # min-max scale
                                           #           plt.scatter(xs* scalex, ys* scaley)   이렇게 scaled해 주었었다.

plt.show()




from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
# 이미지 people.images, people.target
image_shape = people.images[0].shape
print(image_shape) # byte color 흑백 0~255
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
        subplot_kw={'xticks':(), 'yticks':()})

# 2차원을 1차원으로
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
# 87 * 65
print("이미지사이즈 : {}".format(people.images.shape))
print("클래스 개수: {}".format(len(people.target_names)))
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)): #이름뒤에 나오는 숫자는 그 사람사진이 몇장인지.
    print("{0:25} {1:3}".format(name, count), end=' ')
    if (i+1) %3 == 0:
        print()



#boolean 인덱싱
# 타겟 수 만큼 부울린 인덱싱을 하기 위해서, 메모리 공간을 확보합니다
mask = np.zeros(people.target.shape, dtype = np.bool)
for target in np.unique(people.target): #unique 니까, 중복된 이름은 다 제거하고, 유일한 이름만 남김.
    mask[np.where(people.target == target)[0][:50]] = 1 #여러번 반복되는 것들을 50개까지만, (이미지 갯수를 통일하기위해서, 50개까지만) 1로 함. 50번 이상은 0.
X_people = people.data[mask]
y_people = people.target[mask]

#칼라값을 정규화하기위해서 (minmax를 하는게 아니고) 255로 나누면 됨. 정규화하는 이유는 뭐야? 분류율을 높이기 위해서.
X_people = X_people / 255.



from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(
        X_people, y_people, stratify= y_people, random_state=0)
# stratify 층화 : y 값의 분포를 고려해서 뽑아라

# neighbors 1, 2 로 한번 변경해보세요
knn = KNeighborsClassifier(n_neighbors =2)
knn.fit(X_train, y_train)
print("1-최근접 이웃의 테스트 세트 점수: {:.2f}".format(
        knn.score(X_test, y_test)))

#이런식으로 이미지를 처리한다는 걸 알아두세요. 딥러닝 가야지, 정확해집니다.
plt.show()


print(people.images[0].shape[0] * people.images[0].shape[1])


# PCA를 하면 좀 더 정확해집니다.
from sklearn.decomposition import PCA
#원래 주성분 components가 몇 개 있어야 정상이죠? print(people.images[0].shape[0] * people.images[0].shape[1]) 5655가 나오죠
# whiten = 정규화하라. 분포를 고려해서.
pca = PCA(n_components = 100, whiten = True, random_state= 0 ).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


print("훈련차원 : {}".format(X_train_pca.shape))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("테스트 세트 정확도: {:.2f}".format(knn.score(X_test_pca, y_test)))



# components는 주성분에 대한 각 변수들의 기여도입니다.
print("주성분과 특성 shape: {}".format(pca.components_.shape))
# 주성분이 원래 5655가 나와야지 정상이에요. 그 중에 100개를 본 것입니다.
# components_가 뭐에요? 주성분의 특성을 뽑은거죠? 주성분에 대한 특성 추출

fig, axes = plt.subplots(3, 5, figsize=(15, 12),
        subplot_kw ={'xticks':(), 'yticks':()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title('주성분 {}'.format(( i + 1)))
plt.show() #주성분으로 찾은 특성들이 출력되죠?



# 5천차원의 데이터를, 10개 50개 100개 500개의 주성분으로 fit.
# fit하면 뭐가 만들어지나요? 고유값분해를 해야하는데, 여기서는 뭘 쓴다고했죠? SVD를 쓴다고했죠. SVD해서 나오는게 뭐에요?
# 직교축하고 Value가 나오겠죠?
Transform해주고 있어요. Transform하면 뭐가 되요?  PCA를 하면 직교축이 만들어지고, X_test에는 뭐가들어있어요? test할 사진들이 들어있죠.
Transform하면 뭐죠? X_test를  eigen이 가지고 있는 주성분 축에 맞춰서 변환시켜라.
그러면 이 데이터가 어디로 가겠어요? 축에 맞춘 이미지로 가서, 전혀 다른 이미지가 되겠지요?
전혀 새로운 축으로 변환된 이미지가 만들어집니다.
transform 하면 5천 차원 -> 10장으로 변환.
inverse_transform를 하게 되면 10장으로 만들어진 것을 원래 이미지로 바꿔주는 것. 10장으로 구성된 주성분을 합쳐서 다시 이미지를 만드는것입니다.


#이미지 복원
reduced_images = []
for n_components in [10, 50, 100, 500]: # 주성분 갯수: 10개 50개 100개 500개.
    pca = PCA(n_components=n_components) #
    pca.fit(X_train) #훈련시킴 eigen => SVD => 직교 축, value
    X_test_pca = pca.transform(X_test) #transform 함 #새로운 축으로 변환 된 이미지
    X_test_back = pca.inverse_transform(X_test_pca) ## 이미지 복원
    reduced_images.append(X_test_back) #X_test_back을 리스트에 하나하나 넣어서 더해줌.
reduced_images

fix, axes = plt.subplots(3, 5, figsize=(15, 12),
        subplot_kw={'xticks':(), 'yticks':()})
for i, ax in enumerate(axes):
    ax[0].imshow(X_test[i].reshape(image_shape), vmin=0, vmax=1) #vmin vmax로 바꿔서 출력해라.
    for a, X_test_back in zip(ax[1:], reduced_images): #reduced images에는 원본으로 변환된 이미지들이 들어있습니다.
        a.imshow(X_test_back[i].reshape(image_shape), vmin=0, vmax=1)


axes[0, 0].set_title("원래 이미지")
for ax, n_components in zip(axes[0, 1:], [10, 50, 100, 500]):
    ax.set_title("%d 개 주성분으로 복원" %n_components)


plt.show()



# PCA의 문제점은 + - 를 상쇄해버린다는 점입니다. 각 값에 있는  + 와 - 를 상쇄..


# NMF는 음수 미포함. 음수가 아닌 값들의 갯수를 추출해서 .. # 특징 추출방법이 상이합니다. 그림이 많이 다르죠.
#언제 씁니까? 비음수데이터. 비음수데이터가 어떤것이 있죠? 소리데이터, 유전자데이터, 텍스트데이터 는 음수가 없습니다. 때문에 이것들은 NMF를 사용합니다.
#

# 이 것을 NMF로 바꿔보겠습니다.
# Non-Negative matrix Factorization
## NMF : 음수 미포함 행렬 분해
#여기서 선생님은 요약해서 설명하기 때문에, 꼭 스스로 책과 document를 찾아가봐며 공부해야 합니다.
# 공부할 때 공부해야지, 나중에 다 끝나고 하려고 하면, 복기하는데 오래걸립니다.
# '감'이 왔을때 해야지 잘 맞아 떨어집니다.
from sklearn.decomposition import NMF
nmf = NMF(n_components =15, random_state=0)  # 15개 component를 찍음.
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

fig, axes = plt.subplots(3, 5, figsize=(15,12),
        subplot_kw={'xticks': (), 'yticks':()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("성분 {}".format(i))

plt.show()

###


# https://blog.naver.com/ssdyka/221270089810
import mglearn
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.

# 테스트/훈련세트 나누기
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
image_shape = people.images[0].shape
mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
plt.show()

# 결과는 PCA를 사용했을 때와 비슷하지만 품질이 떨어진다
# PCA가 재구성 측면에서 최선의 방향을 찾았기 때문이다.
#
# NMF는 데이터를 인코딩하거나 재구성하는 용도로 사용하기 보다는 주로 데이터에 있는 유용한 패턴을 찾는데 활용한다.



# Signal 데이터를 만들어서

import mglearn
S = mglearn.datasets.make_signals()

plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel('시간')
plt.ylabel('신호')
plt.margins(0)
A = np.random.RandomState(0).uniform(size=(100, 3)) # 100 by 3 Matrix를 만들어서, 가산해주었습니다.  잡음을 섞었습니다.
# 2000, 3   3, 100  => 2000, 100
X = np.dot(S, A.T)
print("측정 데이터 형태: {}".format(X.shape))
from sklearn.decomposition import NMF
nmf = NMF(n_components = 3, random_state = 42)
S_ = nmf.fit_transform(X)
print("복원한 신호 데이터 형태: {}".format(S_.shape))
pca = PCA(n_components = 3)
H = pca.fit_transform(X)
models = [X, S, S_, H]
names = ['측정 신호 (처음 3개)', '원본 신호', 'NMF로 복원한 신호',
        'PCA로 복원한 신호']

fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5},
        subplot_kw={'xticks': (), 'yticks': ()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')
    ax.margins(0)



plt.show()
#첫번째 이미지는 원본이미지에 노이즈 더해서 겹쳐짐.
# 측정신호 는 만들어져서 다른 곳으로 가면, 노이즈가 더해짐.

# 원본신호 -> 측정신호 -> NMF로 복원한 신호. PCA로 복원한 신호
# NMF로 복원한 신호      원본신호와 비슷하게 분류해 냄
# PCA로 복원한 신호      음성신호에는 적합하지 않음.


# non-negative matrix factorization
#
# 유용한 특성을 뽑아내기 위한 또다른 비지도 학습 알고리즘.
# 이 알고리즘은 PCA와 비슷하고 차원축소에서도 사용할 수 있다.
#
# PCA에서는 데이터의 분산이 가장 크고 수직인 성분을 찾았다면
# NMF에서는 음수가 아닌 성분과 계수값을 찾는다.
# 즉, 주성분과 계수가 모두 0보다 크거나 같아야 한다.
#
# 여러 사람의 목소리가 담긴 오디오, 여러 악기로 이루어진 음악에서 유용하다.



import matplotlib.pyplot as plt

import mglearn

mglearn.plots.plot_nmf_illustration()

# plt.show() 오래걸림
# PCA를 사용할 떄와는 달리 NMF로 데이터를 다루기위해서는 모든 데이터가 양수여야 한다.
# 이 뜻은 데이터가 원점(0,0)에 상대적으로 어디에 놓여있는지가 중요하다는 뜻이다.
#
#
# 위 그림에서 왼쪽은 성분이 2개인 NMF로 표현하였고,
# 오른쪽 그림은 1개의 성분만을 사용해서 데이터를 가장 잘 표현할 수 있는 평균으로 향하는 성분을 만든다.
#
# PCA와는 달리 성분 개수를 줄이면 특정 방향이 제거되는 것 뿐만아니라 전체 성분히 완전히 바뀐다.
#
# NMF는 무작위로 초기화 하기 때문에 난수 생성 초기값에 따라 결과가 달라진다.



from sklearn.datasets import load_digits
digits = load_digits()
fig, axes = plt.subplots(2, 5, figsize=(10, 5),
        subplot_kw={'xticks':(), 'yticks':()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)
ca = PCA(n_components= 2)
pca.fit(digits.data)

digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):

    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
    color = colors[digits.target[i]],
    fontdict={'weight' : 'bold', 'size' :9})

plt.xlabels("첫번째 주성분")
plt.ylabels("두번째 주성분")
#손글씨를 두개의 주성분을 통해서 점을 찍어 보았습니다 => 올바르게 분류를 안해줍니다.

# PCA만 잘 이해해도 데이터 분석하는 데 많은 도움이 됩니다.
# PCA는 넓은 범위에 영향을 미칩니다.



# 가까운 것은 더 가깝게, 먼 거리는 더 멀게 해서 변환된 결과값으로 원본으로 복원하는 것이 불가능한 변환이지만,
# 확연하게 분류( 역변환이 불가능 )
# k-means에서 몇개의 그룹으로 나누어야 하는가를 알 때, 이 것을 시각적으로 확인할때
from sklearn.manifold import TSNE
tsne = TSNE(random_state= 42)
digits_tsne = tsne.fit_transform(digits.data)
plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
    color = colors[digits.target[i]],
    fontdict={'weight': 'bold', 'size':9})
plt.xlabel("t-SNE 특성 0")
plt.ylabel("t-SNE 특성 1")




#문제 : PCA와 SVD를 정리하시오

-정방행렬, 대칭행렬, 상관계수행렬, eigen분해에 대하여 정리하시오.

정방행렬 = m(가로) 와 n(세로) 가 같은, 즉 행과 열의 갯수가 같은 행렬을 말한다.
특징으로는, '정방행렬'만, 역행렬을 가질 수 있다.

대칭행렬은 상 하 가 같은 행렬이다. 전치행렬이 스스로와 같은 행렬이다.
예시로는 상관행렬과 거리값행렬이 있다.


공분산 행렬 (Covariance Matrix), 상관계수 행렬 (Correlation Coefficient Matrix)
2이상의 변량들에서, 다수의 두 변량 값들 간의 공분산 또는 상관계수들을 행렬로 표현한 것
확률벡터의 분산 표현 =>  공분산 행렬 : Var [x]

고유값 분해(eigen decomposition)는 고유값 과 고유벡터로 부터 유도되는 고유값 행렬과 고유벡터 행렬에 의해 분해될수있는 행렬의 표현이다.
선형대수학에서 , 고유값 분해는 매트릭스(행렬)를 정형화된 형태로 분해함으로써 행렬이 고유값 및 고유 벡터로 표현된다.
고유벡터는 직교한다. 사이즈가 1인 행렬을 만든다.
벡터가 직교한다는 말은, 서로 독립이다는 의미이다. 데이터가 서로 영향을 미치지 않아, 다중공선성에 영향을 받지 않는다.

통계학에서 주성분 분석(Principal component analysis; PCA)은 고차원의 데이터를 저차원의 데이터로 환원시키는 기법이다.
서로 연관 가능성이 있는 고차원 공간의 표본들을 선형 연관성이 없는 저차원 공간(주성분)의 표본으로 변환하기 위해 직교 변환을 사용한다.
주성분의 차원수는 원래 표본의 차원수보다 작거나 같다.
주성분 분석은 데이터를 한개의 축으로 사상시켰을 때 그 분산이 가장 커지는 축을 첫 번째 주성분,
두 번째로 커지는 축을 두 번째 주성분으로 놓이도록 새로운 좌표계로 데이터를 선형 변환한다.
이와 같이 표본의 차이를 가장 잘 나타내는 성분들로 분해함으로써 여러가지 응용이 가능하다.
이 변환은 첫째 주성분이 가장 큰 분산을 가지고, 이후의 주성분들은 이전의 주성분들과 직교한다는 제약 아래에
 가장 큰 분산을 갖고 있다는 식으로 정의되어있다.
 중요한 성분들은 공분산 행렬의 고유 벡터이기 때문에 직교하게 된다.

특이값 분해(SVD)는 고유값 분해(eigendecomposition)처럼 행렬을 대각화하는 한 방법이다.
특이값 분해가 유용한 이유는 행렬이 정방행렬이든 아니든 관계없이 모든 m x n 행렬에 대해 적용 가능하기 때문이다.



-load_iris()함수로부터 생성되는 데이터를 2개의 주성분으로 분해한 다음 2차원 그래프로


import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = load_iris()
scaler = StandardScaler() #전처리 모델
scaler.fit(iris.data) # 값을 구하는 과정 : 평균/ 표준편차
X_scaled = scaler.transform(iris.data) #일반 머신러닝에서는 predict를 하는데, 여기서는 transform을 하지요
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  #원래 30개를 2개로 표현합니다. 가장 중요한 성분 2개를 출력해줍니다. 가장 주성분-정렬
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print(pca.explained_variance_ratio_)
print("원본 데이터 형태: {}".format(str(X_scaled.shape)))
print("축소된 데이터 형태: {}".format(str(X_pca.shape)))


import matplotlib.pylab as plt
plt.plot(pca.explained_variance_ratio_, 'bo-')
plt.show()

import mglearn
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter    (X_pca[:, 0], X_pca[:, 1], iris.target)
plt.legend(["일번", "이번"], loc="best")
plt.gca().set_aspect('equal')
plt.xlabel("첫 번째 주성분")
plt.ylabel("두 번째 주성분")
plt.show()

-kmeans 예제중에 하나를 골라 TSNE 를 이용하여 시각화하여 군집 갯수를 정해보시오.
