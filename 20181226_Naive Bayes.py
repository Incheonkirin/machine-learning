#신경망
    # MLPClassifier, MLPRegressor : 앞에 MLP가 붙는 것은, multi layer perceptron이기 때문
import os
os.chdir("C://Users/Hyungi/Desktop/workplace")
os.getcwd()


import matplotlib.pylab as plt
import numpy
from sklearn.datasets import load_boston
data = load_boston()

for index, feature_name in enumerate(data.feature_names):
    plt.figure()
    plt.scatter(data.data[:, index], data.target)


from sklearn.neural_network import MLPRegressor
X, y = load_boston(return_X_y=True)
half_n_samples = int(len(X)/2)
X_1, X_2, y_1, y_2 = (X[:half_n_samples], X[half_n_samples:],
        y[:half_n_samples], y[half_n_samples:])

# warm_start는 이전에 사용하던 값을 계속 사용할 것인가 여부를 결정합니다.
for random_state in [None, 3]:
    for warm_start in [False, True]:
        print('\nwarm_start:', warm_start, '\nrandom_state:', random_state)
        model = MLPRegressor(warm_start=warm_start, random_state=random_state)
        model.fit(X_1, y_1)
        weights_1 = model.coefs_
        print('첫번째 coef_:\n', model.coefs_[0][0][:10])
        model.fit(X_2, y_2)
        print('두번째 coef_:\n', model.coefs_[0][0][:10])

#activation함수와 solver, alpha의 parameter tuning  #solver는 뭐인가요? optimize 최적화기.무엇을 최적화하나요? 최적화대상은 cost function을 최적화합니다.
#                                                   #adam을 가지고 설명해보면, adam이 지원하는 것은, momentum이죠. 지역해를 피하기위해서 가는 방향으로 더 전진시켜봅니다.
#                                                    지역해를 극복.
#                                                     또 하나는 learning_rate의 문제. learning_rate가 큰 경우에는 어떻게 되나요? 왔다갔다 점핑하지요? 옵티마이즈 찾지 못하고, 뛰어다녀요
#                                                     작은경우에는 ? 오래걸립니다. 시간지연
#                                                       그래서 처음에는 크게, 나중에는 작게. 해 근처에서는 아주 작게하는 것입니다.
#                                                        adam의 문제점은 스케일에 민감하다는 것입니다. 즉, 정규화가 필요합니다.
#                                                         lbfgs는 대부분의 경우에 잘 작동합니다. (adam이 기본)  그런데 문제가 있습니다. 대규모의 경우 시간이 오래걸립니다.
#                                                         Limited-Memory Broyden–Fletcher–Goldfarb–Shanno (BFGS)
#                                                         최적화알고리즘

# alpha는 뭔가요? 규제입니다. 그래서 best_estimator_에서 activation알아보려면 어떻게 해야 하나요?
# print(model_cv.best_estimator_.out_activation_)를 사용하면 되죠.

#activation 함수의 역할 : 선형을 비선형으로 맵핑. 값을 제약하면서. (지금하는 것은 회귀, 값을 제약하면 안됨.)
# relu ?
# tanh ?
# logistic : 0~1 사이 곡선에서 이뤄지는 값으로 이뤄지는 확률 값.
# identity ?

from sklearn.model_selection import GridSearchCV
train_size = 100
model = MLPRegressor(warm_start="True", random_state=3)
model_cv = GridSearchCV( model, cv=5,
        param_grid={"activation": ["relu", "tanh", "logistic", "identity"],
        "solver":["lbfgs", "adam"],
        "alpha":[0.0001, 0.001, 0.01, 0.1, 1] })#regulization
model_cv.fit(X, y)
print("모델 계수", model_cv.best_estimator_)
print(model_cv.best_estimator_.out_activation_)


# 신경망에서 비지도학습 ( sklearn : 비지도학습 RBM)
# 신경망에서 비지도학습 RBM은 마치 PCA처럼, 특성을 추출해줍니다.

import numpy as np
import matplotlib.pyplot as plt


from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split


def nudge_dataset(X,Y):
    direction_vectors = [ #필터는 이미지의 대상에 따라서 달라집니다. 대상에 따라서 어떤 필터를 적용하는가가 특징추출에 중요한 요소가 됩니다. ex 도시: 선형, 산: 비선형
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 0]]]
    # colvolve라는 함수가 convolution을 제공합니다. 이미지 필터를 적용해서 이미지의 특징 : CNN에서 중심개념
    # CNN(Convolution Neural Network: Convolution이란? 주변을 고려해서 픽셀값을 재설정) , RNN, autoencoder(특징추출에서 비지도학습 지난주 금요일에 설명했었죠), GAN
    # convolution 통신에서 빼먹은 부분 (주변 부분의 평균이 될 것이다).
    #좀 더 정교하게 사용하기 위해서, 이미지를 확장 (주변의 값을 고려해서 확장)
    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()
    #주변의 값으로 convolution 하라.

    X = np.concatenate([X] +
            [np.apply_along_axis(shift, 1, X, vector)
            for vector in direction_vectors])
    #값이 들어오면 열방향으로 X값을 계산하게 됩니다.

    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target) #데이터(이미지)확장 . convolution을 사용.
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001) #정규화
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
        random_state=0)


# Boltzmann Machine 상하좌우  fully-connected를 이용한 연결망으로 학습
# Restricted :  같은 레이어의 회로 제거, 레이어와 레이어만 연결 / 해서 특징을 추출
# RBM은 제한된 Limited Boltzmann Machine  비지도학습으로 특징을 추출 (딥러닝) / sklearn에 들어와서.. PCA처럼 특징을 추출하는데 사용할 수 있게 끔 함.
# RBM은 딥러닝의 신경망을 sklearn에서도 간단히 사용할 수 있게끔 만들어 놓은 모델이다.


###특징 추출 후 로지스틱 회귀분석
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose= True)
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
#               파이프라인에 rbm과 logistic 수행. PCA한 담에 로지스틱하란 이야기와 같은 말.
#                연산이 너무 많으니까, 제한 Limited가 있는 것.
rbm.learning_rate = 0.06
rbm.n_iter =20
rbm.n_components = 100
logistic.C = 6000.0
classifier.fit(X_train, Y_train)
### 일반 로지스틱 회귀분석
logistic_classifier = linear_model.LogisticRegression(C=100.0)
# 제약조건 100을 가진 리그레션을 이용해서 만들어진 모델.

logistic_classifier.fit(X_train, Y_train)

#########################################################################
# ! 지금 RBM이 특징 추출하는데 사용한다는 것을 확실히 알아 두었으면 좋겠다 ! #
#########################################################################

print("logistic 분류기 :\n%s\n" %(
        metrics.classification_report( Y_test,
        logistic_classifier.predict(X_test))))

print("RBM 특성 추출을 이용한 logistic:\n%s\n" % (
        metrics.classification_report( Y_test,
        classifier.predict(X_test))))



## naive bayes
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(0)
X0 = sp.stats.norm(-2, 1).rvs(40) #rvs는 무슨 함수입니까? 랜덤 표본 생성 (random variable sampling) ,
#                                 #pdf                  확률밀도함수 (probability density function),
#                                 #cdf                  누적분포함수 (cumulative distribution function) ,
#                                 #fcdf                 누적분포함수의 역함수 (inverse cumulative distribution function) # 찾아서 써 보시오.
X1 = sp.stats.norm(+2, 1).rvs(60)
X = np.hstack([X0, X1])[:, np.newaxis] #여기서 newaxis를 해 주는 이유는 무엇인가요?
#                                       행으로 합치고, 열로 분리하고.  hstack은 무엇인가요? horizontalstack으로 행으로 쌓고..
y0 = np.zeros(40) #40개에 대해서 0번을 주고
y1 = np.ones(60) #60개에 대해서 1을 줍니다.
y = np.hstack([y0, y1])
y
X0
X1
sns.distplot(X0, rug=True, kde=False, norm_hist=True, label="class 0")
sns.distplot(X1, rug=True, kde=False, norm_hist=True, label="class 1")

plt.legend()
plt.xlim(-6, 6)
plt.show()



# 가우시안 - 이 잘 피팅하느냐. 보는 것입니다.
# 이러한 데이터가 있을 때 분포를 잘 찾아내느냐. 하는 것입니다.
from sklearn.naive_bayes import GaussianNB
clf_norm = GaussianNB().fit(X, y)
clf_norm.classes_
clf_norm.class_count_
clf_norm.class_prior_
clf_norm.theta_, clf_norm.sigma_
xx = np.linspace(-6, 6, 100) #범위값을 -6부터 6까지로 주고..
xx

#평균, 분산
p0 = sp.stats.norm(clf_norm.theta_[0], clf_norm.sigma_[0]).pdf(xx)
p0
p1 = sp.stats.norm(clf_norm.theta_[1], clf_norm.sigma_[1]).pdf(xx)
p1

#데이터 구성이 2개의 확률분포로 부터 생성된 데이터
sns.distplot(X0, rug=True, kde=False, norm_hist=True, color="r", label="class 0 histgram")
sns.distplot(X1, rug=True, kde=False, norm_hist=True, color="b", label="class 1 histgram")
plt.plot(xx, p0, c="r", label="class 0 est. pdf")
plt.plot(xx, p1, c="b", label="class 1 est. pdf")
plt.legend()
plt.show()

# 데이터0가 2개가 합쳐져있는데, 가우시안 나이브 베이즈가 쎄타하고 시그마를 다 잘 찾아내고 있어야해요쎄타는 평균이고 시그마는 분산인데,
# 원래의 데이터 확률 분포를 찾아내고 있는 모양입니다.



# 나이브베이즈는 확률을 기반으로 하는 것 입니다.
x_new = -1
clf_norm.predict_proba([[x_new]]) # 분포속에서 확률을 계산 출력. 확률값 에측
# -1 이 왼쪽에 속하게 될 확률.

# 분포속의 확률을 확인해서, 각기 집합에 속하는 확률값을 확인해줘서 98.3%는 왼쪽집단에 속한다는 이야기에요

#내부적인 계산 절차
#우도 계산
px = sp.stats.norm(clf_norm.theta_, np.sqrt(clf_norm.sigma_)).pdf(x_new)
# 쎄타와 시그마 값이 나왔으니까 ,
px
p = px.flatten() * clf_norm.class_prior_ #우도 값을 곱해주는 것입니다. 베이즈정리에 분자값을 계산.
p

clf_norm.class_prior_
clf_norm.theta_
clf_norm.sigma_
p / p.sum()
#전체 합친 것으로 나눠주게 되면,  베이즈 정리 를 계산하는 절차와 같다.



# 베르누이 나이브 베이즈
#베르누이는 독립변수 및 종속변수 모두 '이항값'?을 가져야 함.

np.random.seed(0)
X = np.random.randint(2, size=(10, 4))
y = np.array([0,0,0,0,1,1,1,1,1,1])
print(X)
print(y)
from sklearn.naive_bayes import BernoulliNB
clf_bern = BernoulliNB().fit(X, y)
clf_bern.classes_
clf_bern.class_count_
np.exp(clf_bern.class_log_prior_)

x_new = np.array([1, 1, 0, 0])
clf_bern.predict_proba([x_new])


# TfidVectorizer

# 정치 경제 등으로 분류된 뉴스 기사
#지금하는 것은 MultinomialNB => 다항분포입니다. 다항분포는 언제 쓴다고 했죠? 이산적 count 데이터일 때 사용한다고 했어요.
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset="all")
X = news.data
y = news.target

# tfidf : term frequencty(단어 빈도) / inverse document frequency (역문서 빈도)
# 역문서 빈도는 언제 높아지나요? 모든 문서에 단어가 존재하면 작아집니다.
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
model1 = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB()), ])
model2 = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB()), ])
# STOPWORD 제거
model3 = Pipeline([('vect', TfidfVectorizer(stop_words="english")), ('clf', MultinomialNB()),
        ])
# 아무 단어나 쓴게 아니라, 특정 패턴을 가지고 있는 단어만을 대상
model4 = Pipeline([
        ('vect', TfidfVectorizer(stop_words="english",
                token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b")),
                ('clf', MultinomialNB()),
                 ])


#평가
from sklearn.model_selection import cross_val_score
for i, clf in enumerate([model1, model2, model3, model4]):
    score = cross_val_score(clf, X, y, cv=5)
    print(("Model{0:d}: Mean score:{1:.3f}").format(i, np.mean(scores)))

print(X[0])
print(y) # 20개의 종류로 나옵니다.



##
# GMM( Gaussian Mixture model ) 확률 분포를 사용해서 타원으로 클러스터링
# kmeans : 원형,
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,
        cluster_std=0.60, random_state=0)
X = X[:, ::-1]
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture (n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
probs = gmm.predict_proba(X)
print(probs[:5].round(3))
plt.show()

##가우시안이 믹스쳐 되어있는 데이터이다. 타원으로.

from matplotlib.patches import Ellipse
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis',
                   zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

# kmeans와 마찬가지로 components를 넣어줘야 한다. 몇 개로 나눠야 할지.
gmm = GaussianMixture(n_components=4, random_state=42)
plot_gmm(gmm, X)
plt.show()
##

# 데이터에 ??행렬을 곱하면 뭐가 된다고 했어요? 선형변환
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2)) #선형변환
# full 컴포넌트 별로 variance matrix을 계산
# tied 컴포넌트 별로 같은 variance matrix을 사용합니다.
# diag 크기만 가지고 있는 것. diagonal variance matrix(대각으로, 대각의 값은 크기만을 결정)
gmm = GaussianMixture(n_components=4, covariance_type="full",
        random_state=42)
plot_gmm(gmm, X_stretched)
plt.show()




# 그래서 확률을 기반으로 한 모델은..
# 로지스틱, 나이브베이즈, 가우시안믹스쳐모델,
# logistic, NB, GMM
# 그 다음에 나중에, LDA(문서 - 토픽 분류)

# 그 다음에 하나 있는게, HMM (Hidden Markov Model)

일단 마코브 모델에 대해서 간단하게 이해해보겠습니다.
회사가 두개 있습니다.
k회사 가 있고, k사 경쟁회사가 있습니다.

초기 시장 점유율이 0.25, 0.75

전이행렬          0.88, 0.12 k사 제품 계속 쓸사람, 0.88, k사에서 경쟁사 제품으로 옮길 사람 0.12,
                 0.15, 0.85 k사 경쟁사 제품, 0.15, k경쟁사에서 k사 0.85

일때, 2년후의 시장 점유율이 어떻게 되겠는가.
이것이 마코프 모델입니다.


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

states = ['K', 'nK']
pi = [0.25, 0.75]
state_space = pd.Series(pi, index=states, name='states')
print(state_space)
print(state_space.sum())

q_df = pd.DataFrame(columns=states, index=states) #확률행렬
q_df.loc[states[0]] = [0.88, 0.12]
q_df.loc[states[1]] = [0.15, 0.85]
print(q_df)
q = q_df.values
print('\n', q, q.shape, '\n')
print(q_df.sum(axis=1))

k1 = np.dot(state_space, q_df)
k2 = np.dot(k1, q_df)
k2

# markov model : 행렬 거듭제곱 연산  markov chain이라고 합니다.
# hidden state 를




# 타이타닉 데이터를 이용한NB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

data = pd.read_csv("dataset/titanic.csv")

# X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))

data["Sex_cleaned"]=np.where(data["Sex"]=="male",0,1)
data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,
        np.where(data["Embarked"]=="C",1,
                np.where(data["Embarked"]=="Q",2,3)
                )
        )




print(data.head())
data=data[[
        "Survived",
        "Pclass",
        "Sex_cleaned",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked_cleaned"
        ]].dropna(axis=0, how='any')

X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))

gnb = GaussianNB()

used_features =[
        "Pclass",
        "Sex_cleaned",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked_cleaned"
        ]


X_train








gnb.fit(
        X_train[used_features].values,
        X_train["Survived"]
        )



y_pred = gnb.predict(X_test[used_features])

print(" 총  {} , 잘못 분류된 것  : {}, 정분류율 :  {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*((X_test["Survived"] == y_pred).sum()/X_test.shape[0])
            ))

# 잘못된 것을 기준으로 할 수 있다.
print(" 총  {} , 잘못 분류된 것  : {}, 정분류율 :  {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])

      ))



문제
1)train과 test 를 5:5 로 나누고, 종속변수는 survived로 하시오

2)운임(fare)과 survived와의 관계 모델을 생성하고, 예측 해 보시오.
#   독립        종속
# => 표준편차를 구해주고,
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()



#답
used_features=["Fare"]
y_pred = gnb.fit(X_train[used_features].values, X_train["Survived"]).predict(
        X_test[used_features])
print("총 {} 중에 잘못 분류된 수: {}, {:05.2f}%"
    .format(
        X_test.shape[0],
        (X_test["Survived"] != y_pred).sum(),
        100*(1-(X_test["Survived"] !=y_pred).sum()/X_test.shape[0])
    ))





3)살아있는 사람들의 운임의 평균과 표준편차를 구하시오
mean_fare_survived = np.mean(X_train[X_train["Survived"]==1]["Fare"])
std_fare_survived = np.std(X_train[X_train["Survived"]==1]["Fare"])
print("mean_fare_survived={:03.2f}".format(mean_fare_survived))
print("std_fare_survived={:03.2f}".format(std_fare_survived))

4)생존확률과 사망확률을 구현하시오
mean_survival=np.mean(X_train["Survived"])
mean_not_survival=1-mean_not_survival
print("생존확률 = {:03.2f}%, 사망확률 = {:03.2f}%".format(100*mean_survival,
        100*mean_not_survival))

5)생존자 운임의 표준편차를 구하시오
print("생존자 운임 표준편차 {:05.2f}".format(np.sqrt(gnb.sigma_)[1][0]))
print("비생존자 운임 표준편차 {:05.2f}".format(np.sqrt(gnb.sigma_)[0][0]))

6)생존자 운임 평균을 구하시오
print("생존자 운임 평균:{:05.2f}".format(gnb.theta_[1][0]))
print("비생존자 운임 평균:{:05.2f}".format(gnb.theta_[0][0]))










# 문제
#이 문제에서 종속변수는, target : unacc
car.data를 로딩하고, naive bayes모델을 사용하여 분류기를 구현하시오.

#columns 이름 주는 방법 꼭 익히자...
data = pd.read_csv('dataset/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot',
        'safety', 'class'])
print(data.head())
