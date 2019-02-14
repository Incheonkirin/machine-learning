import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

#전처리
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

#전처리를 지원하는 패키지
#R에서는  scale = z점수
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale

print((np.arange(10, dtype=np.float) -3))
x = (np.arange(10, dtype=np.float) -3).reshape(-1, 1) #행과 열
print(x)
df = pd.DataFrame(np.hstack([x, scale(x), robust_scale(x), minmax_scale(x),
maxabs_scale(x)]),
columns=['x','scale(x)', 'robust_scale(x)',
'minmax_scale(x)', 'maxabs_scale(x)'])

df



#분포의 변화는 불변
import seaborn as sns
from sklearn.datasets import load_iris
iris = load_iris()

data1 = iris.data
data2 = scale(iris.data)
print( 'old mean:', np.mean(data1, axis=0))
print('old std:', np.std(data1, axis=0))
print('new mean:', np.mean(data2, axis=0))
print('new std:', np.std(data2, axis=0))
sns.jointplot(data1[:,0], data1[:,1])
plt.show()
sns.jointplot(data2[:,0], data2[:,1])
plt.show()


from sklearn.preprocessing import StandardScaler
#MinMaxScaler, MaxaAbsScaleer, RobustScaler, Normalizer(내적: 방향값)

scaler = StandardScaler()
scaler.fit(data1) #적합(평균, 표준편차)
data2 = scaler.transform(data1) #스케일 된 데이터
data1.std(), data2.std()

from sklearn.preprocessing import Normalizer
Normalizer().fit_transform(data1)



#onehotencoding
import numpy as np
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

X = np.array([[0], [1], [2]])
X

ohe.fit(X)
ohe.n_values_, ohe.feature_indices_, ohe.active_features_
print(ohe.transform(X).toarray())



#다음을 onehotencoding 해보시오.
X = np.array([[0, 0, 4], [1, 1, 0], [0, 2, 1], [1, 0, 2], [1, 1, 3]])
X
ohe.fit(X)
ohe.n_values_, ohe.feature_indices_, ohe.active_features_
print(ohe.transform(X).toarray())



# 숫자 범주화
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit([1, 2, 2, 6])
le.classes_ # scikit 에서 결과변수인 경우, '_'를 붙이는 관행이 있다.
le.transform([1, 1, 2, 6])# labeling 하라

le.fit(["서울", "서울", "대전", "부산"])
le.classes_
le.transform(["서울", "서울", "부산"])



# 0과 1로 레이블링하라.
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit([1, 2, 6, 1, 2]) # 1, 2, 6 의 세가지 경우가 있다.
lb.classes_
lb.transform([1, 6]) #1을 표현하는 것과, 6을 표현하는 것을 보여줌.


# dictionary feature 정보를 matrix로 표현.
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse= False)
D = [{'foo':1, 'bar':2},{'foo':3, 'baz':1}] #한 문서에 foo가 1번, bar가 2번 나오고, 다른 문서에는 foo 3번, baz 1번.
X = v.fit_transform(D) #transform
X
v.feature_names_ #알파벳순서, bar baz foo
v.inverse_transform(X)# 다시 위의 dict 형식으로 바꿔줌.


#문자열에 대해서도 가능합니다.
instances = [ {'city' : 'New York'}, {'city': 'San Francisco'},
{'city':'Chapel Hill'}]
v.fit_transform(instances)
v.feature_names_

#결측치 처리
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
#파생변수 생성
# [1, a, b, a^2, ab, b^2]

from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
X

poly = PolynomialFeatures(2)
poly.fit_transform(X)
## [1, a, b, a^2, ab, b^2]


#apply와 같은 효과
from sklearn.preprocessing import FunctionTransformer
def all_but_first_column(X):
    return X[:, 1:]
X = np.arange(12).reshape(4, 3)
X

FunctionTransformer(all_but_first_column).fit_transform(X)


# pip install mglearn
#회귀분석
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)

#독립변수 종속변수를 한꺼번에 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train) #모델생성
print("lr.coef_: {}".format(lr.coef_)) #slope
print("lr.intercept_: {}".format(lr.intercept_)) #절편
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
random_state=0)

lr = LinearRegression().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train))) #평가
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
# train데이터에 과적합




#회귀분석
    # regulization
        #과적합을 방지하기 위해서, 규제를 적용합니다. 1) L1 규제(계수들의 절대값), 2) L2 규제(계수들의 제곱값) 벌점

# Ridge : L2 규제 사용
# Lasso : L1 규제 사용
# ElasticNet : L1, L2 규제를 적용한 모델
# alpha 매개변수  : L1규제에서 alpha값을 낮추면 규제효과가 없어집니다.
# 일반화를 높이려면, alpha값을 높여주면 됩니다.
#   분류로 사용할 때는  LineaSVC를 사용합니다.

# 잔차는 3가지 종류가 있습니다.
    # PRESS잔차, 잔차 제곱합
    # 표준화잔차, 잔차 / 잔차의 표준편차  = 2보다 크면, 특이치
    # Student 잔차 : 잔차 / 표준오차   = 이상치 검출에 유리
    #                     표준오차는 무엇을 고려? 관측치 갯수를 고려


# 대용량 관측치 일 경우
    # SGDClassifier
    # SGDRegressor

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1, max_iter=1000).fit(X_train, y_train) # L2규제 : 과적합방지, alpha계수
#0.93 0.75

# alpha값을 높여서 확인해보세요
#ridge = Ridge(alpha=0.5, max_iter=1000).fit(X_train, y_train) # L2규제 : 과적합방지, alpha계수
#0.69 0.74

print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))



from sklearn.linear_model import Lasso # L1규제

lasso = Lasso(alpha = 0.0001, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}". format(lasso.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso.score(X_test, y_test)))
print("사용한 특성의 개수: {}".format(np.sum(lasso.coef_ !=0)))





import mglearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5,
    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend()
plt.show()



from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=3,
n_redundant=0)
from sklearn.tree import DecisionTreeClassifier #  *tree
dt = DecisionTreeClassifier()
dt.fit(X, y)
preds = dt.predict(X)
(y == preds).mean() # 1.0 (100% 가 나오게 된다.)



#앙상블
from sklearn.datasets import make_classification #학습용함수
X, y = make_classification(1000) #디폴트 주지 않았기 때문에, 자동으로 20개.

from sklearn.ensemble import RandomForestClassifier   #  *ensemble
rf = RandomForestClassifier()
rf.fit(X, y)
print("Accuracy:\t", (y == rf.predict(X)).mean())

#랜덤 포레스트는 변수 중요도를 출력해줍니다.
f, ax = plt.subplots(figsize=(7, 5))
ax.bar(range(0, len(rf.feature_importances_)), rf.feature_importances_)
ax.set_title('특성중요도')
print('종속변수 갯수', rf.n_classes_ )
print('클래스 종류', rf.classes_ ) #0,1
print('특성 수', rf.n_features_ )#
print('모델', rf.estimators_ )

plt.show()



#alfair는 javascript로 만들어져있음. 파이참, 스파이더 등에서 출력이 안됨.

#실제 데이터
#conda install -c conda-forge graphviz
#conda install -c conda-forge python-graphviz

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(forest.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(forest.score(X_test, y_test)))
from sklearn.tree import export_graphviz
# impuritty 불순도 출력 여부 : gini계수
export_graphviz(forest.estimators_[0], out_file='tree.dot',
class_names=['악성', '양성'], feature_names=cancer.feature_names,
impurity=True, filled=True)
#impurity=False, filled=True)

import graphviz
from Ipython.display import display #display는 이미지 출력할 때 사용함.
with open("tree.dot", "rt", encoding="UTF-8") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))




# 문제 : dicision tree / randomForest 모델로 다음 데이터를 분석하시오.
# - 정규화진행
# - 결측치진행

from sklearn.datasets import load_iris
