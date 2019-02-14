# XGBoost

# anaconda search -t conda xgboost

import os
os.getcwd()
os.chdir("C://Users/Hyungi/Desktop/workplace")

import matplotlib.pylab as plt

# pipeline (작업처리 순서를 지정한 객체), train / test (( R에서는 따로따로 처리했는데, 여기서는 아예 분리해서..여러번 할 것을 한번에))
# accurracy_score
# GridSearchCV : parameter tuning : ( R에서는 caret 패키지에서 지원했음)

# validation_curve : 하나의 변수에 대해서 여러값으로 tuning
# GridSearchCV : 행렬 그리드 경우의 수를 조합해서 처리
# parameterGrid : 여러개의 변수, dictionary를 이용해서 1차원적으로/ dictionary를 이용하여 경우의 수를 조합


# SelectFromModel : 임계점을 기준으로 모델에서 변수를 선택해서 모델을 만든다.
# ndarray - pandas =>

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()

dir(iris)
print(type(iris))
feat_labels = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
X = iris.data
y = iris.target
X[0:5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# n_jobs 는 CPU 병렬처리 , -1 을 주면 가능한 모든 CPU를 사용하라.
# cpu core 병렬처리
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)
for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)
    #변수 중요도 Petal Length가 가장 높음

# 전처리함수 transform이 있다.
# machine learning model들은 fit()가 있다.
sfm = SelectFromModel(clf, threshold=0.15)
sfm.fit(X_train, y_train)
for feature_list_index in sfm.get_support(indices=True): #indice는 순서
    print(feat_labels[feature_list_index])
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)


#그 데이터를 사용해서 다시 Random Forest를 한번..
clf_important = RandomForestClassifier(n_estimators = 10000, random_state=0, n_jobs=-1)
clf_important.fit(X_important_train, y_train)
# 4개의 변수 93%
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# 중요한 변수 2개 88%
y_important_pred = clf_important.predict(X_important_test)
accuracy_score(y_test, y_important_pred)





## GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import make_scorer

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, random_state=0)

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
learner = RandomForestClassifier(random_state = 2)
n_estimators = [12, 24, 36, 48, 60]
min_samples_leaf = [1, 2, 4, 8, 16]
# 25가지의 경우의 수

parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf}
# 사용자 정의 평가 함수

def auc_scorer(target_score, prediction): #모델 평가시 (R에서) roc커브.
    auc_value = roc_auc_score(prediction, target_score) # area under curve(auc)
    return auc_value
#평가 방법: 좋은 점(긍정평가), cost평가(작은것이 좋다), 오차평가

#아래는 '긍정평가' 방법, 값이 크면 좋다.
scorer = make_scorer(auc_scorer, greater_is_better= True) # factory class
grid_obj = GridSearchCV(learner, parameters, scorer) #모델, 파라메터, 평가 ( 3가지가 들어감. )
grid_obj.fit(X_train_scaled, y_train)

#평가 결과 값이 많아요.  mean_test_score값이, 평가 결과 값 중에서 ..
#                                               reshape로 2차원 만드는이유? 결과가 1차원(배열)으로되어있어요. 2중 for문으로 만들어,
scores = grid_obj.cv_results_['mean_test_score'].reshape(len(n_estimators), len(min_samples_leaf))
# 평가를 하게 되어 집니다.
plt.figure(figsize=(8,6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('n_estimators')
plt.ylabel('min_samples_leaf')
plt.colorbar()
plt.xticks(np.arange(len(n_estimators)), n_estimators)
plt.yticks(np.arange(len(min_samples_leaf)), min_samples_leaf)
plt.title('Grid Search AUC Score')
plt.show()



# leaf 단말노드의 가장 작은 샘플 수를 몇개로 할 것인가?
# 작게주면, 잘게 나누어라 = 과적합




#이번에는 두가지를 섞어보겠습니다.
#파이프라인이랑 그리드서치CV를 두가지를 섞어서
#지금은 모델이 중요한 게 아니라, 그리드서치CV를 어떻게 사용하는지 아는 것이 중요합니다.

# pipeline / GriidSearchCV
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
import mglearn
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

from sklearn.preprocessing import PolynomialFeatures

pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(),
    Ridge())
# pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

param_grid = {'polynomialfeatures__degree': [1, 2, 3],       #특징 추가값. 이걸 어디서 확인할 수 있을까??
              'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
# param_grid = {'PolynomialFeatures__degree':[1, 2, 3], 'ridge__alpha':[0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
# grid = GridSearchCV(pipe, param_grid = param_grid, cv=5, n_jobs=-1)
# grid.fit(X_train, y_train)

# cv : 5개로 접어라

# print(gird.cv_results_['mean_test_score'])

mglearn.tools.heatmap(grid.cv_results_['mean_test_score'].reshape(3, -1),
                      xlabel="ridge__alpha", ylabel="polynomialfeatures__degree",
                      xticklabels=param_grid['ridge__alpha'],
                      yticklabels=param_grid['polynomialfeatures__degree'], vmin=0)

plt.show()
# 이렇게 하면 모델이 어디있다는 말인가? Grid를 fittingg 했으니까, 이제 뭐가 남아야 정상인가??
# 제일 좋은 녀석이 나와야 함..

print(grid.cv_results_)

#제일 좋은 건, best_estimator에 으로 되어있어요
print(grid.best_estimator_)
print(grid.best_score_)
print(grid.best_params_)
print(grid.n_splits_)


#홈페이지에서 가져온 내용...
print("최적의 매개변수: {}".format(grid.best_params_))
print("테스트 세트 점수: {:.2f}".format(grid.score(X_test, y_test)))

aram_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("다항 특성이 없을 때 점수: {:.2f}".format(grid.score(X_test, y_test)))





#전용모델로 사용
import xgboost as xgb
dtrain = xgb.DMatrix('agaricus.txt.train')
dtest = xgb.DMatrix('agaricus.txt.test')
#obje

param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'} #logistic 은 종속변수가 범주형일때.
#                     쪼개서들어가라
#                              아무런 메시지없이
num_round = 2
bst = xgb.train(param, dtrain, num_round)

preds = bst.predict(dtest)
print(preds[:5])




from sklearn.datasets import load_boston
boston = load_boston() # 키이 데이터 형식, 보스턴 데이터 data(독립변수), target
print(boston.keys())
print(boston.data.shape)
print(boston.feature_names)
print(boston.DESCR) #이렇게 하면, 데이터에 대한 설명을 볼 수 있습니다.



#이번에는 이걸 판다스로 바꿔보겠음.

import pandas as pd
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

data.head()
data['PRICE'] = boston.target
data.info()
data.describe()


# scikits wrapper 방식 / DMatrix를 이용하는 전용방법
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
X, y = data.iloc[:,:-1], data.iloc[:,-1] # iloc 은 숫자일 때 사용함. 맨 마지막은 y값으로, 그 앞에는 x값으로 변환시켜줌.
data_dmatrix = xgb.DMatrix(data=X, label=y) # X,y를 분리해서 집어넣습니다.
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# 목적함수, 열을 취하는 개수

# hyper parameters - parameter tuning을 해주어야 함. 여기서는 fixed해서 학습하는 목적으로 했지만, 원래는 앞에서 gridsearchCV로 tuning을 해 주어야합니다. 튜닝없이 아래와 같이 결정할 수 없습니다.
xg_reg = xgb.XGBRegressor(objective = 'reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
#                                                                                                               규제
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
#여기서는 평가를 부정적으로, error 로 하고 있습니다.

print("RMSE: %f" % (rmse))



# DMatrix 전용모델
params = {"objective": "reg:linear", 'colsample_bytree':0.3, 'learning_rate':0.1, 'max_depth':5, 'alpha':10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
#                                                               boost되어지는것을 몇번?
#                                                                                   10라운드까지 모델 개선 안되면 중지하라
#                                                                                                                           결과를 데이터프레임으로(아니면 ndarray로) 만들어라
cv_results.head()
print((cv_results["test-rmse-mean"]).tail(1))
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
import matplotlib.pyplot as play_13_14_top30

xgb.plot_tree(xg_reg, num_trees=0)
plt.rcParams['figure.figsize'] = [100, 20]
plt.show()



xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()



#   XGBRegressor와 DMatrix의 차이점
# XGBRegressor는 n_estimators 개수를 지정
# DMatrix는  booster가 작동, booster- pruning 을 하고, 가지치기를 함으로써 정확도를 개선.
#                                                     개선이 안되면, early_stopping_rounds으로 마치게 된다.

#그래서 어떤게 더 좋습니까 ?

# xgboost의 전용방법이 더 좋다고 생각한다.



# XGBClassifier
import xgboost

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dataset = loadtxt('pima-indians-diabetes.data', delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size = test_size, random_state = seed)
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
from xgboost import plot_importance
from matplotlib import pyplot

plot_importance(model)
pyplot.show()
y_pred = model.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("정확도: %.2f%%" % (accuracy * 100.0))



# 변수 중요도에 따른 모델 평가

print(model.feature_importances_)
thresholds = np.sort(model.feature_importances_)
print(thresholds)


#thresholds에 들어있는 변수 중요도에 따라서,

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    #                                     경계값 중심으로 하고
    select_X_train = selection.transform(X_train)
    # 모델 선택한 담에
    selection_model = XGBClassifier()
    #선택 된 변수만 나오게 되고
    selection_model.fit(select_X_train, y_train)
    # fit해서
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    #predict해서
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    #accuracy해서 모델을 평가
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh,
    select_X_train.shape[1], accuracy*100.0))


#모델을 저장을 합니다. 피클을 이용해서..
#피클은 메모리에 있는 내용을 그대로 덤프해서 저장합니다.


import pickle
pickle.dump(model, open("pima.pickle.dat", "wb"))
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
y_pred = loaded_model.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
accuracy
print("Accuracy: %.2f%%" % (accuracy * 100.0))



# 문제 people.data 와 people.test 데이터를 로딩한 다음 문제를 처리하시오
변수명
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
             'wage_class']


1) 결측치 처리 (없는것, 물음표)
2) 범주형 변수 처리 (14번째 변수에 크거나 같다, 작다(2가지). 뒤에 '.' 이 붙여있는 것 제거)
                    분류기나 예측기는 수치형데이터 가능.
                    분류나 예측을 하려면 데이터는 숫자만 가능, 또는 범주형
                    그런데 object로 나오는건 뭐에요? 숫자에요. 그걸 범주형으로 바꿔줘야해요.

3) 초기 모델을 만들고 평가
4) 모델을 GridSearchCV를 이용하여 개선하시오

#
# import xgboost as xgb
#
#
# pdata = pd.read_csv('people.data', header = None)
# ptest = pd.read_csv('people.test', skiprows = 1, header = None)
# # pdata.dropna()
#
# pdata.columns=col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
#               'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
#              'wage_class']
# ptest.shape
# ptest[1:]
# pdata.replace(0, np.nan, inplace= True)
# pdata.replace(' ?', np.nan, inplace= True)
#
#
# from sklearn.preprocessing import Imputer
# imp = Imputer(missing_values = 'NaN', strategy='mean')
# imp.fit_transform(pdata)



#정답

import numpy as np
import pandas as  pdb;
#확장자와는 무관하고 seprator를 어떤 것을 사용했는가 가 기준
# 분류와 예측에서는 문자가 들어왔다면, 범주형입니다.

train_set = pd.read_csv('people.data', header = None)
test_set = pd.read_csv('people.test', skiprows = 1, header = None)
train_set.head()
train_set.info()

col_labels =  ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
             'wage_class']

train_set.columns = col_labels
test_set.columns = col_labels
train_set.info()


train_set.replace(' ?', np.nan).dropna().shape
train_nomissing = train_set.replace(' ?', np.nan).dropna()
test_nomissing = test_set.replace(' ?', np.nan).dropna()

test_nomissing['wage_class'] = test_nomissing.wage_class.replace({' <=50K.': ' <=50K', ' >50K.':' >50K'})
test_nomissing.wage_class.unique()
info 보면 오브젝트타입이 많이 있죠? 그걸 처리해 줍니다.
R에서는 rbind와 같습니다. cbind하려면 axis를 1로 바꿔주면 돼요
combined_set = pd.concat([train_nomissing, test_nomissing], axis = 0)
combined_set.info()
디스크라이브 했더니 열/행이 다 나타나지 않아.
디스크라이브은 ㄴ숫자형 데이터에 ...
안나타났다는 거는 오브젝트타입이다.
다시한번 컨바인드를 봐야해요

combined_set.describe()
combined_set.info()

오브젝트가 있죠?


for feature in test_nomissing.columns:
    if test_nomissing[feature].dtype == 'object':
        print(test_nomissing[feature].unique())

범주형 데이터가 쭉 나타납니다.


카테고리칼 했었죠 범주형으로 바꿔주는 것


for feature in combined_set.columns:
    if combined_set[feature].dtype == 'object':
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes

오브젝트를 카테고리로 바꾸고, 코드스. 범주형에 들어있는 숫자형 으로 바꿔줌

그리고나서 디스크라이브 확인해보면, 전체가 숫자형으로 나타납니다.
combined_set.describe()


그 다음에 뭐해줘야해요?
combined 로 묶어준거 다시 찢어줘야하죠
갯수 어떻게 볼 수 있어요?
판다스니까 shape로 볼 수 있죠.

print( train_nomissing.shape[0]) #행
final_train = combined_set[: train_nomissing.shape[0]]
final_test = combined_set[train_nomissing.shape[0]:]
찢어주고.. 우리가 할려는게 뭐에요?
분류야 회귀야?
분류죠?
분류면 뭐가 있어야해요?
종속변수가 있어야 해요.
나눠야죠
지금 한꺼번에 들어가있으니까


종속변수를 나눠줍니다.

#분류
y_train = final_train.pop('wage_class') # 빼고, 그걸 종속변수로 넣어줍니다.
y_test = final_test.pop('wage_class')


이제 모델 처리하면 돼죠?


from xgboost import XGBClassifier
xgb1 = XGBClassifier( seed=21)
xgb1.fit(final_train, y_train)
result = xgb1.predict(final_test)
print(result)
score = (result == y_test).mean()
print(score)





%% 오늘 한 것만 잘해도 프로젝트할 때 잘 할 수 있습니다.


옵티미제이션 형태..

#GridSearchCV 를 이용한 parameter tuning을 합니다.
cv_params = {'max_depth':[3, 5, 7], 'min_child_weight':[1,3, 5]}
ind_params = {'learning_rate':0.1, 'n_estimators':1000, 'seed':0, 'subsample':0.8,'collsample_bytree':0.8, 'objective': 'reg:linear'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params, scoring='accuracy', cv=5, n_jobs= -1)

optimized_GBM

optimized_GBM.fit(final_train, y_train)

파라미터 튜닝을 어떤 조합이 가장 좋은 결과를 내는지 한 번 해보세요



평가방법 2가지로 알아보세요
첫번째는 scorring확인하는 방법, 두번째는 test데이터로 확인하는 방법. 해보시고,
그 다음에 조합을 만들어서, 어떤 결과가 가장 좋은 결과를 내는지 테스트 해보세요.




#평가방법.
#방법은.. 옵티마이즈 지비엠하면 뭐가 만들어져요? 베스트.. 만들어지죠?

그래서 베스트 에스티메이터를 갖고, 그걸 바로 실행하면 됩니다.

그게 모델이에요
왜냐면 파이썬에선 전부 에스티메이터를 찾는거니까

프레딕트하면 바로 예측 평가할 수 있어야해요

estimator_best = optimized_GBM.best_estimator_




result = estimator_best.predict(final_test)
print(result)
score = (result == y_test).mean()
print(score)



grid_scores_하게 되어지면,
평가한것에 대한 파라메터들 ,파라메터들 조합하고, 위에 accuracy줬기때문에 정확도 평가할 수 있어요.
estimator_best = optimized_GBM.grid_scores_





과제
예측기에 대하여 추가학습이 가능합니다.
완성된 estimator에 추가로 데이터를 학습을 진행하는 방법에 대해서 생각해보고 구현하시오
