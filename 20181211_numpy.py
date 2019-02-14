#strides
import numpy as np
# 1차원으로 데이터 표현
print("스트라이드의 값", np.ones((3,4,5), dtype=np.float64).strides)
# 160 40 8
#     행 열

c = np.full((2,2), 7)
# 2by 2행렬 만들어서 7로 다 채움
print(c)

d = np.eye(4)
print("단위행렬", d)

e = np.random.random((2,2))
# random은 값이 0에서 1사이로 나오게 됨.
print(e)

arr1 = np.array([1,2,3], dtype=np.float64)
print(arr1)



#삼항연산자
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
# zip:  요소끼리 묶어줄 때 사용함.
# zip object 를 만들어 줌. 다시 리스트로 만들어 주어야 함.
print(zip(xarr, yarr, cond))
print(list(zip(xarr, yarr, cond)))

#꺽쇄 안에 for문 이 들어가면, 내장 리스트
result = [(x if c else y) for x,y,c in zip(xarr, yarr, cond)]
print("result = ", result)

result = np.where(cond, xarr, yarr)
print(result)

arr=np.random.randn(4,4) # rand'n' normal
print(arr)
# filter 값 변화
print(np.where(arr>0, 2, -2))
print(np.where(arr>0, 2, arr))



#실습: 다음 데이터를 이용하여 두개 다 참일 때는 0, cond1이 참일 때는 1, cond2가 참일때는 2 그외에는 3으로 처리하는 데이터를 생성해 보시오.
result = []
cond1 = np.array( [1,0,1,1,0,0,1], dtype=bool )
cond2 = np.array( [0,1,1,1,0,1,1], dtype=bool )

#해1
# range => List, arange => ndarray
for i in range(len(cond1)):
    if(cond1[i] and cond2[i]):
        result.append(0) #append 메모리 공간 확보하면서, 동적으로 확장하면서 데이터 추가
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)

print("조건문", result)

#해2
result = np.where(cond1 &cond2, 0, np.where(cond1, 1, np.where(cond2, 2, 3)))
print("3항 연산자", result)

#해3 - 수식( false==0, true==1 결과 )
result =1*(cond1 & ~cond2) + 2 *(cond2 & ~cond1)+ 3* ~(cond1 | cond2) #물결표시는 전체부정.
print( "수식을 이용한", result)



# broadcasting
arr = np.zeros((4,3))
arr[:] = 5
print(arr)
#행 우선인 데이터
col = np.array([1.28, -0.42, 0.44, 1.6])
print(col)
print(col[:, np.newaxis])
arr[:] = col[:, np.newaxis] # [:] 전체 행 지정.
print(arr)

#행에 지정
arr[:2] = [[-1.37], [0.509]]
print(arr)



# 통계함수
a = np.array([[3, 7, 5], [8, 4, 3], [2, 4, 9 ]])
print("범위", np.ptp(a))    #peak to peak : range
print("열범위", np.ptp(a, axis = 1))    # 0 : 행 , 1 : 열
print("분위수", np.percentile(a, 50))    # quantile
print(np.percentile(1, 50, axis=1))
print("행중위수", np.median(a, axis = 0) )
print("열중위수", np.mean(a, axis = 1))

a = np.array([1,2,3,4])
wts=np.array([4,3,2,1])    #가중치  ( 1<- 4 ) ( 2<-  3 ) (  3<- 2 ) (  4<- 1 )
print(np.average(a, weights = wts) )   #가중 평균
x = np.array([7,8,9])
std = np.sqrt(np.mean(abs(x - x.mean()) **2))

print(np.var([1,2,3,4]))
print(np.std([1,2,3,4]))


# matplotlib
# Figures (최상위 객체) + Axes(도화지)
# pyplot

import matplotlib.pyplot as plt
x = [1,2,3,4,5,6]
y = [1,2,3,4,6,5]
# --, -, -., :
plt.plot(x,y, '-.rs') #Linetype , color(b, g, r, c, m, y, k, w) , marker(. , o, s, d, x, +, *)

plt.axis([0,8,0,7]) #x,y 축으로 축범위 지정
plt.xlabel(" X values")
plt.ylabel(" Y values")
plt.title("test")



from numpy.random import randn
# 0,0   0,1   1,0   1,1  축을 공유
# alpha 는 투명도 1~0 , 반투명
fig, axes = plt.subplots(2, 2, sharex = True, sharey= True)
for i in range(2):
    for j in range(2):
        axes[1,j].hist(randn(500), bins=10, color='k', alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)



from matplotlib import pyplot as plt
### sine 그라프
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x) # sin(0) , sin(90)=1, ,sin(180) 0   ,sin(270) = -1
plt.title("sine wave form")
plt.plot(x, y)
plt.show()
#데이터 분할 출력

x = [5, 8, 10]
y = [12, 16, 6]

x2 = [6, 9, 11]
y2 = [6, 15, 7]

plt.bar(x, y, color ='r', align = 'center')
plt.bar(x2, y2, color= 'g', align='center')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('X axis')



a = np.linspace(0, 2*np.pi, 50)
b = np.sin(a)
plt.plot(a,b)
mask = b>= 0
plt.plot(a[mask], b[mask], 'bo')
mask = (b>=0) & (a <=np.pi /2)
plt.plot(a[mask], b[mask], 'go')
plt.show()



from matplotlib import pyplot as plt
import numpy.random as npr
npr.seed(123)
roll = 1.0/6
print([roll]*6)
# 갯수, 확률, 최대 수치
x = npr.multinomial(100, [roll]*6, 10)
x
# 균등분포
x = npr.uniform(-1, 1, (100, 2)) #100행 2열
print(x)
plt.scatter(x[:,0], x[:,1], s=50)ㅇ
plt.axis([-1.05, 1.05, -1.05, 1.05])   #축 범위값



x = np.arange(10)
npr.shuffle(x)
x
x = np.arange(10, 20)
npr.choice(x, 10, replace=False) #비복원추출
npr.choice(x, (5,10), replace=True) #복원추출



# 이미지 출력 matrix를 이미지 heatmap
a = [1, 0, 1, 0,
     1, 0, 1, 0,
     1, 1, 1, 1,
     0, 0, 1, 0 ]
np1 = np.array(a)
plt.imshow(np1.reshape(4,4), cmap='Greys', interpolation='nearest')
plt.show()

x = np.linspace (-100, 100, 1000)
y = np.power(x, 5)
plt.plot(x, y)


A = np.random.random((100, 100))
plt.imshow(A)
plt.hot()
plt.colorbar()
plt.savefig('imageplot.pdf')

#랜덤 normal 정규분포
#loc => 중심점, scale은 표준편차
samp1 = np.random.normal(loc=0., scale=1., size=100)
samp2 = np.random.nromal(loc=1., scale=2., size=100)
samp3 = np.random.normal(loc=0.3, scale=1.2, size=100)
#형태 구성시 subplots 사용
f, ax = plt.subplots(1, 1, figsize=(5,4))
ax.boxplot((samp1, samp2, samp3))
ax.set_xticklabels(['sample 1', 'sample 2', 'sample 3'])
plt.savefig('boxplot.pdf')



from mpl_toolkits.mplot3d import axes3d
ax = plt.subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.1)
# 면이 아닌 선으로 출력
ax.plot_wireframe(X,Y,Z, linewidth=0.1)
plt.savefig('wire.pdf')
plt.show()


#선형대수
# 2X1 + X2  -2x3 = -3
A = np.array([[2,1,-2], [3,0,1], [1,1,-1]])
print(A)
b = np.transpose(np.array([[-3, 5, -2]]))   # y값   행방향 벡터 -> 열방향 벡터
print(b)
x = np.linalg.solve(A,b) #방정식의 해
print(x)
np.dot(np.linalg.inv(A), b)



# 문제
# 벡터 (1,1,1) 와 벡터(1,1,0) 의 각도를 구하시오
# np.dot, np.linalg.norm
# norm = 벡터의 크기 함수(피타고라스값에 의한 크기값)

a = np.array([1,1,1])
b = np.array([1,1,0])
res=np.dot(a,b)

# 내적을 내면, norm    |A||B| cos(theta)
# cos(theta) = 내적 / (|A||B|)
# theta = arccos(내적/ |A||B| )

rad = res / (np.linalg.norm(a) * np.linalg.nrom(b))
print(rad) # 0.816496...

np.arccos(rad)



#방정식의 해를 풀어보시오

x + y + z = 6
2y + 5z = -4
2x + 5y - z = 27

# 답
A = np.array([[1,1,1], [0,2,5], [2,5,-1]])
b = np.transpose(np.array([[6,-4,27]]))
np.dot(np.linalg.inv(A), b)



x = np.array([[1,2], [3,4]])
y = np.linalg.inv(x)
print(x)
print(y)
print(np.dot(x,y))



##행렬곱
import numpy as np
from numpy.linalg import multi_dot
A = np.random.random((10000, 100))
B = np.random.random((100, 1000))
C = np.random.random((1000, 5))
D = np.random.random((5, 333))
#10000 * 333 차원 행렬


print("1번", multi_dot([A, B, C, D]))
print("2번", np.dot(np.dot(np.dot(A, B), C), D))
print("3번", A.dot(B).dot(C).dot(D))
res = A.dot(B).dot(C).dot(D)
print(res.shape)



#역행렬은 정방행렬일때 행렬식이 존재함. 유일한 해를 갖는가?  (행렬식이 0이면,)
#행렬식의 의미는 선형변환시 부피 확대시키는 정도를 표현
#행렬식은 역행렬을 구할 때 사용이 된다.
a = np.array([[1,2], [3,4]])
print(np.linalg.det(a))

b = np.array([[6,1,1],[4,-2,5], [2,8,7]])
print(b)
print(np.linalg.det(b))
print(6 * (-2*7 - 5*8 ) -1 *(4*7 - 5*2) + 1* (4*8-  -2*2))

points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs **2 + ys**2)
plt.imshow(z, cmap=plt.com.gray)
plt.colorbar()
pltt.show()



# lstsq : 최소제곱법 = 수학적으로 선형회귀를 한다는 것과 같음.
import numpy as np
x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
A = np.vstack([x, np.ones(len(x))]).T
A
m,c = np.linalg.lstsq(A, y)[0]
print(m, c)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
plt.show()



import numpy as np # ndarray 데이터를 저장하고 로딩(binary)
a = np.array([1,2,3,4,5])
np.save('outfile', a)
b = np.load('outfile.npy')
print(b)
# text mode 로 저장

import numpy as np
a = np.array([1,2,3,4,5])
np.savetxt('out.txt', a)
b = np.loadtxt('out.txt')
print(b)



import matplotlib
import matplotlib as mpl
import matplotlib.font_manager as font_manager

font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
print(len(font_list))

font_list[:10]
[(f.name, f.fname) for f in matplotlib.font_manager.fontManager.ttflist if 'Nanum' in f.name]



#개별적으로 폰트를 수정하고 싶을 때 사용
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
path = 'C:\\windows\\Fonts\\NanumMyeongjo.ttf'
fontprop=fm.FontProperties(fname=path, size=18)
data = np.random.randint(-100, 100, 50).cumsum()
data
plt.plot(range(50), data, 'r')
plt.title('가격변동 추이', fontproperties=fontprop)
plt.ylabel('가격', fontproperties=fontprop)
plt.show()



import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
#전역적으로 사용하고자 할 때
# rcParams 옵션 지정
plt.rcParams["font.family"] = 'New Gulim'
plt.rcParams['font.size'] = 14.
plt.rcParams['xtick.labelsize'] = 16.
plt.rcParams['ytick.labelsize'] = 16.
plt.rcParams['axes.labelsize'] = 28.

data = np.random.randint(-100, 100, 50).cumsum()
data
plt.title('가격의 변화')
plt.plot(range(50), data, 'r')
plt.show()


import matplotlib as mpl
mpl.rc('font', family="New Gulim")


import matplotlib as mpl
mpl.rc('font', family='New Gulim')
mpl.rc('axes', unicode_minus=False)
X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.title('한글 제목')
plt.plot(X, C, label='코사인')
t = 2*np.pi/3
plt.scatter(t, np.cos(t), 50, color='blue')
plt.xlabel("엑스축 라벨")
plt.ylabel("와이축 라벨")
plt.annotate("여기가 0.5 ! ", xy =(t, np.cos(t)), xycoords='data',
             xytext =(-90, -50),
             textcoords = 'offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->"))
plt.legend(loc=2)
plt.show()



import matplotlib as mpl
plt.title('삼각함수 그라프')
plt.plot(X, C, label="cosine")
plt.plot(X, S, label="sine")
plt.xlabel("pi 값")
plt.legend(loc=3) #위치를 나타내는 loc
plt.show()



### 지수함수
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

#plt.plot(np.exp(x1), 'yo-')
plt.plot(np.exp(-x1), 'yo-')



y1 = np.cos(2*np.pi * x1) *np.exp(-x1)
y2 = np.cos(2*np.pi*x2)

ax1 = plt.subplot(2, 1, 1)  #2행 1열로 나눈 것 중에 1번째
plt.plot(x1, y1, 'yo-')
plt.title('첫번째 subplots')

print(ax1)

ax2 = plt.subplot(2, 1, 2) # 2행 1열로 나눈 것 중에서 2번째
plt.plot(x2, y2, 'r.-')
plt.xlabel('time(s)')

print(ax2)



fig = plt.figure(figsize = (10, 5))    #사이즈 지정
ax1 = fig.add_subplot(121)    #도화지
ax2 = fig.add_subplot(122)

ax1.bar([1,2,3],[3,4,5], color='y')
ax2.barh([0.5, 1, 2.5],[0,1,2])    #horizontal
ax1.axvline(0.65)     # vertical
ax2.axhline(0.45)    # horizontal
plt.tight_layout()   # 가능한  한 공간이 없도록 조정을 해 주어라.
plt.show()



points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs**2 + ys**2)

plt.imshow(z, cmap=plt.cm.rainbow);
plt.colorbar()
plt.title(" $\sqrt{x^2+y^2}$")
plt.show()



## 매쉬그리드를 3차원으로 찍어봄.

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)  #격자 만드는 meshgrid
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

#ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap='hot')



from mpl_toolkits.mplot3d import Axes3D
def f(x,y) : return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(f(X,Y))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
plt.show()

#등고선을 몇단계로 8단계로 구분!
# f: fill 색을 채워서 출력
plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap='jet')
C = plt.contour(X, Y, f(X,Y), 8, colors='black', linewidth=.5)
plt.show()

#
# 점선은 내려간 것
# 실선은 양수를 나타냄
#



# 랜덤을 정리하겠습니다.
import random
print("random 1 : ", random.random())  # 0에서 1사이
print("random.uniform 1 : ", random.uniform(3,4)) # uniform 3-4
print("random.gauss 1: ", random.gauss(1, 1.0)) # 가우시안은 뭐뭐가 필요해 ? 정규분포.이니까, 1이 뭐야 중심. 표준편차.
print("random.randrange 1 : ", random.randrange(20)) #
print("random.randint 1 : ", random.randint(1, 10)) #1부터 10미만까지의 값

list = [3, 50, 20, 2, 100, 2005, 35, 94, 88, 22, 999]
print("random.choice :", random.choice(list))  #랜덤으로 골라주는데, 가중치 를 지원함.

list = [a for a in range(10)]
print("list: ", list)
random.shuffle(list)
print("list :" , list)
print("random.sample:" , random.sample(list,3))




# pandas

# numpy로 개발된 고수준 자료구조
#pannel data(계량경제학의 구조화된 데이터)
#데이터의 색인과 게층적 색인 + 기술 통계 지원
# seroies , dataframe, pannel 로 구성
# series 는 동질적인 자룍구조  / 데이터
# data frame은 열간 이질적
# 기본요소는 series 로 구성된다.

import numpy as np
import pandas as pd
data = np.array(['a','b','c','d']) #ndarray

s = pd.Series(data) #인덱스를 이용해서 고속으로 접근하기 위함

print('시리즈 데이터', s.values)
print('시리즈 인덱스', s.index)
print('시리즈 초기화', s)
print(s[0])



#시리즈 초기화 : 키이 + 데이터 형식

data = {'a' : 0., 'b' : 1., 'c':2.}
s = pd.Series(data)
print(s['a'])
s = pd.Series(data, index=['b', 'c', 'd', 'a'])
print(s['a'])
print(s['d'])   # nan
s = pd.Series([1,2,3,4,5], index=['a','b','c','d','e'])
print("인덱스에 의한 출력", s[0])  #배열식 접근도 가능 !  기본이 array이기 때문에
print(s['a']) #인덱싱에 의한 접근도 가능하다 !
print(s['e'])



#판다스는, 배열임에도 불구하고 '추가' 동적 메모리 할당이 가능하면서도, 고속 데이터 저장 가능함 '인덱스'
# 멀티코아 가 접근 가능하다. 그렇기 때문에 작업이 빠르다.
# 배열 이라고 하면, 기본적으로 메모리가 연속적으로 사용된다는 것이고, 프로그램이 로딩할때, 메모리가 fixed. 됨.
s['a']=100
print(s['a'])
s['f']=10
print(s['f'])



print("filtering에 의한 출력", s[s>4]) # boolean indexing
print(s)
print(s*2)  #vertorrizing 되어서, 각 요소에 2 를 곱할 수 있다.
print(s)



#dictionary를 이용한 초기화
sdata = {'Ohio':35000, 'Texas':71000, 'Oregon':16000, 'Utah':5000}
obj3 = pd.Series(sdata)
print(obj3)
states=['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
print(obj4)
print("시리즈 인덱스", obj3.index)
print("시리즈 인덱스", obj4.index)
print(pd.isnull(obj4))
print(pd.notnull(obj4))
print("obj3을 출력합니다.", obj3)
print("obj4를 출력합니다.", obj4)
print("연산결과를 출력합니다.", obj3+ obj4)  #연산시 양쪽 데이터가 존재해야 함.
