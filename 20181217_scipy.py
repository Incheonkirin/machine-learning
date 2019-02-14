import numpy as np
# miscellaneous
from scipy.misc import imread, imresize # 이미지로딩, 사이즈변경 함수
import matplotlib.pyplot as plt
from scipy import misc # 기본이미지
from scipy.misc import face
from skimage.data import coins # skimage : scikit이미지
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

import pandas as pd, numpy as np

img = face()
print(img)
# 이미지 픽셀, 배열 행, 열
print(img.shape) #(768, 1024, 3)  3차원 ndarray
#                  R     G    B
img_tinted = face() * [1, 0.95, 0.9] # filter 적용 #broadcasting되면서, 전체 데이터를 [1, 0.95, 0.9]로 바꿔줍니다.
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(np.uint8(img_tinted)) # 색을 좀 뺀거지.. 0.95, 0.9로 맞춰서
plt.show()
f = misc.face(gray=True)
misc.imsave('face.png', f)

byte = 768*1024*3 # 3이 바이트 단위
kilo = byte/1024
mega = kilo/1024
print(mega, "메가")

face = misc.imread('face.png') # ndarray 이미지는 숫자 배열
print(type(face))
print(face.shape, face.dtype)
face[1,1,1]  ### 면, 행 , 열 3차원
face[767,1023,2]

# grayscale
f = misc.face(gray=True)
print(f.shape, f.dtype)
plt.imshow(f, cmap=plt.cm.gray)
plt.show()
plt.axis('off')

plt.contour(f, [50, 200])
# 일부 화면 확대
plt.show()

plt.imshow(f[320:340, 510:530], cmap=plt.cm.gray, interpolation='nearest')
plt.show()

# 원형 마스킹
face = misc.face(gray=True)
face[0, 40]

plt.imshow(face)
plt.show()

face[10:13, 20:23]
face[100:120] = 255
lx, ly = face.shape

X, Y = np.ogrid[0:lx, 0:ly]  # "meshgrid"
plt.imshow(face)

# 원의 방정식 circle = X^2 + Y^2
# lx * ly / 4 = 반지름
mask = (X - lx /2 ) ** 2 + ( Y - ly / 2 ) ** 2 > lx * ly / 4
#        X ^2                 Y^2              >  상수

face[mask] = 0 #컬러에서 0 은 black이다.

face[range(400), range(400)] = 255
plt.imshow(face)

# ndimage 필터 적용
# 정규 분포를 사용 타원을 지원 # 정규분포 고려한다= 가까운 곳 강한 영향, 먼곳 적은 영향
# 주변값을 고려해서 컬러값을 결정
# blurred face 잡티를 제거함.
# 가우시안 필터 = 정규분포 = 주변 큰 영향, 멀리 적은 영향
# 이미지:CNN(convolution neural network)   이미지 특성을 잘 알고 있어야 합니다.
# 사운드, text mining

from scipy import ndimage
face = misc.face()
blurred_face = ndimage.gaussian_filter(face, sigma=3) #주변값을 고려한다 !

plt.imshow(blurred_face)
plt.show()

im = np.zeros((256, 256))
im[64:-64, 64:-64] = 1
im[90:-90, 90:-90] = 2
plt.imshow(im)
im = ndimage.gaussian_filter(im, 8)


plt.imshow(im)
plt.show()


#이미지에 노이즈를 넣어보시오.
import scipy
f = scipy.misc.face(gray=True)
noisy = f + 0.4*f.std()*np.random.random(f.shape)
gauss_denoised = ndimage.gaussian_filter(noisy, 10)
med_denoised = ndimage.median_filter(noisy, 2) #노이즈를 제거하는 작업

plt.figure(figsize=(12, 2.8))
plt.subplot(131)
plt.imshow(noisy, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')

plt.subplot(132)
plt.imshow(gauss_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')
plt.subplot(133)
plt.imshow(med_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')

plt.subplots_adjust(wspace=0.02, hspace=0.02, top= 0.9, left=0, right=1)
plt.show()



from PIL import Image
img = Image.open("image007.jpg")
plt.imshow(img)
print(type(img))
plt.show()


import matplotlib.pyplot as plt

from scipy import stats
import seaborn as sns #matplotlib wrapper해서 출력
tips = sns.load_dataset('tips')
tips
# f: figure
f, (ax1, ax2) = plt.subplots(1, 2,
figsize=(12, 5),
sharey=True)

print(tips.describe())
#분포를 출력할 때 사용

sns.distplot(tips.total_bill,
ax=ax1,
hist=True)

sns.distplot(tips.total_bill,
ax=ax2,
hist=False,
kde=True,
rug=True,
fit=stats.gamma,   # kws keyword
fit_kws=dict(label='gamma'),   # 옵션지정
kde_kws=dict(label='kde'))

ax.legend()
plt.show()


# 회귀선출력 (regression plot)
sns.regplot(x = "total_bill", y="tip", data=tips)


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10), sharey=True)
# boxplot과 같은 역할 ( + 데이터 분포까지 고려해서 출력)
sns.barplot(x='sex', y='tip', data=tips, ax=ax1)
sns.violinplot(x='sex', y='tip', data=tips, ax=ax2)
sns.swarmplot(x='sex', y='tip', data=tips, ax=ax3)


#화면 분할 - 범주형 변수에 대해서 고려해서 출력
g = sns.FacetGrid(tips, col='smoker', row='sex')
g.map(sns.regplot, 'total_bill', 'tip')


#lmplot = regplot + facetGrid 범주형데이터를 고려해서 출력
#R의 ggplot의 개념으로 매개변수를 지정
import pandas as pd
import matplotlib as mpl
mpl.style.use('ggplot')
df = pd.read_csv('Pokemon.csv', index_col=0)
df.head()
sns.lmplot(x='Attack', y='Defense', data=df)
# hue= 'stage' 범주형 데이터를 고려해서 출력
sns.lmplot(x='Attack', y='Defense', data=df,
fit_reg=True,hue='Stage')
sns.lmplot(x='Attack', y='Defense', data=df,
fit_reg=False,hue='Stage')

sns.boxplot(data=df) #비교가 불규칙
stats_df = df.drop(['Total', 'Stage', 'Legendary'], axis=1)
print(stats_df)
sns.boxplot(data=stats_df)


# heatmap
corr = stats_df.corr()
print(corr)
sns.heatmap(corr)


# kind = scatter, reg, kde, hex옵션이 가능.
sns.jointplot(x='Attack', y='Defense', kind='scatter', data=df)

sns.jointplot(x='Attack', y='Defense',
marginal_kws=dict(bins=15, rug=True), kind="reg", data=df)


#상관도 출력 #버전마다 다름(출력 모양)
import seaborn as sns; sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g = sns.pairplot(iris)
g = sns.pairplot(iris, hue="species")
g = sns.pairplot(iris, hue="species", palette='husl')
# markers = ["o", "s", "D"]  옵션을 주면, 바뀝니다.


current_palette = sns.color_palette()
sns.palplot(current_palette)
sns.palplot(sns.color_palette("hls", 8))
sns.palplot(sns.color_palette("husl", 8))


#잔차그래프
rs = np.random.RandomState(7)
x = rs.normal(2, 1, 75)
y = 2 + 1.5*x + rs.normal(0, 2, 75)
sns.residplot(x, y, lowess=True, color="g")


import altair as alt
alt.renderers.enable('default')
from altair import Chart, X, Y, Axis, SortField
alt.__version__
import altair as alt
print(alt.renderers.active)
import altair as alt
alt.renderers.enable('default')

budget = pd.read_csv("https://github.com/chris1610/pbpython/raw/master/data/mn-budget-detail-2014.csv")
budget.head()
# %matplotlib inline
budget_top_10 = budget.sort_values(by='amount',ascending=False)[:10]
# alt.renderers.enable('notebook')
Chart(budget_top_10).mark_bar().encode(x='detail', y='amount')
plt.show()


import altair as alt
from vega_datasets import data

iris = data.iris()

alt.Chart(iris).mark_point().encode(
    x='petalLength',
    y='petalWidth',
    color='species'
)


# scipy.optimize
#방정식이 의미하는 것 ? x 와 y의 함수적 관계
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
# 3차원으로 fitting 해서 z값을
z = np.polyfit(x, y, 3) #다항식 (3차원) . 다항식 계수를 생성하시오.
print(z)
p= np.poly1d(z) #방정식 생성
print(p)
# Y = 0.08703704 X^3 + -0.81349206 X^2  + 1.69312169 X -0.03968254
p(0.5)
p(3.5)
p(10)

p30 = np.poly1d(np.polyfit(x, y, 30)) #과적합 30차원
#3승으로, 5승으로 한번 해보세요..
# p30 = np.poly1d(np.polyfit(x, y, 5)) #과적합 30차원
# p30 = np.poly1d(np.polyfit(x, y, 3)) #과적합 30차원

p30(4)
import matplotlib.pyplot as plt
xp = np.linspace(-2, 6, 100)
_ = plt.plot(x,y, '.', xp, p(xp), '-', xp, p30(xp), '--')
plt.ylim(-2, 2)
plt.show()




def func(x, a, b, c):
    return a * np.exp(-b * x) + c
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')
#비선형 최소제곱법에 의해서 피팅
popt, pcov = curve_fit(func, xdata, ydata)
popt
plt.plot(xdata, ydata, 'b-', label='data')
plt.plot(xdata, func(xdata, *popt), 'r-',
label='fit:a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.show()


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
def fm(p):
    x, y = p
    return (np.sin(x) + 0.05*x**2   # sin그라프와 x**2(밖으로 나가면서)으로 결합,
    + np.sin(y) + 0.05*y**2)

x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)

X, Y = np.meshgrid(x, y)
Z = fm((X, Y))

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm,
linewidth=0.5, antialiased=True)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
# 지금은 이해하는 거니까, 그림을 보고 이해하는 정도만 해도 괜찮음.
# 이걸 우리가 직접 할 일은 없음.

# 그래프가 나왔음?  이 그래프의 최저점을 찾아야 함. 어떻게 찾느냐?

# 이걸 optimization 해보겠음. 아까는 fitting해서 구현했지만, 이건 안되는 것.
# 그래서

# brute force 무차별 공격 : 패스워드 조합
import scipy.optimize as spo
def fo(p):
    x, y = p
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y **2
    if output == True:
        print('%8.4f %8.4f %8.4f' % (x, y, z))
    return z

output = True
#값의 조합 생성해서 대입
opt1 = spo.brute(fo, ((-10, 10.1, 5), (-10, 10.1, 5)), finish=None)
print(opt1)

output = False #값이 너무 많이 나오니까 False로 둡니다.
# output = True

# 제일 작은 조합을 출력
opt1 = spo.brute(fo, ((-10, 10.1,0.1), (-10, 10.1, 0.1)), finish=None)
print(opt1)
# 제일 작은 값이 나옵니다.


#도함수가 없는 비선형 최적화에 사용.
# brute force + function minimization ( fmin )
# 이해만 하면 돼요. 써먹을 데가 있을지.
# 할강단체법
output = True
# tolerance : 공차 ( 어느정도 에러를 허용할 것 인가 0.01이 넘어가면 허용하는 것으로)
# function tolerance
#횟수 제한
#                위에서 대충 찾은 놈 (초기값) ; opt1 : spo.brute 주면, 정밀하게 (최소점) 찾아줌
opt2 = spo.fmin(fo, opt1, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20)
print(opt2) #더 정확하게 나옵니다.
fm(opt2)
output = False
spo.fmin(fo, (2.0, 2.0), maxiter=250)


#변수가 하나인 경우에 최적해를 구하는 방법
def f(r):
    return 2 * np.pi * r**2 + 2/r
r_min = spo.brent(f, brack=(0.1, 4))
print("최소점", f(r_min))
r = np.linspace(0, 2, 100)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(r, f(r), lw=2, color='b')
ax.plot(r_min, f(r_min), 'r*', markersize=15)
ax.set_title(r"$f(r) = 2\pi r^2+2/r$", fontsize=18)
ax.set_xlabel(r"$r$", fontsize=18)
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.set_ylim(0.30)

fig.tight_layout()
plt.show()


# interpolate(보간법)
from scipy.interpolate import *
def f(x):
    return np.sin(x)

n = np.arange(0, 10)
x = np.linspace(0, 9, 100)
#노이즈가 낀 데이터

y_meas = f(n) + 0.1 * np.random.randn(len(n))
# 함수에 의한 데이터
y_real = f(x)
# interp1d - 노이즈를 잡아줌
linear_interpolation = interp1d(n, y_meas)
y_interp1 = linear_interpolation(x)

cubic_interpolation = interp1d(n, y_meas, kind='cubic')
y_interp2 = cubic_interpolation(x)
fig, ax = plt.subplots(figsize = (10, 4))
ax.plot(n, y_meas, 'bs', label='noisy data')
ax.plot(x, y_real, 'k', lw=2, label='true function')
ax.plot(x, y_interp1, 'r', label='linear interp')
ax.plot(x, y_interp2, 'g', label='cubic interp')
ax.legend(loc=3);
plt.show()



from scipy.interpolate import interp1d

x = np.arange(0, 10)
y = np.array([3.0, -4.0, -2.0, -1.0, 3.0, 6.0, 10.0, 8.0, 12.0, 20.0])
f = interp1d(x, y, kind= 'cubic')
# xint = 3.5
# yint = f(xint)
# plt.plot(x, y, 'o', c='b')
# plt.plot(xint, yint, 's', c='r')
# plt.show()


xint = np.arange(0, 9.01, 0.01)
yint = f(xint)
plt.plot(x, y, 'o', c='b')
plt.plot(xint, yint, '-r')
plt.show()





# 문제
# (1, 3), (2, -2), (3, -5), (4, 0) 을 지나가는 3차 다항식을 구하라.


from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4])
y = np.array([3, -2, -5, 0])
f = interp1d(x, y, kind='cubic')
xint = np.arange(1, 4, 0.01)
yint = f(xint)
plt.plot(x, y, 'o', c = 'b')
plt.plot(xint, yint, '-r')
plt.show()


# differentials (미분)
def f(x): # 꼭지점 2
    return x**3 - 3*x**2 + x

# 한 번 미분하면
# 3x**2 - 6x + 1

# 한 번 더 미분하면
# 6x - 6

def fprime(x): # 포물선 기울기
    return 3*x**2 - 6*x + 1

def fprime2(x):# 직선
    return 6*x - 6


x = np.linspace(-1, 3, 400)
plt.figure(figsize=(10, 15))

plt.subplot(311)
plt.plot(x, f(x))
plt.xlim(-2, 4)
plt.xticks(np.arange(-1, 4))
plt.yticks(np.arange(-5, 4))
plt.title('f(x)')

plt.subplot(312)
plt.plot(x, fprime(x))
plt.xlim(-2, 4)
plt.xticks(np.arange(-1, 4))
plt.yticks(np.arange(-3, 11))
plt.title("f'(x)")

plt.subplot(313)
plt.plot(x, fprime2(x))
plt.xlim(-2, 4)
plt.xticks(np.arange(-1, 4))
plt.yticks(np.arange(-1, 4))
plt.title("f'(x)")

plt.show()



# sympy 기호 연산
from sympy import *
init_printing()
x = Symbol('x')
( pi + x ) ** 2
pi.evalf (n=50)
y = ( x + pi ) ** 2
N ( y, 5 )
(x+1)*(x+2)*(x+3)
expand((x+1)*(x+2)*(x+3))
print(expand((x+1)*(x+2)*(x+3)))

x, y, z = symbols("x, y, z") #x, y, z 변수 지정
f = sin(x*y) + cos(y*z)
diff(f, x, 1, y, 2) #미분
print(diff(f, x, 1, y, 2))

f
integrate(f, x)
print(integrate(f, x))


integrate(f, (x, -1, 1))
print(integrate(f, (x, -1, 1)))
n = Symbol("n")
Sum(1/n**2, (n, 1, 10))
#정밀도 자리수 지정
Sum(1/n**2, (n, 1, 10)).evalf(50)
Sum(1/n**2, (n, 1, oo)).evalf()
Product(n, (n, 1, 10))

print(Product(n, (n, 1, 10)))


series(exp(x), x)
# 매트릭스
m11, m12, m21, m22 = symbols("m11, m12, m21, m22")
b1, b2 = symbols("b1, b2")
A = Matrix([[m11, m12], [m21, m22]])
A
b = Matrix([[b1], [b2]])
b
A**2
A * b
A.det()
A.inv()


# 편미분
import sympy
x, y = sympy.symbols('x y')
f = x ** 2 + x * y + y ** 2
f

sympy.diff(f, x) # x를 변수로 보고, y는 상수 취급

sympy.diff(f, y) # y를 변수로 보고

# 정규분포식
x, mu, sigma = sympy.symbols('x mu sigma')
f = sympy.exp((x - mu) ** 2 / sigma **2)
f
sympy.simplify(sympy.diff(f, x))
# 적분

x, y = sympy.symbols('x y')
f = 2 * x + y
f

sympy.integrate(f, x) # x에 대해서 적분






import numpy as np
import sympy
sympy.init_printing()
from scipy import optimize
r, h = sympy.symbols("r, h") #기호 지정
# sympy.pi : 3.14
# 원의 면적과 둘레 : 원기둥의 면적
Area = 2 * sympy.pi * r**2 + 2*sympy.pi*r*h
Volume = sympy.pi * r**2 *h # 체적
h_r = sympy.solve(Volume -1)[0]
Area_r = Area.subs(h_r)
rsol=sympy.solve(Area_r.diff(r))[0]
rsol


# 적분
import scipy.integrate as sci
import numpy as np
def f(x):
    return np.sin(x) + 0.5 * x
a = 0.5
b = 9.5
x  = np.linspace(0, 10)
y = f(x)

from matplotlib.patches import Polygon
fig, ax = plt.subplots(figsize=(7, 5))
plt.plot(x, y, 'b', linewidth=2)
plt.ylim(ymin=0)

Ix = np.linspace(a, b)
Iy = f(Ix)
verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)]
# LATEX 기호는 $와 $사이에 넣고,  a와 b사이에 ^표시는 적분표시.
poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
ax.add_patch(poly)
plt.text(0.75 * (a + b), 1.5, r"$\int_a^b f(x)dx$",
horizontalalignment='center', fontsize=20)

plt.show()

# 적분
sci.fixed_quad(f, a, b)[0]
sci.quad(f, a, b)[0]
sci.romberg(f, a, b)
xi = np.linspace(0.5, 9.5, 25)
sci.trapz(f(xi), xi)
# simulation


for i in range(1, 20):
    np.random.seed(1000)
    x = np.random.random(i*10) * (b-a) + a
    print(np.sum(f(x)) / len(x) * (b - a))



# scipy stats_df
from scipy import stats
import scipy as sp
n, min_max, mean, var, skew, kurt = stats.describe(s)
print("Number of elements: {0:d}".format(n))
print("Minimum: {0:8.6f} Maximum: {1:8.6f}".format(min_max[0], min_max[1]))
print("Mean: {0:8.6f}".format(mean))
print("Variance: { 0:8.6f}".format(var))
print("Skew: {0:8.6f}".format(skew))
print("Kurtosis : {0:8.6f}".format(kurt))
stats.norm.cdf(-1)
stats.norm.cdf(2)
stats.norm.cdf(1)-stats.norm.cdf(-1) # 68%
1 - stats.norm.cdf(2)
stats.norm.ppf(.95) #확률값에 대한 표준편차의 범위수.
stats.norm.cdf(2)




# 독립 샘플에 대한 평균 비교 two independent samples of scores
dt = np.array([24, 43, 58, 71, 43, 49, 61, 44, 67, 49, 53, 56, 59, 52, 62, 54, 57, 33, 46, 43, 57])
dc = np.array([42, 43, 55, 26, 62, 37, 33, 41, 19, 54, 20, 85, 46, 10, 17, 60, 53, 42, 37, 42, 55, 28, 48])
print(stats.ttest_ind(dt,dc))
# pvalue=0.02
print(stats.ttest_ind(dt, dc, equal_var=False)) #등분산성이 아닌 경우
print(stats.jarque_bera(dt)) #정규분포와 일치하는가
print(stats.jarque_bera(dc))
print(stats.pearsonr(dt, dc)) #피어슨 상관계수
print(stats.spearmanr(dt,dc))
print(stat.kendalltau(dt,dc))


import matplotlib.pyplot as plt

read_file = pd.read_csv('play_13_14_top30.csv', skiprows=1)
read_file.describe()
read_file.head()

a = read_file.describe()
a.boxplot()
plt.show()

re_file = read_file.rename(columns={'P': 'points', 'G':'goals',
'A': 'Assists', 'S%':'shooting_percentage',
'Shift/GP1': 'shifts_per_game_played'})


G = re_file[['goals', 'Assists', 'points']]
t = G.corr(method='pearson')
t

import seaborn as sns
sns.heatmap(t)
plt.show()

#문제 re_file중 S열에 대해서 분포도를 출력해 보시오
a1 = list(re_file['S'])
plt.plot(a1, 'ro')
plt.show()



#문제 전체 변수들에 대한 상관계수를 heatmap으로 출력하시오
aa = re_file.corr(method="pearson")
sns.heatmap(aa)
plt.show()


#문제 {"Player", "points", "goals", "Assists"} 컬럼만 선택하고,
#이 중에 points가 75보다 크고 80보다 작거나 같은 데이터를 추출하시오

def select_player(x):
    a = pd.DataFrame(x, columns = {"Player", "points", "goals", "Assists"})
    return a


def point_75(x):
    a = x.where(x['points'] > 75)
    a = a.where(a['points'] <= 80)
    a = a.dropna()
    return select_player(a)
point_75(re_file)



#문제 goals > 40보다 크고 points가 80보다 작은 데이터를 추출하시오
def gp(x):
    a = x.where(x['goals']>40)
    a = a.where(a['points']<80)
    a = a.dropna()
    return select_player(a)
gp(re_file)



import statsmodels.api as sm
import statsmodels.formula.api as smf

re_ols = smf.ols('Assists ~ goals', data =re_file).fit()
print(re_ols.summary())



#예측 형태
aa = pd.DataFrame({'goals':[50, 70, 40, 20]})
re_ols.predict(aa)

#정규성 테스트 shapiro 귀무가설 : 정규성을 만족한다.

import scipy.stats as stat
def shairo_test(x):
    test = ['points', 'goals', 'Assists']
    for i in test:
        c = stat.shapiro(x[i])
        print(c)
        if c[1]>= 0.05:
            print(i, '정규성을 만족')
        else:
            print('x')

shairo_test(re_file)
anova_test2 = stat.barlett(re_file['points'], re_file['goals'], refile['Assists'])
print(anova_test2)

anova_test1 = stat.ttest_ind(re_file['points'], re_file['goals'], equal_var=True)
print(anova_test1)
