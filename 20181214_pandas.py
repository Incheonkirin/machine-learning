#한글 깨짐 현상
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

import pandas as pd, numpy as np

def convert_currency(val):
    new_val = val.replace(',','').replace('$','')
    return float(new_val)
def convert_percent(val):
    new_val = val.replace('%', '')
    return float(new_val) / 100

#pd.to_numeric의 매개변수 errors, downcast
df_2 = pd.read_csv("sales.csv",
dtype ={"Customer Number":'int'},
converters={'2016':convert_currency,
'2017':convert_currency,
'Percent Growth': convert_percent,
'Jan Units': lambda x : pd.to_numeric(x, errors='coerce'),
'Active': lambda x : np.where(x == "Y", True, False)
})

print(df_2)



# 문제
# fire_station 데이터를 로딩한 후, 다음 문제를 해결하시오

fire = pd.read_csv("fire_station.csv")
print(fire)
print(list(fire))
print(fire['Road Ramp'])

# 문제1 : 불평의 종류(중복되지 않는) 와 불평의 개수를 출력하시오
print(fire['Complaint Type'])
f=(pd.crosstab(index=fire['Complaint Type'], columns="count"))
print(f)
 value_counts()


# 불평의 상위 10개를 출력하시오
f=(f.sort_values(by=["count"],ascending=False))
print(f.head(10))

# 불평의 상위 10개를 bar 형태로 출력하시오
plot(kind='bar')

# 거주지별 불평의 개수를 확인하시오
print(list(fire))
# "Borough"

# 불평의 종류별로 카운트하시오
# 불평의 종류별로 카운트한 내용을 kde로 출력하시오
# 소음 불평이 많은 지역을 확인하시오
# "Noise-Street/Sidewalk"
# 거주비별로 불평의 개수를 카운트하시오
# 소음 불평이 전체 불평에서 차지하는 비율을 출력하시오




import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial.distance import euclidean
# %matplotlib inline
x = np.linspace(0, 50, 100)
# #              진폭          주기
ts1 = pd.Series(3.1* np.sin(x/1.5)+3.5)
ts2 = pd.Series(2.2* np.sin(x/3.5+2.4)+3.2)
ts3 = pd.Series(0.04* x+3.0)
#
ts1.plot()
ts2.plot()
ts3.plot()

plt.ylim(-2, 10)
plt.legend(['ts1', 'ts2', 'ts3'])
plt.show()
def euclid_dist(t1, t2):
    return np.sqrt(sum((t1-t2)**2))

print(euclid_dist(ts1, ts2))
print(euclid_dist(ts1, ts3))



def DTWDistance(s1, s2):
    DTW={}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')    # inf 를 주는 이유?  아래 min 할때 0값이 선택되기 위해서,  이산적으로 데이터 들어와 있기 때문에, 양자화가 끝난데이터. for문 2개돌아가면서 차의 제곱으로 거리값을 잰다. 그 거리값에  i-1 j-1  두개의 거리값을 구한다음에 데이터 중에 가장 작은 것을 선택해서 더해준다. 그러면, 모든 위치에서 누적 거리값이 계산되어진다.
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i] -s2[j]) **2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)],
            DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

print(DTWDistance(ts1, ts2))
print(DTWDistance(ts1, ts3))
print(DTWDistance(ts2, ts3))


# fastdtw를 이용하면 간단히 구할 수 있습니다.
from fastdtw import fastdtw
fastdtw(ts1, ts2, dist=euclidean)
fastdtw(ts1, ts3, dist=euclidean)
fastdtw(ts2, ts3, dist=euclidean)

# 시간 관리, 최근에 많이 쓰이는 패키지입니다.
# pytz
import numpy as np
import pandas as pd

import pytz
print(pytz.common_timezones[-5:])
tz=pytz.timezone('US/Eastern')
print(tz)

# # Q-JAN : 분기별 첫달의 마지막 날을 기준으로
# # Q-DEC : 분기별로 마지막 날짜를 기준으로
rng=pd.date_range('3/9/2012 9:30', periods=6, freq='Q-DEC')
# rng=pd.date_range('3/9/2012 9:30', periods=6, freq='Q-JAN')


ts=pd.Series(np.random.randn(len(rng)), index=rng)
print(ts.index)
# 시간UTC : 국제 표준시
# GMT 그리니치 천문대 를 기준으로 한 시간
# 시간 문자열은 iso 8601 표준


print(pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC'))
ts_utc=ts.tz_localize('UTC')
print(ts_utc.index)
print(ts_utc.tz_convert('US/Eastern'))
print(ts.index.tz_localize('Asia/Seoul'))
#국제 표준시를 localize가능


# 앞에 A가 붙으면, Annuel 이라는 의미입니다. 회계기간지정시 사용합니다.
p = pd.Period('2012', freq='A-JUN')
print('기간',p) #2012
print(p.asfreq('M', how='start')) #2011-07
print(p.asfreq('M', how='end')) #2012-06 종료점
print(p.asfreq('D', how='start'))
print(p.asfreq('D', how='end'))
print(p)
print(p+5) #5년 후를 나타냅니다.
print(p-2)


from datetime import timedelta
start = datetime(2011, 1, 7)
start + timedelta(12) #day


from pandas import Series, DataFrame
from datetime import datetime
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)
ts
print(type(ts))
print(ts.index)
print(ts.index.dtype) # datetime64[ns] 나노초까지 표현
ts['1/10/2011']
ts['20110110']
#날짜로 된 인덱스의 이점
longer_ts = Series(np.random.randn(1000),
index=pd.date_range('1/1/2000', periods=1000))


longer_ts
longer_ts['2001']
longer_ts['2001-05']
ts[datetime(2011, 1, 7):]
ts['1/6/2011':'1/11/2011']
ts.truncate(after='1/9/2011') #이후 날짜는 잘라서 없애 줌.


# shift 날짜하고 데이터 일치시키기 위하여
ts = pd.Series(np.random.randn(4))
index=pd.date_range('1/1/2000', periods=4, freq='M'))

ts
ts= ts.shift(2)
ts.shift(-2)
ts.shift(2, freq='M')
ts.shift(3, freq='D')
ts


# diff
df = pd.DataFrame({'a':[1,2,3,4,5,6],
'b':[1,1,2,3,5,8],
'c':[1,4,9,16,25,36]})

print(df.diff())
df.diff(axis=1) # 맨 처음것은 뺄 게 없음.
df.diff(periods=3) #앞의 데이터 3개 없음
df.diff(periods=-1) #역순으로 / 맨 마지막 것은 뺄 게 없음


# resample                                     분단위 T
rng = pd.date_range('1/1/2014', periods=12, freq='T')
ts=pd.Series(np.arange(12), index=rng)
print(ts)
print(ts.resample('5min'))
print(ts.resample('5min').ohlc())
#open 시작가 high 고가 low 저가 close 종가


print("5분", ts.resample('5min').sum()) #합을 구해주는데, 인덱스를 보면, resample한 것을 보여줌.




index = pd.date_range('1/1/2000', periods=9, freq='T')
series = pd.Series(range(9), index=index)
series
# 좁히고
print(series.resample('3T').sum())
# 오른쪽을 기준으로 하기 때문에 0이 없음.
print(series.resample('3T', label='right').sum()) #맨 마지막 데이터 사용
print(series.resample('3T', label='right', closed='right').sum())
#30초 단위로 데이터를 늘리고(부족)
print(series.resample('30S').asfreq()[0:5]) #NaN
print(series.resample('30S').pad()[0:5]) #앞의 것을 뒤로 채우고
print(series.resample('30S').bfill()[0:5]) #뒤의 것을 앞으로 채우고.





# 날짜 파싱
close_px = pd.read_csv('stock_px.csv', parse_dates=True, index_col=0)
volume=pd.read_csv('volume.csv', parse_dates=True, index_col=0)
prices=close_px.loc['2011-09-05':'2011-09-14',['AAPL', 'JNJ', 'SPX', 'XOM']]
volume=volume.loc['2011-09-05':'2011-09-12', ['AAPL','JNJ','XOM']]
print(prices)
print(volume)
print(prices * volume)


# #두개의 데이터 인덱스 불일치 할 때
gdp = pd.Series([1.78, 1.98, 2.08, 2.01, 2.15, 2.31, 2.46],
index=pd.period_range('1984Q2', periods=7, freq='Q-SEP'))
infl = pd.Series([0.025, 0.045, 0.037, 0.04],
index=pd.period_range('1982', periods=4, freq='A-DEC'))

print(gdp)
infl
infl_q = infl.asfreq('Q-SEP', how='end')
print(infl_q.reindex(gdp.index, method='ffill')) #  날짜는 일치시키고.. 없는 데이터는 채워나가면서..
#reindex에 대해서 설명함.


#boxcar
#triang
#blackman
#hamming
#bartlett
#parzen
#bohman
#blackmanharris
#buttall
#barthann
#kaiser(needs beta)
#gaussian(needs std)

#general_gaussian(needs power, width)
#slepian(needs width)


# rolling
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
print(df)
df.rolling(2, win_type='triang').sum()
print(df.rolling(2, win_type='triang').sum())

import pandas as DatetimeIndexResampler
frame=pd.DataFrame(np.random.randn(2,4), index=pd.date_range('1/1/2014',
periods=2, freq='W-WED'), columns=['Colorado', 'Texas', 'New York', 'Ohio'])
close_px_all=pd.read_csv('stock_px.csv', parse_dates=True, index_col=0)
print(close_px_all.head())
close_px=close_px_all[['AAPL', 'MSFT', 'XOM']] #데이터 읽은 것 중 , 3개만.
#휴일을 평일데이터 리샘플링()

close_px=close_px.resample('B').ffill() #없는 데이터를 채워줍니다.
fig, axes= plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(12,7))


#오늘가 는 60일 동안의 영향을 받은 값으로 고주파 > 저주파로 변경 = 완만해져서 추세를 알 수 있게 끔 함.
appl_px=close_px.AAPL['2005':'2009']
ma60 = appl_px.rolling(60, min_periods=50).mean() #60일 이동평균선. 50개가 되어지면, 계산하라
ewma60=appl_px.ewm(span=60) #지수 이동 평균법/ alpha 값을 계산 !  span값이 바로 alpha값이 되는 게 아니고.. 이렇게 계산된다. => 2/(span+1) (단, span>=0)
# alpha 값은 이전 데이터의 중요성을 고려해서 정해지는 지수.
# alpha값이 크다(=현재값을 중시한다)     span값이 커지면, alpha값이 작아잠 ! span값이 작아지면, alpha값이 커짐 !
# decay 영향을 미치는 정도
appl_px.plot(style='k-', ax=axes[0])
ma60.plot(style='k--',ax=axes[1])
plt.show()
#위에서 span 값을 조금씩 변경해서 변화하는 모습을 살펴보세요.


# 일별 변환폭
spx_px = close_px_all['XOM']
spx_rets = spx_px / spx_px.shift(1) - 1    # 변화율을 보기 위해서 (한종목)
returns = close_px.pct_change()  # 세 종목 모두
returns.plot()

# 125일 이동평균선 - 상관계수
corrr = returns.AAPL.rolling( 125, min_periods=100).corr(spx_rets)
corr.plot()
corr=returns.rolling( 125, min_periods=100).corr(spx_rets)
corr.plot()
plt.show()

# 주식데이터 볼때는,
import FinanceDataReader as fdr
# 이거 많이 씀.
fdr.__version__
# 경동보일러 주식

df = fdr.DataReader('267290', '2017')
df.head(10)
df['Close'].plot()

df = fdr.DataReader('AAPL', '2017-01-01', '2018-03-30')
df.tail()
df['Close'].plot()

df = fdr.DataReader('USD/KRW', '1995')
df['Close'].plot()


df = fdr.DataReader('105560', '2017')
df.head()
plt.rcParams["figure.figsize"] = (14,6)
plt.rcParams["axes.grid"] = True
df.unstack(level=0)['Close'].plot(subplots=True)

# 거래량과 주가의 비교
hynix = fdr.DataReader('000660', '2014-01-01', '2018-12-12')
hynix.head(10)
fig = plt.figure(figsize=(12, 8))
# 화면 조절
top_axes = plt.subplot2grid((4,4,), (0,0), rowspan=3, colspan=4)
bottom_axes = plt.subplot2grid((4,4,),(3,0), rowspan=1, colspan=4)
bottom_axes.get_yaxis().get_major_formatter().set_scientific(False)

top_axes.plot(hynix.index, hynix['Close'], label='Adjusted Close')
bottom_axes.plot(hynix.index, hynix['Volume'])

plt.tight_layout()


# 봉차트
import datetime
import matplotlib.ticker as ticker
import mpl_finance as matfin
start = datetime.datetime(2016, 3, 1)
end = datetime.datetimee(2016, 3, 31)
hynix = fdr.DataReader('000660', start, end)
hynix.head(10)
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
day_list=range(len(hynix))
name_list=[]
for day in hynix.index:
    name_list.append(day.strftime('%d'))
ax.xaxis.set_major_locator(ticker.FixedLocator(day_list))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(name_list))
matfin.candlestick2_ohlc(ax, hynix['Open'], hynix['High'], hynix['Low'],
hynix['Close'], width=0.5, colorup='r', colordown='b')
plt.show()

#멀티다운로드
tiker_list = ['AAPL', 'MSFT', 'AMZN']
df_list = [fdr.DataReader(ticker, '2010-01-01', '2016-12-31')['Close']
for ticker in tiker_list]
panel_data = pd.concat(df_list, axis=1)
panel_data.columns = ['AAPL', 'MSFT', 'AMZN']
panel_data = panel_data.dropna()
panel_data.head(10)
close = panel_data
close.head(10)

#원래 데이터/ 20일 이평선 / 100일 이평선
msft = close.loc[:, 'MSFT']
short_rolling_msft = msft.rolling(window=20).mean()
long_rolling_msft = msft.rolling(windw=100).mean()

fig, ax=plt.subplots(figsize=(16,9))

ax.plot(msft.index, msft, label='MSFT')
ax.plot(short_rolling_msft.index, short_rolling_msft, label='20 days rolling')
ax.plot(long_rolling_msft.index, long_rolling_msft, label= '100 days rolling')

ax.set_xlabel('Date')
ax.set_ylabel('closing_price')
ax.legend()

# 20일 이평, 100일 이평, 변화율을
data = pd.read_csv("stock_px.csv", parse_dates=True, index_col=0)
data.head(10)
short_rolling = data.rolling(window=20).mean()
short_rolling.head(25)
long_rolling = data.rolling(window=100).mean()
long_rolling.tail()
returns = data.pct_change(1)
returns.head()

# 정규화 (로그 후 차분)
log_returrns = np.log(data).diff()
log_returns.head()

# 20일 지수 이평선을  기준으로 매매포인트를 설정해서 자동 거래 할 수 있는 기준
import matplotlib.dates as mdates
my_year_month_fmt = mdates.DateFormatter('%m%y')
start_date = '2010-01-02'
end_date = '2018-10-14'
ema_short = data.ewm(span=20, adjust=False).mean()

trading_positions_raw = data - ema_short
trading_positions_raw.tail()

# 차이가 원래 가격과 지수이평선의 차이가 벌어지면, =매매포인트
# event 발생 => 증권 프로그램 에서는 자동으로 사고팔고
# 증권 프로그램(200만원)
# 증권사 OpenAPI (가입후 OpenAPI 키를 받아서 사용)
# 부호가 바뀌는 시점
# momentum, pair, deep Learning 패턴 -> event 프로그램으로
trading_positions = trading_positions_raw.apply(np.sign) * 1/3
trading_positions

trading_positions_final = trading_positions.shift(1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))

ax1.plot(data.loc[start_date:end_date, :].index, data.loc[
start_date:end_date, 'MSFT'], label='Price')
ax1.plot(ema_short.loc[start_date:end_date, :].index, ema_short.loc[
start_date:end_date, 'MSFT'], label='20-days EMA')

ax1.set_ylabel('$')
ax1.legend(loc='best')
ax1.xaxis.set_major_formatter(my_year_month_fmt)

ax2.plot(trading_positions_final.loc[start_date:end_date, :].index,
trading_positions_final.loc[start_date:end_date, 'MSFT'],
label= 'Trading position')


ax2.set_ylabel('trading_positions')
ax2.xaxis.set_major_formatter(my_year_month_fmt)


# 빨리 변화   /
# 20일 이평선 / 60일 이평선
#
#  => 교차 나타나면 매매시전



# prophet 비선형에측기 이고 계절성을 고려해주고, 일별 주기성 데이터 강함

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') #출력형태의 다양성
from fbprophet import Prophet
df = pd.read_csv('AirPassengers.csv')
df.head(5)
df['Month'] = pd.DatetimeIndex(df['Month'])
df.dtypes

df= df.rename(columns={'Month': 'ds',
'AirPassengers' : 'y'})

df.tail(5)

from matplotlib import pyplot as plt
ax = df.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Airline AirPassengers')
ax.set_xlabel('Date')
plt.show()

my_model = Prophet(interval_width=0.95)
my_model.fit(df)

future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
future_dates.tail()

forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']].tail()

my_model.plot(forecast, uncertainty=True) #기존 데이터 , 예측데이터 출력
my_model.plot_components(forecast) # 분해 시계열 출력
plt.show()








# Leaflet : Folium pip install folium
import folium
map_1 = folium.Map(location=[37.565711, 126.978090], zoom_start=16)
folium.Marker([37.565718, 126.978098], popup='서울시청',
icon=folium.Icon(icon='cloud')).add_to(map_1)
folium.Marker([37.565818, 126.978198], popup='서울시청밑',
icon=folium.Icon(icon='cloud')).add_to(map_1)
map_1




#
input("학교이름을 입력하시오")
# -> 그 학교의 위도 경도를 출력해줌.
# 위로 i  경도 j  학교이름 k로 묶고

# append로 리스트에 넣어준다.

append(i)
append



for i,j,k in univ_list:
    folium.Marker([i, j], popup='k', icon=folium.Icon(icon='cloud')).add_to(map_1)


#

folium.Map.save(map_1, "index.html")

import webbrowser

webbrowser.open_new("index.html")

# 1.숙제 : 서울 소재 대학교(10) 의 위경도 좌표를 얻고 지도 표현해 보시오.


# 2. 삼성 주식데이터를 2010~ 어제까지 시각화 하고 20일 이평과 60일 이평,
# 120일 이평을 한꺼번에 시가화하여 비교해보시오.

# 구글맵스에서 키
