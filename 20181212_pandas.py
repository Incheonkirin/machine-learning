import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(10, 22).reshape(3,4),
                  index=["x","y","z"], #행
                  columns=["A","B","C","D"]) #열
#     A   B   C   D
# x  10  11  12  13
# y  14  15  16  17
# z  18  19  20  21

3+4
print(df)
print("")
print(df.loc["x", "A"])
print(df.loc["y", "A"])
print(df.loc["x", :])
print(df.loc[["x", "y"], ["B", "D"]])
print(df.loc[df.A > 10, :]) #boolean indexing

print(df[:1]) #행
print(df.iloc[1,2])

df.loc["e"] = [90, 91, 92, 93] #데이터를 실시간으로 추가
print(df)
print(df.iloc[0,1])
print(df.iloc[0,-2 :])
print(df.iloc[2:3, 1:3])
print(df.iloc[-1])
df.iloc[-1] = df.iloc[-1] * 2 #값의 수정 : 마지막 행 값 변경
print(df)




data = {'state' : ['경기', '강원', '서울', '충북', '인천'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop' : [1.5, 1.7, 3.6, 2.4, 2.9]}
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],#NaN
                      index = ['one', 'two','three', 'four', 'five'])

print(frame2)
print(frame2['state'])
print(frame2.year)
print(frame2.state)

print(frame2['debt'])
frame2['debt']=16.5 # broadcasting
print("debt 값 적용 후", frame2)

val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
print("시리즈 데이터 삽입후", frame2)

frame2['eastern'] = frame2.state =='서울'
print("데이터 추가", frame2)
del frame2['eastern']

print(frame2.colummns)



# dataframe이 제공하는 통계함수

data = {'name': ['하늘이', '찬호박', '우리야', '함께가', '하성공'], 'age': [40, 50, 30, 20, 70],
        'preScore': [14, 28, 39, 25, 32], 'postScore': [20, 90, 55, 65, 79]}
df = pd.DataFrame(data, columns=['name', 'age', 'preScore', 'postScore'])
df
print(df['age'].sum())
print(df['preScore'].mean())
print(df['preScore'].cumsum()) #누적합계
print(df['preScore'].describe()) #summary
print(df['preScore'].var())
print(df['preScore'].std())
print(df['preScore'].skew()) # 왜도 : 0
print(df['preScore'].kurt()) # 첨도: 3



## dataframe 연산
# 사각형 형태로 일치, 데이터 없는 부분은 NaN처리
df = pd.DataFrame({'one' : pd.Series(np.random.randn(3),
                                     index=['a','b','c']),
                   'two' : pd.Series(np.random.randn(4),
                                     index=['a', 'b', 'c', 'd']),
                   'three' : pd.Series(np.random.randn(3),
                                       index=['b','c','d'])})
print(df)

row= df.iloc[1]
print(row)
column=df['two']
print(column)


#데이터 프레임 연산 #  add, mul, div, sub
df.sub(row, axis='columns') #열방향으로 빼기 연산
df.sub(row, axis=1)
df.sub(column, axis='index')
df.sub(column, axis=0)
#상관분석

#상관분석 => 열을 기준으로 !!
print(df.corr()) #상관계수
print(df.cov()) #공분산

print(df.describe()) #일별로 통계데이터 출력
print(df+df) #데이터프레임끼리 통합연산 가능


df2 = pd.DataFrame({'one' : pd.Series(np.random.randn(3),
                                     index=['a','b','c']),
                   'two' : pd.Series(np.random.randn(4),
                                     index=['a', 'b', 'c', 'd']),
                   'three' : pd.Series(np.random.randn(3),
                                       index=['b','c','d'])})



#두개 비교해보면, add 함수를 쓰는게 유리하다.
print(df + df2)
df.add(df2, fill_value=0) # NaN값을 0으로 채우고


df.gt(df2) #관게연산자 greater than >
df.ne(df2) # not equal
# => 결과값은 전부 True False로 나오게 됨.


# 기준? => 열 !!!
(df >0). all() #   개의 열 중에 모두 맞아야 참.
(df >0). any() #   개의 열 중에 하나라도 맞으면 참.
(df >0). any().any()  # 각기 계산한 다음, 하나라도 .
(df >0). any().all() # 각기 계산한 다음 , 모두 .


#데이터 프레임의 정렬과 비교
df1 = pd.DataFrame({'col':['foo', 0, np.nan]})
df2 = pd.DataFrame({'col':[np.nan, 0, 'foo']}, index=[2,1,0])
print(df1)
print(df2)
print(df2.sort_index()) #인덱스 정렬

#값에 의한 정렬은 에러발생 (숫자와 문자가 동시에 존재하면 에러 사용불가)
#df2.sort_values(by=['col']) =>

print(df1.equals(df2)) #False
df1.equals(df2.sort_index()) #True = 요소값으로 비교한 결과로 판단


#값에 의한 정렬
df = pd.DataFrame({
    'col1' :['A', 'A', 'B', np.nan, 'D', 'C'],
    'col2' : [2, 1, 9, 8, 7, 4],
    'col3' : [0, 1, 9, 4, 2, 3],
})

df
df.sort_values(by=['col1']) # by기준이 되는 열을 젝고

# combine에 의한 dataframe의 합성

df1 = pd.DataFrame({'A' : [1., np.nan, 3., 5., np.nan],
                    'B' : [np.nan, 2., 3., np.nan, 6. ]})
print(df1)
df2 = pd.DataFrame({'A': [5., 2., 4., np.nan, 3., 7.],
                   'B' : [np.nan, np.nan, 3., 4., 6., 8.]})

print(df2)
combiner = lambda x, y : np.where(pd.isnull(x), y, x)
df1.combine(df2, combiner) #함수를 요구하는 combine함수

#시간 : 2000.1.1부터 경과시간을 mili초로 표현
# 1초 = 1000 밀리초
import time
print(time.time())
print(time.localtime())
yesterday = time.localtime(time.time()-60*60*24)
yesterday
time.strftime('%Y %m %d') #시간을 format


from datetime import datetime, timedelta #timedelta는 시간차
now = datetime.now()
type(now)
now
now.year, now.month, now.day
now.timestamp() # => Linux에서는 시간을 timestamp로 함.
now_str= now.strftime('%Y-%m-%d %H:%M:%S')
datetime.strptime(now_str, '%Y-%m-%d %H:%M:%S') #시간 추출함수


delta = datetime(2015, 1, 7) - datetime(2010, 6, 24, 8, 15)
print("시간차는 =" , delta, delta.days, delta.seconds)

#시간차 연산 단위
#days, weeks, hours, minutes, seconds
start = datetime(2015, 1, 7)
print(start + timedelta(12)) #날짜 :: timedelta의 디폴트는 날짜다 !
print(start -2*timedelta(12))
start + timedelta(hours= -5)


# 판다스에서는 시계열 분석에서 시간표시가 중요하기 때문에, 판다스는 시간에 관한 함수를 제공함.
# 판다스에서는 nano초를 제고해서 정밀한 시간 표현 가능.
import pandas as pd # 행인덱스(시간)를 사용하기 위해서 제공
print (pd.datetime.now())
print (pd.Timestamp('2017-03-01'))
# 기간을 표현
# 주파수 = 간격

print(pd.date_range("11:00", "13:30", freq="30min"))
print(pd.date_range("11:00", "13:30", freq="30min").time)
print(pd.date_range("11:00", "13:30", freq="H"))

s = pd.Series(pd.date_range('2012-1-1', periods=3, freq='D'))
print(s)
td = pd.Series([ pd.Timedelta(days=i) for i in range(3)])
print(td)

df = pd.DataFrame(dict(A = s, B = td))
df['C']=df['A']+df['B']
df['D']=df['C']+df['B']
print(df)


# A의 타입을 확인하시오
print(type((df['A'])))  #Series


#%matplotlib inline
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum() #이걸 찍었을 때, 나타나는 그림이 주는 의미? => np.random.randn(1000) 순 증가
ts.plot()
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=list('ABCD'))
df = df.cumsum()
df.plot();


# 출력함수 제공 (matplotlib을 wrapper 해서 제공)
df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df2.plot.bar()
df2.plot.barh() # horizontal
df2.plot.bash(stacked=True) #행별로 4개의 열을 출력하는 것을 하나로 출력
df2.plot.hist(alpha=0.5)
df2.plot.hist(stacked=True, bins=20)
#시리즈도 출력 함수 제공

series = pd.Series(3*np.random.rand(4), index=['a', 'b', 'c', 'd'],
                   name='series')
print(series)
series.plot.pie(figsize=(6, 6))



#범주화
df = pd.DataFrame({'value': np.random.randint(0, 100, 20)})
labels = ["{0} - {1}".format(i , i + 9) for i in range(0, 100, 10) ]
print(labels)
# R에서 연관분석 -> 범주형 데이터  'cut 함수' 사용했었음
df['group'] = pd.cut(df.value, range(0, 105, 10), right=False,
                     labels=labels)
# group 이라는 파생변수가 만들어 짐.
df.head(10)


# Categorical 함수
raw_cat = pd.Categorical(["a", "b", "c", "a"], categories=["b", "c", "d"],
                         ordered=False) # 순서가 없다.

print(raw_cat)
pd.Series(raw_cat) #범주형변수는 하나의 열로 추가

df = pd.DataFrame({"A":["a","b","c","a"]})
df["B"] = raw_cat

print("범주화 데이터가 있는 데이터프레임")
print(df)
print(df['B'])

s = pd.Series(["a", "b", "c", "a"])
s_cat = s.astype("category", categories=["b", "c", "d"], ordered=False)
s_cat


#열 이름을 기준으로 정렬
frame = pd.DataFrame((np.arange(8).reshape(2,4)), columns=['d', 'a', 'b', 'c'],
                     index=['three', 'one'])
print(frame)
print(frame.sort_index()) #행이 기본
print(frame.sort_index( axis=1))
print(frame)


# 데이터에 함수를 적용 (numpy 함수의 매개변수로 전달)
frame = pd.DataFrame((np.random.randn(4,3)), columns=list(['rain',
                                                           'income', 'tax']), index=['seoul', 'daejun', 'incheon', 'daegu'])
print(frame)
print("absolute함수 적용", np.abs(frame))


#apply 함수는 열단위나 행단위로 적용 되어 집니다.
f = lambda x : x.max() - x.min()
print("함수 객체의 행 적용 (열방향)", frame.apply(f))                 #    apply     열
print("함수 객체의 열 적용 (행방향)", frame.apply(f, axis=1))         #    apply     행

format = lambda x : '%.2f' %x #소수점 2번째 자리
print(frame.applymap(format)) #요소에 적용이 됨.                         applymap  요소
print(frame['rain'].map(format))#                                       map      시리즈


#pandas는 문자열 처리함수 제공
df = pd.DataFrame(np.random.randn(3, 2), columns =[' Column A',
                                                    'Column B'], index=range(3))

print(df.columns.str.strip())
print(df.columns.str.lower())
# chaining 을 사용하고 있음.

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
#                     strip은 앞 뒤 공백만 제거
print(df)


pd.Series(['a1', 'b2', 'c3']).str.extract('([ab])(\d)', expand=False)
                                 #a나 b를 선택해서 추출하고,
                                             #    숫자
# dummy 변수 생성
s = pd.Series(['a', 'a|b', np.nan, 'a|c'])  # a, b, c
print(s)
s.str.get_dummies(sep='|')


# 문제
sqldate = pd.Series(["2014-0-1", "2015-10-10", "1990-23-2"])
pdf = pd.Series(["2014.pdf", "2015.pdf", "1999.pdf"])
# 위의 두 데이터로 dataframe을 생성하고 년도를 비교한 결과를 newcol 이라는 필드로 추가
df = pd.DataFrame({"sqldate":sqldate , "pdf": pdf})
df['newcol'] = df['sqldate'].str[0:4] == df['pdf'].str[0:4]
df['newcol'] = df['sqldate'].str.slice(0,4) ==df['pdf'].str.slice(0,4)
print(df)

# 문제
#['한글','미국','일본?'] 문자열을 데이터 프레임에 저장하고 문자의 개수 'text_length'라는 파생변수를
# 만들어서 입력하시오

df = pd.DataFrame(['한글', '미국', '일본?'],columns=['text'])
df['text_ength'] = df['text'].map(len)
print(df)


#파일처리
names = ['한국성', '공하자', '희망이', '꿈꾼다', '아리랑']
births = [25, 30, 38, 28, 31]
BabyDataSet = list(zip(names, births))
df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])
print(df)

df.to_csv('births1880.csv', index=False, header=True, encoding="UTF-8")
Location = r'./births1880.csv'
print(df)
df=pd.read_csv(Location, names=['Names', 'Births'], encoding='utf-8')
df

# pima.csv
pima = pd.read_csv("pima.csv", index_col=0)
pima.head()
# npreg 임신자수
# glu  글루코스 포도당 검사 : 당화혈색소
# bp 혈압
# skin 피부 결침
# bmi 체중지수
# ped 가족력
# type 당뇨병여부

print(pima.count())
print(pima.mean())
print(pima.median())
print(pima.quantile())
print(pima.std())
print(pima.var())
print(pima.min())
print(pima.max())
pima.mean(axis=1)
pima[0:2]
pima[-3:]
pima["bmi"]
type(pima["bmi"])
print(pima[pima["bmi"]>30].shape)
pima.describe()
pima["bmi"].hist()
import matplotlib.pyplot as plt
plt.show()

pima["bmi"].plot(kind="kde")
print(pima.head())
pima.groupby("type") #그룹바이 객체를 생성
#aggregation함수를 적용해서 데이터 출력
pima.groupby("type").mean()
grouped_by_type = pima.groupby("type")
grouped_by_type.mean()
grouped_by_type.std()
grouped_by_type.var()
#계층적 인덱스가 적용
grouped_by_type.agg([np.mean, np.std, np.var])

print(np.mean(pima[pima["type"]=="Yes"]["skin"]))
np.std(pima[pima["typpe"]=="Yes"]["skin"])


#문제 weather_2012.csv 데이터를 로딩하시오

# 데이터로딩
# 차수를 확인하고 데이터의 min max 값의 범위를 확인하시오
# 로딩된 데이터의 Temp(C) 컬럼을 시각화하시오
# 데이터를 boxplot으로 이상치여부를 확인하시오
# 결측치를 제거하시오

weather_2012_final = pd.read.csv("weather_2012.csv", index_col='Date/Time')
print(weather_2012_final.shape) #(8784, 7)
print(weather_2012_final.head())
weather_2012_final.apply(type)
weather_2012_final.applymap(type).head()

bigFilePath="weather_2012.csv"
#chunksize=1000

chunker = pd.read_csv(bigFilePath, chunksize=1000, index_col='Date/Time', encoding="UTF-8")
weather_2012_final = pd.concat([x for x in chunker], ignore_index= True)
print(weather_2012_final.head(3))
print(weather_2012_final.describe())

weather_2012_final['Temp(C)'].plot(figsize=(30, 12))
weather_2012_final = weather_2012_final.dropna(axis=1, how='any')


#multi file loading
import glob
import os
filePathList = glob.glob("./same_format_files/*.csv")

for i in range(0, len(filePathList)):
    temp = os.path.basename(filePathList[i])
    temp = os.path.splitext(temp)[0]
    vars()["data_" + str(temp)] = pd.read_csv(filePathList[i])
print(data_1763.head(3))
print(data_1770.head(3))
print(data_1763.shape)
print(data_1770.shape)
print(data_1772.shape)

vars()

#merge
import pandas as pd
df1 = pd.DataFrame({ 'key':['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1':range(7)})
print(df1)
df2=pd.DataFrame({'key':['a','b','d'], 'data2':range(3)})
print(df2)

print(pd.merge(df1, df2, how='inner')) #일치하는 키에 따라서
print(pd.merge(df1, df2, on='key'))
print(pd.merge(df1, df2, left_on='key', right_on='key'))
pd.merge(df1, df2, how='outer') #합집합
#R에서 melt
data = pd.DataFrame(np.arange(6).reshape(2,3), index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'], name='number'))
print(data)
result=data.stack()
print("분리")
print(result)
print(result.unstack())

# duplicate 중복
data=pd.DataFrame({'k1':['one']*3 + ['two']*4,
                'k2':[1,1,2,3,3,4,4]})
print(data)
print(data.duplicated())
data.drop_duplicates()
data['v1'] = range(7)
print(data.drop_duplicates(['k1']))
data.drop_duplicates(['k1','k2'],keep='last') # 중복된 것 중에 마지막것 남긴다


## replace
data = Series([1., -999., 2., -999., -1000., 3.])
print(data)
print(data.replace(-999, np.nan))
print(data.replace([-999, -1000], np.nan))
print(data.replace([-999, -1000], [[np.nan, 0]))
print(data.replace({-999:np.nan, -1000:0}))


#축 이름 변경
data = pd.DataFrame(np.arange(12).reshape((3,4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])

print(data.index.map(str.upper)) #변경안됨
print(data)

data.index=data.index.map(str.upper)
data.rename(index=str.title, columns=str.upper)
data.rename(index={'Ohio':'Indiana'}, columns={'three':'peekaboo'})


# 문제 : olive.csv를 로딩한 후 다음 문제를 해결 하시오
1) 첫번째 컬럼의 이름을 ID_area 로 지정하시오 (index_col='Id_area')

df.rename(columns={df.columns[0]:'ID_area'}, inplace=True)

2) regions의 값들이 중복하지 않고 몇개의 범주인지 확인하시오
3) area 도 위와 같이 처리하시오
4) 처음 컬럼(ID_area) 에 들어온 이상한 숫자를 제거하시오
5) 산성관련 성분인 'palmitic', 'palmitoleic','stearic','oleic',
'linoleic','linolenic','arachidic','eicosenoic'을 추려서 별도의 sub 데이터 프레임
(변수이름=dfsub)으로 생성하시오
6) dfsub의 데이터를 모두 100으로 나누어서 소수점으로 나타내시오
7) palmitix산과 linolenic 산의 분포도를 시각화 하시오
8) palmitic 히스토그램을 간단하게 시각화 하시오
9) xacids=['oleic','linolenic','eicosenoic'],
   yacids=['stearic','arachidic'] 을 가지고 각기
   xacids-yacids의 scatter plot를 그리기 (6개)





10) groupby 를 활용해서 region 을 기준으로 묶어서 region_groupby 객체로 생성
11) region_groupby 에 describe() 메소드를 적용해서 출력해보시오
