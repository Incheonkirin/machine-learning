#한글 깨짐 현상
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

import pandas as pd, numpy as np

df = pd.Series([1, np.nan, np.nan, 3])
print(df)
df.interpolate() #보간법


## resampling : 시계열 데이터 중에 없는 데이터를 채우기 위해서 사용
s = pd.Series([1,2,3], index=pd.date_range('20180101', periods=3, freq='h'))
print(s)
print(s.resample('30min').backfill())

# multi Index
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
['one', 'two', 'one','two', 'one', 'two', 'one', 'two']]
tuples = list(zip(*arrays))
print(tuples)
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
print(index)
s = pd.Series(np.random.randn(8), index=index)
print(s)



# state     Colorado  Ohio
# color        Green Green Red
# key1 key2
# a    1           2     0   1
#      2           5     3   4
# b    1           8     6   7
#      2          11     9  10

frame = pd.DataFrame(np.arange(12).reshape((4,3)),
index=[['a','a','b','b'],[1,2,1,2,]],
columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])

frame.index.names=['key1', 'key2']
frame.columns.names=['state','color']
print(frame)
print(frame['Ohio'])
print(frame.swaplevel('key1', 'key2'))
print("sort_index시작")
print(frame.sort_index(1)) #콜로라도와 오하이오 순서가 바뀜. (1) = 열방향으로





# pivot
from collections import OrderedDict
table = OrderedDict((
("Item", ['Item0', 'Item0', 'Item1', 'Item1']),
("CType", ['Gold', 'Bronze', 'Gold', 'Silver']),
("USD", ['1dollor', '2dollor', '3dollor', '4dollor']),
("EU", ['1E', '2E', '3E', '4E'])
))

d = pd.DataFrame(table)
print(d)
p=d.pivot(index='Item', columns='CType', values='USD')
print(p)

#v value를 지정하지 않을 때
p = d.pivot(index='Item', columns='CType')
print(p)



# crosstab 행과 열/ 축을 지정해서 도수분포표를 만든다.
import pandas as pd
import numpy as np
# df = pd.DataFrame({'A' : ['일반', '일반', '특수', '예외'] * 6,
# 'B' : ['서울', '대구', '부산']* 8,
# 'C' : ['건조', '건조', '건조', '우기', '우기', '우기'] *4,
# 'D' : np.random.randn(24),
# 'E' : np.random.randn(24)})

print(df)
print(pd.crosstab(df.A, df.B))
print(pd.crosstab(df.A, df.C, margins=True)) #margins 합계
print(pd.crosstab([df.A, df.B], df.C, margins=True)) #중복키

print(pd.crosstab(df.A, df.B).apply(lambda r: r/r.sum(), axis=1)) #열방향으로 비율 생성




d = {
'Name': ['김중호', '일구', '이구', '삼구', '사구', '오구',
'육구', '칠구', '팔구', '박진', '비옴', '최훈'],
'Exam':['중간고사', '중간고사', '중간고사', '중간고사', '중간고사', '중간고사',
'기말고사','기말고사','기말고사','기말고사','기말고사','기말고사'],
'Subject':['수학','수학','수학','과학','과학','과학',
'수학','수학','수학','과학','과학','과학'],
'Result':['Pass','Pass','Fail','Pass','Fail','Pass','Pass','Fail','Fail',
'Pass','Pass','Fail'],
'Jumsu' : [100, 80, 70, 60, 70, 80, 100, 80, 60, 100, 80, 80]
}

df = pd.DataFrame(d, columns=['Name', 'Exam', 'Subject', 'Result','Jumsu'])
print(df)

1)과목별 패스 여부를 확인하시오
print(pd.crosstab(df.Subject, df.Result))

2)과목별로 분류하고 시험분별로 분류된 패스 여부를 확인하고 합계를 출력
print(pd.crosstab([df.Subject,df.Exam], df.Result, margins=True))

3)시험별로 패스 여부를 확인하시오
print(pd.crosstab([df.Name,df.Exam], df.Result))

4)시험별 이름별로 점수를 확인하시오 #범주형일 때는 카운팅을하나, 숫자일 경우에는 aggfunc을 정의해 주어야 함.
pd.crosstab(df.Exam, df.Name, df.Jumsu, margins=True) => Error !
t=pd.crosstab(df.Exam, df.Name, df.Jumsu, aggfunc=[np.sum], margins=True)
print(t)


# 문제 : diamond.csv를 로딩한 다음, 다음 문제를 해결하시오 .

path = "diamond.csv"
df = pd.read_csv(path)
print(df.dtypes) #object #문자열     오브젝트가 문자열이라고 확인하면 됨.
print(df.applymap(type).head(1))
print(df.describe())
print(df)
# values에 숫자형 데이터가 오면, 반드시 aggfunc이 들어가야 함.

my_tab= pd.crosstab(index = df["clarity"], columns=df["cut"], values=df["price"], aggfunc=[np.sum])
                                             #범주형 변수
print(my_tab)
my_tab.head()
my_tab.plot.bar()

# oneway table 하나만 주는 것임.
my_tab = pd.crosstab(index=df["clarity"], columns="count")
print(my_tab.head())
my_tab.plot.bar()
my_tab.shape()

# -clarity와 cut간 crosstab을 생성하고 barplot하시오.

# -price와 clarity간의 barplot을 출력하시오.(clarity별)
df.boxplot(column="price", by="clarity", figsize=(8,8))

# -cut과 clarity로 군집화하고 그 size를 확인하시오.
#군집화하라 ? -> groupby
grouped = df.groupby(['cut', 'clarity'])
print(grouped.head())
print(grouped.size())


# -clarity와 color간의 crosstab을 생성하고, barplot 하시오.
clarity_color = pd.crosstab(index=df["clarity"], columns=df["color"])
print(clarity_color)
clarity_color.plot(kind="bar",
figsize=(8,8), stacked=True)


from pandas import DataFrame

Employees = {'Name of Employee': ['Jon','Mark','Tina','Maria','Bill','Jon','Mark','Tina','Maria','Bill','Jon','Mark','Tina','Maria','Bill','Jon','Mark','Tina','Maria','Bill'],
             'Sales': [1000,300,400,500,800,1000,500,700,50,60,1000,900,750,200,300,1000,900,250,750,50],
             'Quarter': [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
             'Country': ['US','Japan','Brazil','UK','US','Brazil','Japan','Brazil','US','US','US','Japan','Brazil','UK','Brazil','Japan','Japan','Brazil','UK','US']
            }
df = DataFrame(Employees)
print(df)

df = DataFrame(Employees, columns= ['Name of Employee', 'Sales','Quarter','Country'])
print(df)

#위 아래 다른 점? 아래와 같이 하면, 필요한 부분만 빼서 낼 수 있음.


# 1) 종업원 당 전체 매출액
pivot = df.pivot_table(index=['Name of Employee'], values=['Sales'], aggfunc='sum')
print(pivot)

#2)국가 별 총 매출액 계산 출력
pivot = df.pivot_table(index=['Country'], values=['Sales'], aggfunc='sum')
print(pivot)

#3)영업사원 별 국가별 매출액 출력
# 'Name of Employee', 'Country' : 계층인덱스
pivot = df.pivot_table(index=['Name of Employee', 'Country'], values=['Sales'],
aggfunc='sum')


# 국가별 단일 최고 매출액을 출력하시오
pivot = df.pivot_table(index=['Country'], values=['Sales'], aggfunc='max')
print(pivot)
pivot = df.pivot_table(index=['Country'], values=['Sales'], aggfunc={'median', 'mean', 'min'})
print(pivot)
df.pivot_table(index=['Country'], values=['Sales'], aggfunc={'median', 'mean', 'min'}).plot()



df = pd.read_excel("sales-funnel.xlsx")
df.head()
print(df.dtypes)
# Account      int64
# Name        object
# Rep         object
# Manager     object
# Product     object
# Quantity     int64
# Price        int64
# Status      object
# dtype: object

# 위 데이터 중 범주형이 어떤 것??
#  status !!!
# 범주형으로 바꿀 때 무엇을 씀 ?  astype을 사용하면 됨.

df["Status"] = df["Status"].astype("category")
#                                          범주형에서 범주를 제한할 때, set category를 이용해서 !

df["Status"].cat.set_categories(["won", "pending", "presented", "declined"], inplace=True)

pd.pivot_table(df, index=["Name"])
#전체열 8개 인데, 그 중 3개만 출력된다 !    -----------------> 중복된 것은 제외함.

pd.pivot_table(df, index=["Name"], columns=["Manager"])
#이 중에서 price만 하려면, value값을 제한하면 됨.
print(pd.pivot_table(df, index=["Name"], columns=["Manager"], values=["Price"]))

print(pd.pivot_table(df, index=["Manager","Rep"], values=["Price"]))
print(pd.pivot_table(df, index=["Manager","Rep"], values=["Price"], aggfunc=np.sum))
print(pd.pivot_table(df, index=["Manager","Rep"], values=["Price"], aggfunc=[np.mean, len])) #두가지 같이 보고 싶을 때

# 좀 더 복잡하게 !
print(pd.pivot_table(df, index=["Manager","Rep"], values=["Price"], columns=["Product"], aggfunc=[np.sum], fill_value=0 ))

#열 합계까지 내줌.
print(pd.pivot_table(df, index=["Manager","Rep"], values=["Price"], columns=["Product"], aggfunc=[np.sum], fill_value=0, margins=True ))

# 좀 복잡해짐..
# 이걸 데이터 베이스라고 하고, 쿼리를 날리려 함.
# 매니저가 이 사람인 사람만 보고 싶다.
table = pd.pivot_table(df, index=["Manager","Status"], columns=["Product"], values=["Quantity", "Price"],
aggfunc={"Quantity":len, "Price": [np.sum, np.mean]}, fill_value=0)

# 쿼리
table.query('Manager == ["Devbra Henley"]')
table.query('Status == ["pending", "won"]')



import seaborn as sns
tips = sns.load_dataset("tips")
tips.tail()
print(tips.tail())

#문제 : 팁의 비율이 요일과 점심/저녁, 인원수에 어떤 영향을 받는지 살펴본다.
#어떤 요인이 가장 크게 작용하는지 판단할 수 있는 방법이 있는가?

tip = tips.tip
type = tips.time
count = tips.size

#결론 : 요일은 조금 영향, 점심 저녁은 영향 없음. 인원수는 큰 영향
#남자가 담배를 피우면 팁이 많아진다.


tips.pivot_table(index=['time', 'size'], columns=['day'], values='tip', margins=True)
def peak_to_peak(x):
    return x.max() - x.min()
tips.groupby(["sex", "smoker"])[["tip"]].agg(peak_to_peak)


# dataset.csv
#  문제 :

# 1. '공백' 이 문제를 일으킴
# 2. describe할 때 숫자 데이터가 나타나지 않음. R에서는 잘 사용하던 데이터인데, 왜 안되는가?

df = pd.read_csv("dataset.csv", encoding="UTF-8")
print(df.head())
print(df.columns.tolist())

df.columns = df.columns.str.strip() #공백 제거 # #하면 제대로 나옴.
print(df.columns.tolist())

print(df.describe) #출력되지 않는 게 나옴.
print(df.head())
print("데이터 타입확인:", df.dtypes)
#숫자형 데이터로 변경.

df['age']=df['age'].astype(str).str.strip()
df["age"]=df["age"].fillna('',inplace=True)
df["age"]=pd.to_numeric(df['age'].astype(float), errors='ignore')

print("데이터 타입확인:", df.dtypes)
df['price']=df['price'].astype(str).str.strip()
df["price"].fillna('',inplace=True)
df["price"]
df["price"] = pd.to_numeric(df['price'], errors='ignore')
df["price"]=df["price"].astype(float)
print("데이터 타입확인:", df.dtypes)

#panel(3차원) python에서 사이즈 다르면 : 제일 큰 것을 기준으로
data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)),
'Item2' : pd.DataFrame(np.random.randn(4, 2)),
'Item3' : pd.DataFrame(np.random.randn(4, 2))}
p = pd.Panel(data)
print(p)
print(p['Item1'])
print(p['Item2'])
print(p['Item3'])
print(p.major_xs(0))
print(p.minor_xs(1))

# Major_axis axis: 0 to 3  행으로
# Minor_axis axis: 0 to 2  열로


#  1) dataset.csv를 로딩하고, resident열과 position열을 출력하시오


#  2) describe로 확인할 때 숫자 데이터인 것이 나타나지 않는 원인을 확인하시오

#  3) 각 열의 데이터 타입을 확인하시오
#  4) price와 age의 데이터 타입이 숫자가 아니면 숫자 데이터로 변경하시오


import pymysql
import numpy
conn = pymysql.connect(host="192.168.1.81", port=3306, user='root',
passwd="acorn1234", db="acorn", charset="utf8mb4",
cursorclass=pymysql.cursors.DictCursor)
cursor = conn.cursor()
cursor.execute("select * from student")
rez = cursor.fetchall()
df = pd.DataFrame(rez)
print(df)
# print(df.grade)


# 문제 : 점수를 학년 반으로 입력하고, pivot table을 이용해서
# 학년 반 별 평균을 출력하시오

avr = (df_numeric.apply([mean]))
d=df.pivot(index='class', columns='grade', values='average')

print(d)


# #국어 영어 수학의 총점과 최고점을 출력하시오.
df_numeric = df[['eng', 'kor', 'mat']]
print(df_numeric.apply([sum,max]))



print(df.describe())
print(df.dtypes)
df.plot.bar()
from collections import OrderedDict
table = OrderedDict((
("Item", ['Item0', 'Item0', 'Item1', 'Item1']),
("CType", ['Gold', 'Bronze', 'Gold', 'Silver']),
("USD", ['1dollor', '2dollor', '3dollor', '4dollor']),
("EU", ['1E', '2E', '3E', '4E'])
))

d = pd.DataFrame(table)
print(d)
p=d.pivot(index='Item', columns='CType', values='USD')
print(p)


import json
json_data1 = """{
"color":"red",
"value":"#f00"
}"""

result = json.loads(json_data1)
print(result)
print(result["color"])
print(result["value"])



#
from bs4 import BeautifulSoup
import urllib3
http = urllib3.PoolManager() #connect 여러번 할 경우에 (connect)

movieListUrl = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieList.json?key=430156241533f1d058c603178cc3ca0e&openStartDt=2016&itemPerPage=20"

response = http.request('GET', movieListUrl)
movieIdData = BeautifulSoup(response.data, "html.parser")

print(type(movieIdData))
print(movieIdData.prettify())
movieIdData = json.loads(str(movieIdData))
print(movieIdData["movieListResult"])
print(len(movieIdData["movieListResult"]["movieList"]))
movieIdData["movieListResult"]["movieList"]['movieCd'])) #movieCd를 알 고 싶다면...
movieIdData["movieListResult"]["movieList"][i]['movieCd']))


# 데이터 프레임 작업
movieIDDF = pd.DataFrame()
movieIDDF = movieIDDF.append({"movieCd":"", " movieNm":"",
"movieNmEn":"", "openDt":"", "directorpeopleNm":""}, ignore_index=True)


# 문제

# 데이터를 munging해서 movieCd, movieNm, movieNmEn, openDt, directorpeopleNm을
#        파싱
# 정리해서 데이터 프레임에 입력하시오.


#    directorpeopleN === ["directors"][0]["peopleNm"]

# openDt를 year, month, day 로 분리해서 입력하시오.


# 년도별, 월별, 일자별 개봉작수를 카운트하는 pivot_table을 생성하시오.

# #데이터 프레임은 실시간으로 추가 가능
num = len(movieIdData["movieListResult"]["movieList"])
print(num)
for i in range(0, num):
    movieIDDF.loc[i,"movieCd"] = movieIdData["movieListResult"]["movieList"][i]["movieCd"]
    movieIDDF.loc[i,"movieNm"] = movieIdData["movieListResult"]["movieList"][i]["movieNm"]
    movieIDDF.loc[i,"movieNmEn"] = movieIdData["movieListResult"]["movieList"][i]["movieNmEn"]
    movieIDDF.loc[i,"openDt"] = movieIdData["movieListResult"]["movieList"][i]["openDt"]
    movieIDDF.loc[i,"directorpeopleNm"] = movieIdData["movieListResult"]["movieList"][i]["directors"][0]["peopleNm"]
movieIDDF
movieIDDF["year"] = movieIDDF["openDt"].apply(lambda x:str(x)[:4])
movieIDDF["month"] = movieIDDF["openDt"].apply(lambda x:str(x)[4:6])
movieIDDF["day"] = movieIDDF["openDT"].apply(lambda x:str(x)[6:])
print(movieIDDF)

pd.pivot_table(movieIDDF, index=["year", "month", "day"], values=["directorpeopleNm"],aggfunc=[np.size], fill_value=0)




# sales.csv
df = pd.read_csv("sales.csv")
df
1) , $ 제거
2) % 제거 (100으로 나누기)
3) Y/N 을 True False로 바꾸기
4) 문자를 숫자로 변경할 것을 변경하시오
