# conda install dask
# conda install -c conda-forge dask-tensorflow
# conda install distributed -c conda-forge

# 아래 첫번째 단계 실행시키고,
# dask-scheduler 을 쳐 넣으면, 주소가 나옴
# 다른 pc에 가서 dask-worker 하고 주소 쳐넣으면 연결 됨

#dask pyhthon 3.7에서는 에러가 남. 토네이도서버
# 3.6을 사용해야 함.

#@delayed decorator를 붙였다. 아래 호출되면 실행되어야 하는데, 실행이 안되게 함. compute를 호출해야 실행되게 함.

# 3.6 버전에서 실행 할 것 environment 환경에서 실행
print('hello world')

import graphviz
from dask import delayed, compute
import dask

@delayed
def square(num):
    print("square fn:", num)
    print()
    return num*num

@delayed
def sum_list(args):
    print("sum_list fn:", args)
    return sum(args)

items = [1, 2, 3]


computation_graph = sum_list([square(i) for i in items]) # 실행 안됨

computation_graph.visualize()
print("Result", computation_graph.compute()) # 실행 됨. session하고 같은 역할


#



# dask-worker 1192.168: 이걸 3개의 텐서플로 아나콘다 열어서 쳐 넣으면 붙음.

#
from dask.distributed import Client
client = Client('192.168.0.17:8786')
client.get_versions(check=True)
def square(x):
    return x**2
def neg(x):
    return -x

A = client.map(square, range(10))
B = client.map(neg, A)
total = client.submit(sum, B)
total.result()
total.gather()
client.restart()

client.get_versions(check=True)




##(분산처리없이 )대용량 데이터 처리 (배열 처리)
import dask.array as da
import numpy as np
arr = np.random.randint(1, 1000, (10000, 10000))
darr = da.from_array(arr, chunks=(1000, 1000))


print(darr.shape)

result = darr.compute()
result



#데이터 프레임 처리
import dask.dataframe as dd
df = dd.read_csv("wine-quality.csv", blocksize=50e6)
agg = df.groupby(['fixed acidity']).aggregate(['sum', 'mean', 'max', 'min'])

agg.compute().head()




from dask import delayed,compute
import dask
@delayed
def square(num):
    print("Square function:",num)
    print()
    return num*num
@delayed
def sum_list(args):
    print("Sum_list function:",args)
    return sum(args)
items=[1,2,3]
computation_graph = sum_list([square(i) for i in items])
computation_graph.visualize()
