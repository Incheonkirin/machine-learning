import numpy as np
a = np.array([1,2,3]) #결과가 ndarray. 함수이기 때문에 리스트로 들어가야 함.
print(a)
a = np.array([[1,2], [3,4]]) #2차원
print(a)
a = np.array([1,2,3,4,5], ndmin=2)  #결과가 2차원으로 나오게 됨.
print(a)
a = np.array([1,2,3], dtype=complex) #데이터타입을 명시적으로 줄 수 있다. 안 주어도 알아서 됨.
print(a)

a = no.array([[1,2,3],[4,5,6]])
print(a.shape)

a = np.array([[1,2,3],[4,5,6]])
type(a) #ndarray

a.reshape(3,2)
print(a)

a = np.arraay([[1,2,3],[4,5,6]]) #1,2,3, 4,5,6
b = a.reshape(3,2)
print(b)
#전체 메모리 공간

x = np.array([1,2,3,4,5], dtype=np.int8) #8비트 = 1바이트
print(x.dtype)

# fancy indexing #원하는 데이터만 찾아내는 fancy indexing
# 마지막 자기자신은 미포함
a = np.arrange(0, 100, 10)  # 0부터 100까지 10씩 증가하면서
indices = [1, 5, -1]
print(a.dtype)
print(a.itemsize)
b = a[indices]
print(a)
print(b)


#11부터 35까지 발생. 2차원으로. (5,5)

a = np.arrange(11,36)
#reshape을 써도 되고, a.shape써도 됨.
a.shape(5,5)
print(a)
print(type(a)) #class
print(a.dtype) # 요소타입 int32 (기본)
print(a.size) #전체 25 : 요소의갯수
print(a.shape) # (5,5) :행렬 수
print(a.itemsize) # 4바이트 : 요소의 바이트 수
print(a.ndim) # 2 :전체 몇차원이다.   3차원은 뭐라고 하나? tensor !
print(a.nbytes) # 100   : 전체 바이트 수, 메모리 바이트 수.





import matplotlib.pyplot as plt
x = np.arrange(1, 11)
y = 2 * 2 + 5
plt.plot(x, y, "ob")
plt.show()


import numpy as np
arr = np.arrange(10) # 0~9
print(type(arr)) # ndarray
#1차원
arr
print(arr[5] ) # 5는 6번째에 있음.
print(arr[5:8])
arr[5:8] = 12  # 값 변경
print(arr[1:6])
#2차원
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[0]) #행 지정
print(arr2d[0][2])
print(arr2d[0,2])
print(arr2d[:2])
print(arr2d[:2,1:]) #0,1 행. 1행부터 끝까지.
print(arr2d[1,:2]) #1행 0열부터시작, 2열까지.
print(arr2d[2, :1])
arr2d[:2, :1]=0
arr2d[:2, :1]=0
print(arr2d)


print(np.ones((10,5)))
print(np.ones((10,5)).ndim) # ones 는 1이 들어감.
print(np.ones((10,5)).shape)
print(np.ones((10,5)).dtype)
d = np.eye(4)
print("단위행렬", d)

#3차원
arr3d = np.array([[[1,2,3], [4,5,6]],[[7,8,9],[10,11,12]]]) # 2개 짜리가 2개 들어가 있음.  ' 2면 2행 3열 '
print(arr3d)
print("첫차원")
print(arr3d[0] ) # 면
print("첫 요소는 = ", arr3d[0][0][0]) # 요소
print("첫 차원 첫 행은= ", arr3d[0][0]) # 행까지

old_values = arr3d # 대입하면 주소값 복사가 벌어짐. 데이터 있는 위치는 같고, 주소값이 둘 다 같이 나옴.

old_values = arr3d.copy() # 명시적으로 데이터 복사

arr3d[0] = 42
print("값의 변경후")
print(arr3d)
print(" 이전 값으로 복구")
arr3d = old_values
print(arr3d)

#
Z = np.arrange(36) # 0부터 35까지 출력이 된다.
Z = Z.reshape((6,6))
reshape2 = Z.reshape((2,3,6))
print(reshape2)
reshape3 = Z.reshape((2,3,2,3))
print(reshape3)

#
import numpy as np
arr = []

for i in range(6):
    ad=[]
    for j in range(6):
        ad.append( i*10 + j)
    arr.append(ad)
print(arr)
arr = np.array(arr)

#1번
print(arr)
print(arr[[0,1,2,3,4],[1,2,3,4,5]])
# 각기 겹쳐지는 곳에 데이터를 얻어내는 것을 fancy indexing 이라고 한다.

#2번
print("정상으로", arr[[0,2,5],2])
# boolean indexing을 이용해서
mask = np.array([1,0,1,0,0,1], dtype=bool)
print(arr[mask, 2])

#3번
print("fancy = ", arr[[3,4,5],[0,2,5]])


# tile 과 broadcasting
# 행렬연산 = 요소 끼리 연산,
# 행렬 곱(내적) = 앞의 열 수 와 뒤의 행 수가 같아야 함.

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
print(x)
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1)) # v를 행으로 4개, 열로 1개 - 반복할 때 tile을 사용함.
print(vv)
y = x + vv
print(y)


# x + v 도 가능함.
# x = 4행 3열
# v = 1행 3열
# => 차수가 다른데, 브로드캐스팅이 되기 때문에. (마치 tile 된 것처럼). 늘어나서 일치시킴.

# x + 3 을 해도 됨.
# 그러면, '3'이 열로 늘어나고 행으로 늘어나서 일치.

x = np.arange(4).reshape((2,2))
print(x)
print(np.transpose(x)) #전치

import numpy as np
                           # 0,1,2
arr = np.arange(24).reshape((2,3,4))
print(arr)
print(arr.transpose((1,0,2)))    #3면 2행 4열
print(arr.transpose((0,2,1)))


arrr = np.arange(32).reshape((8,4))
print(arr)
#다항선택
print(arr[[1,5,7,2]])  # 괄호가 하나만 들어왔다. => 행
print(arr[[-1,-5,-7,-2]]) # 거꾸로
print(arr[[1,5,7,2]][[0,3,1]]) # 콤마가 없는 것 => filtering , 앞에 것을 구하고 나서, 뒤에 것을 구하라.
print(arr[[15,7,2],[0,3,1,2]]) # 콤마가 있는 것 => fancy indexing
print(arr[[1,2,5,7]][:, [0,1,2,3]]) # 뒤에서 행은 모두 선택
print(arr[[1,5,7,2]][:, [0,3,1,2]])

#iterator 반복자
import numpy as np
a = np.arange(0, 60, 5)
print(a)
for x in np.nditer(a):
    print(x)

# 2차원으로 만들더라도, 다시 1차원으로..



names = np.array(['Seoul', 'Daejun', 'chungju', 'seoul', 'chungju',
                  'Daejun', 'Daejun'])

#data = np.random.rand # 0~1 사이의 값
data = np.random.randn(7,4) # 랜덤한 수, 7에서 4       #normal 정규분포에서
print(names=='Seoul')
print(data[names=='Seoul'])  # boolean indexing
print(data[names=='Seoul', 2:])
print(data[names=='Seoul', 3])

print(names != 'Seoul')
print( ~ ( names=='Seoul'))
mask = (names == 'Seoul') | (names=='chungju')
print(mask)
print(data[mask])
