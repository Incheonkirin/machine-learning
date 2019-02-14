import os
os.chdir("C://Users/Hyungi/Desktop/workplace")
os.getcwd()


#영상처리의 원리
import numpy as np
import mahotas

f = np.ones((256, 256), bool)
f[200:,240:] =False
f[128:144, 32:48] = False
gray()
imshow(f)
show()


import pylab as p
dmap = mahotas.distance(f) #제일 가까운 검정색까지의 거리값으로 변환시킨다.
p.imshow(dmap)
p.show()


# 이번에는 이미지처리 한번 해 보겠습니다.
import numpy as np
import mahotas as mh

image = mh.imread("C:/Users/Hyungi/Desktop/workplace/img/lena-ring.jpg")
from matplotlib import pyplot as plt
plt.imshow(image)
plt.show()

image = mh.colors.rgb2grey(image, dtype=np.uint8) #256개의 컬러로 변경
plt.imshow(image)
plt.gray()
thresh = mh.thresholding.otsu(image)
plt.show()
print('Otsu threshold is {}.'.format(thresh))
# Otsu 경계 값은 138이다.
plt.imshow(image > thresh)
plt.show()


# import numpy as np
# import mahotas as mh
#
# image = mh.imread("C:/Users/Hyungi/Desktop/workplace/img/lena-ring.jpg")
# from matplotlib import pyplot as plt
# plt.imshow(image)
# plt.show()
#
# image = mh.colors.rgb2grey(image, dtype=np.uint8)
# plt.imshow(image)
# plt.gray()
# thresh = mh.thresholding.otsu(image)

im16 = mh.gaussian_filter(image, 16) #주변컬러값 16개
plt.imshow(im16)
plt.show() # 색깔이 뭉개져



import mahotas.demos
wally = mahotas.demos.load('Wally')
imshow(wally)
show()

wfloat = wally.astype(float)
r, g, b = wfloat.transpose((2,0,1)) #transpose 시킴. 인덱스 번호는 0, 1, 2 순서로 갑니다. 그런데 지금 '2'가 면으로 왔습니다. ???????????????


w = wfloat.mean(2)# 2개씩 평균을 취해감.
pattern = np.ones((24, 16), float)#직삭각형을 만든 것 입니다. 직사각형 패턴

#셔츠의 패턴
for i in np.arange(2): # i에는 0, 1
    pattern[i::4] = -1 # 4칸씩 건너뛰면서 -1

v = mahotas.convolve(r-w, pattern) #컨볼루션 연산을 패턴을 가지고 합니다. 위에 셔츠의 패턴이 '필터'가 됩니다. #여기에는 stride 개념이 있습니다. 몇칸씩 건너뛸것이냐. 지금 지정이 없으니까 '1픽셀씩 건너뛰면서 = -1 '
mask = (v == v.max()) # v값과, v최고값인 것

mask = mahotas.dilate(mask, np.ones((48, 24))) #dilate 확장=키우라.  원래 24,16인데 48,24로 키웠음.
np.subtract(wally, .8*wally * ~mask[:, :, None], out=wally, casting='unsafe')
imshow(wally)
show()


##
from PIL import Image
img = Image.open('C:/Users/Hyungi/Desktop/workplace/img/image.png')
dim = (100, 100, 400, 400)
crop_img = img.crop(dim)
crop_img.show()


resize_img = img.resize((200, 200))
resize_img.show()
rotated_img = img.rotate(90)
rotated_img.show()
img.save('new_image.png')

from PIL import ImageEnhance
img = Image.open('C:/Users/Hyungi/Desktop/workplace/img/image.png')
enhancer = ImageEnhance.Contrast(img)
new_img = enhancer.enhance(5)
new_img.show()


enhancer = ImageEnhance.Brightness(img)
bright_img = enhancer.enhance(2)
bright_img.show()


import cv2
cv2.__version__
cv2.namedWindow('image')

from matplotlib import pyplot as plt
img = cv2.imread('C:/Users/Hyungi/Desktop/workplace/img/image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200, 3) # canny = edge detection 외곡선 : 값의 변화가 심한곳을 찾아냄.
cv2.imwrite("canny_edges.jpg", edges)
cv2.imshow("canny_edges", edges)
plt.imshow(img)
plt.show()
plt.imshow(edges)
plt.show()



#Contours 등고선 처리 높낮이 - 안을 채울 수도 있고, 라인으로 구성할 수도 있다.
#threshold 임계값을 기준으로 binary 생성하기 위하여, 127을 기준으로 위아래로, 가장 큰 값은 255까지.
thresh_img = cv2.threshold(gray, 127, 255, 0)
im, contours, hierachy = cv2.findContours(thresh_img[1], #v2.RETR_TREE 등고선을 찾고 선후를 고려해서
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #어떤 것을 같은 것으로 볼 것인가 - 근사치를 가지고 게산

# 등고선 출력 , contour index 등고선, 컬러값, 라인타입 - 두께
cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
plt.imshow(img)
plt.show()


#### scikits 패키지 중 하나가 skvideo가 있습니다.
#
# import skvideo.io
# import skvideo.datasets
# import numpy as np
#
# filename = skvideo.datasets.bigbuckbunny()
# filename
#
# vid_in = skvideo.io.FFmpegReader(filename)
# vid_in
#
# data = skvideo.io.ffprobe(filename)['video'] #ffprobe 모든 이미지파일과 동영상파일은 header정보가 있습니다. header정보를 읽는 것이 ffprobe입니다.
# data
#
# rate = data['@r_frame_rate'] #초당 프레임 : 만화영화 초당 15~20프레임. TV는 NTSC방식은 29.7프레임.
# rate
#
# T = np.int(data['@nb_frames']) #몇 프레임인지, 몇 장인지 ? frame rate로 나누면, 시간이 나오겠죠. 시간계산이 가능합니다.
# T
#
# # mpeg4 방식
# vid_out = skvideo.io.FFmpegWriter("corrupted_video.mp4", inputdict={ #mpeg는 압축방식을 이야기합니다. 지금 우리가 쓰는것은 mpeg4에요. 그래서 이 방식으로 저장하는 것이고,
#         '-r':rate,
#     },
#     outputdict={
#         '-vcodev':'libx264', #H264 라고 해서, 압축포맷, 이미지나 화상포멧은 압축해야하는 구나 하고 알면 돼요.
#         '-pix_fmt':'yuv420p', #
#         '-r':rate,
# })
#
# vid_out


import numpy as np
import skvideo.datasets

filename = skvideo.datasets.bigbuckbunny()

vid_in = skvideo.io.FFmpegReader(filename)
data = skvideo.io.ffprobe(filename)['video']
rate = data['@r_frame_rate']
T = np.int(data['@nb_frames'])


vid_out = skvideo.io.FFmpegWriter("corrupted_video.mp4", inputdict={
      '-r': rate,
    },
    outputdict={
      '-vcodec': 'libx264',
      '-pix_fmt': 'yuv420p',
      '-r': rate,
})



for idx, frame in enumerate(vid_in.nextFrame()):
  print("Writing frame %d/%d" % (idx, T))
  if (idx >= (T/2)) & (idx <= (T/2 + 10)):
    frame = np.random.normal(128, 128, size=frame.shape).astype(np.uint8)
  vid_out.writeFrame(frame)
vid_out.close()



# 동영상 로딩
import skvideo.io
import skvideo.datasets
import skvideo.motion
vid = skvideo.io.vread('corrupted_video.mp4')

print("동영상 전체 차수", vid.shape)
for i , shortcut in enumerate(vid):
    print(shortcut)

#동영상 출력
from IPython.display import YouTubeVideo
# from Ipython.display import YouTubeVideo
YouTubeVideo("mc3XGJaDEMc")



#동영상 출력
import cv2
import matplotlib.pyplot as plt
from IPython import display
#
vc = cv2.VideoCapture("corrupted_video.mp4")
if vc.isOpened():
    is_capturing, frame = vc.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    webcam_preview = plt.imshow(frame)
else:
    is_capturing = False
while is_capturing:
    try:
        is_capturing, frame = vc.read()
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            webcam_preview.set_data(frame)
            plt.draw()
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.pause(0.05)
        except :
            is_capturing = False
    except KeyboardInterrupt:
        vc.release()

#
if vc.isOpened(): # try to get the first frame
    is_capturing, frame = vc.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # makes the blues image look real colored
    webcam_preview = plt.imshow(frame)
else:
    is_capturing = False

while is_capturing:
    try:    # Lookout for a keyboardInterrupt to stop the script
        is_capturing, frame = vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # makes the blues image look real colored
        webcam_preview.set_data(frame)
        plt.draw()

        try:    # Avoids a NotImplementedError caused by `plt.pause`
            plt.pause(0.05)
        except Exception:
            pass
    except KeyboardInterrupt:
        vc.release()
#DEBUG:matplotlib.backends:backend nbAgg version unknown



import numpy as np
import cv2

cap = cv2.VideoCapture("corrupted_video.mp4")

#
# while(True):
#     ret, frame = cap.read()
#     if ret is True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     else:
#         continue
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


#
# -*-coding: utf-8 -*-
import cv2

# cap 이 정상적으로 open이 되었는지 확인하기 위해서 cap.isOpen() 으로 확인가능
cap = cv2.VideoCapture(0)

# cap.get(prodId)/cap.set(propId, value)을 통해서 속성 변경이 가능.
# 3은 width, 4는 heigh

print 'width: {0}, height: {1}'.format(cap.get(3),cap.get(4))
cap.set(3,320)
cap.set(4,240)

while(True):
    # ret : frame capture결과(boolean)
    # frame : Capture한 frame
    ret, frame = cap.read()

    if (ret):
        # image를 Grayscale로 Convert함.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
cv2.closeAllWindows()


import cv2
import numpy as np

cap = cv2.VideoCapture('corrupted_video.mp4')
if (cap.isOpened() == False):
    print("에러발생")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


import numpy as np
import cv2

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
        (frame_width, frame_height))

while(True):
    ret, frame = cap.read()
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        continue
    out.write(frame)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()
cv2.closeAllWindows()



import cv2
import numpy as np

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False):
  print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10,
        (frame_width,frame_height))

while(True):
  ret, frame = cap.read()

  if ret == True:

    # Write the frame into the file 'output.avi'
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()



import cv2

cap = cv2.VideoCapture('vtest.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




# -*-coding: utf-8 -*-

import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (640,480))

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        # 이미지 반전,  0:상하, 1 : 좌우
        frame = cv2.flip(frame, 0)

        out.write(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()




import cv2
import numpy as np
import imutils
cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print("에러 발생")
kernel = np.ones((5,5), np.float32) / 25
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = imutils.resize(frame, width=500)
        frame = cv2.flip(frame, 1)
        frame = cv2.filter2D(frame, -1, kernel)
        rows, cols = frame.shape[:2]
        Mat = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1.0)
        frame = cv2.warpAffine(frame, Mat, (cols, rows))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (21, 21), 0)
        cv2.imshow('Frame', gray)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


## 여기서는 openCV를 하려는게 목적이 아니고, CNN을 하려면 이해해야하기 때문에
# 원리를 이해하고 넘어갑니다.



배경을 빼보겠습니다.
import cv2
import numpy as np
import imutils
cap = cv2.VideoCapture(0)
cap
if (cap.isOpened() == False):
    print("에러 발생")
kernel = np.ones((5,5), np.float32) / 25


history = 30
accelerate = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2() #이걸 넣으면 배경이 없어)


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = imutils.resize(frame, width=1000)
        frame = cv2.flip(frame, 1)
        frame = cv2.filter2D(frame, -1, kernel)
        rows, cols = frame.shape[:2]
        Mat = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1.0)
        frame = cv2.warpAffine(frame, Mat,(cols, rows))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (21, 21), 0)
        cv2.imshow('Frame', gray
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()






이미지로 돌아가서

from skimage import data, segmentation, color
from skimage.io import imread
from skimage import data
from skimage.future import graph
from matplotlib import pyplot as plt

img = data.astronaut()
# kmeans를 적용한 클러스터링 구현 알고리즘 'slic' 실제 거리값까지 고려
img_segments = segmentation.slic(img, compactness =30, n_segments=200)
out1 = color.label2rgb(img_segments, img, kind='avg')
segment_graph = graph.rag_mean_color(img, img_segments, mode="similarity")
img_cuts = graph.cut_normalized(img_segments, segment_graph)
normalized_cut_segments = color.label2rgb(img_cuts, img, kind='avg')
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6, 8))
ax[0].imshow(img)
ax[1].imshow(normalized_cut_segments)


for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()



여기에 적용해보겠습니다.

import cv2
import numpy as np
import imutils
cap = cv2.VideoCapture("corrupted_video.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
        (frame_width, frame_height))

if (cap.isOpened()== False):
    print("에러 발생")


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        img_segments = segmentation.slic(frame, compactness =15, n_segments=400)
        out1 = color.label2rgb(img_segments, frame, kind='avg')
        segment_graph = graph.rag_mean_color(frame, img_segments, mode="similarity")
        img_cuts = graph.cut_normalized(img_segments, segment_graph)

        normalized_cut_segments = color.label2rgb(img_cuts, frame, kind='avg')
        out.write(frame)
        #cv2.imshow('Frame',normalized_cut_segments )
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# cv2로 동영상 저장하면 영상에 효과적용하기




#감정분석
# 카운트 확률을 이용해서 분류하는, '카운트 확률 분류기'가 뭐였죠? => 나이브 베이즈 , 중에 뭐였죠? multinomial -이죠.
# 5점 척도로 평가
# 0 - negative 1 - somewhat negative  2- neutr al 3 - somewhat positive 4 - positive
#

# 카운터에서 중요도를 고려하는 것 ? inversedocumentfrequency, idf. 그리고 hash로 카운터 하는 것도 있었지.
# 나이브에서는 파라미터 튜닝을 안하고 있죠? 파라미터 튜닝하는 것을 찾아보고.
# 전체적으로 찾아보라는 말이에요, 내가 할 수 있는게 뭐가 있나...

import pandas as pd
data = pd.read_csv('senti.tsv', sep='\t')
data.head()

data[:3].Pharse
data.iloc[0, 2]
data.info()

data.Sentiment.value_counts() # 119분석할 때 많이 썼지요? 어느 동네에서 사고가 많이 났는지
Sentiment_count=data.groupby('Sentiment').count()
print(type(Sentiment_count))
print(Sentiment_count.columns)
print(Sentiment_count['Pharse'].head(10))
plt.bar(Sentiment_count.index.values, Sentiment_count['Pharse'])
plt.xlabel('감정종류')
plt.ylabel('리뷰의 갯수')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

#RegexpTokenizer는 못보던거죠? Regexp는 정규표현식이죠, 정규표현식에 있는 것만 찾으라는 이야기에요.
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,1),
        tokenizer = token.tokenize)
text_counts = cv.fit_transform(data['Parse'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( text_counts, data['Sentiment'],
        text_size=0.3, random_state=123)


# Text와 딱 맞는 MultinomialNB.
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(X_test, predicted))


=>개선해 보시오


import numpy as np
import cv2

def showImage():
    imgfile= "C:/Users/Hyungi/Desktop/workplace/img/suzi.jpg"
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)

    cv2.imshow('model', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
showImage()




import numpy as np
import cv2

def showImage():
    imgfile = "C:/Users/Hyungi/Desktop/workplace/img/suzi.jpg"
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('model', cv2.WINDOW_NORMAL)
    cv2.imshow('model', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

showImage()




import numpy as np
import cv2

def showImage():
    imgfile = "C:/Users/Hyungi/Desktop/workplace/img/suzi.jpg"
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    cv2.imshow('model', img)

    k = cv2.waitKey(0) & 0xFF

    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('c'):
        cv2.imwrite("C:/Users/Hyungi/Desktop/workplace/img/suzi_copy.jpg", img)
        cv2.destroyAllWindows

showImage()



import numpy as np
import cv2
import matplotlib.pyplot as plt

def showImage():
    imgfile = "C:/Users/Hyungi/Desktop/workplace/img/suzi.jpg"
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.title('model')
    plt.show()

showImage()



import numpy as np
import cv2

def showVideo():
    try:
        print("카메라를 구동합니다.")
        cap = cv2.VideoCapture(0)
        cap.open(0)
        print("카메라 오픈여부",cap.isOpened())

    except:
        print("카메라 구동 실패")
        return
    cap.set(3, 480)
    cap.set(4, 320)

    while True:
        cap.open(0)
        print(cap.isOpened())

        ret, frame = cap.read()
        print ("ret",ret)
        print ("frame",frame)
        # print(cap.isOpened())
        # cap.open(0)

        if not ret:
            print("비디오 읽기 오류")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2COLOR_BGR2GRAY)
        cv2.imshow('video', gray)

        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

showVideo()




from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import string
import operator
from collections import Counter

def cleanSentence(sentence):
    sentence = sentence.split( ' ')
    sentence = [word.strip(string.punctuation+ string.whitespace) for word in sentence]
    sentence = [word for word in sentence if len(word) > 1 or (word.lower() == 'a' or word.lower() == 'i')]

    return sentence




def cleanInput(content):
    content = content.upper()
    content = re.sub('\n', ' ', content)
    content = bytes(content , 'UTF-8')
    content = content.decode('ascii', 'ignore')
    sentences = content.split('. ')
    return [cleanSentence(sentence) for sentence in sentences]





def getNgramsFromSentence(content, n):
    output = []
    for i in range(len(content)-n + 1):
        output.append(content[i:i+n])
    return output


def getNgrams(content,  n):
    content = cleanInput(content)
    ngrams = Counter()
    ngrams_list = []
    for sentence in content:
        newNgrams = [' '.join(ngram) for ngram in getNgramsFromSentence(sentence, n)]
        ngrams_list.extend(newNgrams)
        ngrams.update(newNgrams)
    return (ngrams)


content = str(urlopen('http://pythonscraping.com/files/inaugurationSpeech.txt').read(), 'utf-8')
ngrams = getNgrams(content, 3)

print(ngrams)






from urllib.request import urlopen
from bs4 import BeautifulSoup

def getNgrams(content, n):
  content = content.split(' ')
  output = []
  for i in range(len(content)-n+1):
    output.append(content[i:i+n])
  return output

html = urlopen('http://en.wikipedia.org/wiki/Python_(programming_language)')
bs = BeautifulSoup(html, 'html.parser')
content = bs.find('div', {'id':'mw-content-text'}).get_text()
ngrams = getNgrams(content, 2)
print(ngrams)
print('2-grams count is: '+str(len(ngrams)))





import re

def getNgrams(content, n):
    content = re.sub('\n|[[\d+\]]', ' ', content)
    content = bytes(content, 'UTF-8')
    content = content.decode('ascii', 'ignore')
    content = content.split(' ')
    content = [word for word in content if word != '']
    output = []
    for i in range(len(content)-n+1):
        output.append(content[i:i+n])
    return output


html = urlopen('http://en.wikipedia.org/wiki/Python_(programming_language)')
bs = BeautifulSoup(html, 'html.parser')
content = bs.find('div', {'id':'mw-content-text'}).get_text()
ngrams = getNgrams(content, 2)
print(ngrams)
print('2-grams count is: '+str(len(ngrams)))
