# 나이브베이즈
# 보충::
# 왜 나이브라고 했지요? 나이브베이즈에 나이브는 '독립성 가정'을 가지고 있다는 의미입니다.
# 그것이 모든 '결합 관계'를 고려해야합니다. 영향력이 모두 다릅니다.
# '나이브'라고 하는것은 그것(결합관계)을 모두 고려하지않고, 갯수(카운팅)만 고려한다는 말입니다.
#
# 보충::
#  lbfgs는 수학적최적화 기법. 딥러닝에서는 잘 안 씀.
# =>sklearn은 GPU를 지원하지않겠다.. sklearn 수학적 최적화기에 집중. pandas와 결합하고자 함.
#
#  sgd 확률적 경사하강법(딥러닝/텐서플로)를 씀.
#  adam ( momentum + learning rate처음에는 크게 나중에는 작게)

conda install
conda install nltk

conda install gensim
conda install -c conda-forge pyro4
conda install -c conda-forge jpype1
pip install KoNLPy 안깔림
pip install simplejson
pip install PyTagCloud < fonts(나눔복붙) fonts 마우스 오른쪽 서브라임으로 열어라. -안깔림


위에 하나 복붙
이름Korean
ttf는 나눔바른고딕.ttf는

conda install -c conda-forge wordcloud


#text mining

#특징을 추출해 주는 게 뭐라고 했나요? term frequency, tfidf, hash :어떤걸 hash라고 하죠? 일정공간을 매핑하는 것을 hash라고 합니다.
# term frequency라고 하면,
from sklearn.feature_extraction.text import CountVectorizer
# 문서에서 특징을 추출하기 위한 기본 단어
text = ["The quick brown fox jumped over the lazy dog"] #산림에 대한 책, = 산림 + 일반단어
vectorizer = CountVectorizer() # 핵심단어를 뽑는다.
vectorizer.fit(text)
print(vectorizer.vocabulary_) #단어들
vector= vectorizer.transform(text) #왜 2번을 할까요?
print(vector.shape) #
print(type(vector)) #
print(vector.toarray()) #

print(vectorizer.vocabulary_)



from sklearn.feature_extraction.text import TfidfVectorizer
text = ["The quick brown fox jumped over the lazy dog","brown fox","lazy dog"]#문서를 여러개로 두어보겠습니다.
vectorizer = TfidfVectorizer()
vectorizer.fit(text)
print(vectorizer.vocabulary_)
print(vectorizer.idf_) #총 문서수 / 단어가들어간 문서수  2/3+1
vector = vectorizer.transform([text[0]])
print(vector.shape)
print(vector.toarray())



from sklearn.feature_extraction.text import HashingVectorizer
text = ["The quick brown fox jumped over the lazy dog","brown fox","lazy dog"]
vectorizer = HashingVectorizer(n_features=20) #단어 복원이 안됨, 숫자, 문서를 비교
#메모리를 절약

##nltk의 Stemmer(단어 원형)
#어간 추출을 통해 단어를 기본형으로 변형하는 모듈
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

input_words = ["writing", "calves", "be", "branded", "horse", "randomize",
        "possibly", "provision", "hospital", "kept", "scratchy", "code"]
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')
#Create a list of stemmer names for display
stemmer_names = ["PORTER", "LANCASTER", "SNOWBALL"]
formatted_text = '{:>16} '* (len(stemmer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *stemmer_names),
        '\n', '='*68)

for word in input_words:
    output = [word, porter.stem(word),
            lancaster.stem(word), snowball.stem(word)]
    print(formatted_text.format(*output))



import nltk
nltk.download()
from nltk.corpus import brown
brown.words()[0:10]
brown.tagged_words()[0:10]# 태그이름 'AT'
len(brown.words()) #1161192
from nltk.book import *

text1
text1.concordance("monstrous") #같이 출현하는 단어 bigram으로 출력
text1.similar("monstrous") #근처에서 사용되는 단어
text2.common_contexts(["monstrous","very"]) #같은 근방에 오는 단어들

len(text3)
text4[173]
text4.index('awaken')


sorted(set(text3)) #중복배제
len(set(text3)) #순수단어수
len(set(text3)) / len(text3) #순수 단어의 비율, 어휘력


text3.count("smote")
100* text4.count('a') / len(text4)

def lexical_diversity(text): #어휘력을 함수로
    return len(set(text)) / len(text)

def percentage(count, total): # 전체에서 단어 구성비
    return 100 * count/ total
lexical_diversity(text3)
percentage(text4.count('a'), len(text4))

fdist1 = FreqDist(text1)
print(fdist1)
print(fdist1.most_common(50))
fdist1['whale']


V = set(text1)
long_words = [w for w in V if len(w) > 15]


fdist5 = FreqDist(text5)#길이와 빈도수를 제한
sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7)


from nltk import bigrams # 두단어씩 쌍을 이룸 # bigrams는 두 단어씩 쌍을 이룬다는 거에요.
list(bigrams(['more', 'is', 'said', 'than', 'done']))

sorted(w for w in set(text1) if w.endswith('ableness'))
sorted(term for term in set(text4) if 'gnt' in term)
sorted(w for w in set(sent7) if not w.islower())
sorted(t for t in set(text2) if 'cie' in t or 'cei' in t)


#문장으로 분맇하거나 단어로 분리
from nltk import sent_tokenize, word_tokenize, pos_tag
text = "Machine learning is the science of getting computers to act without beingg explicitly programmed."
sents = sent_tokenize(text, language = 'english')
print(sents)
len(sents)

sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
tokens = nltk.word_tokenize(text)
tokens


#중지는 quit : rule base
from nltk.chat.util import Chat, reflections
pairs = ( (r'나는 (.*) 필요하다',('얼마나 필요하냐', '필요한게 뭔데')),
        (r'너는 왜 (.*)에 있니', ('밥먹으려고','학교에 가려고')) )

eliza_chatbot = Chat(pairs, reflections)
def eliza_chat():
    print("에이콘에 오신걸 환영합니다.")
    eliza_chatbot.converse()
def demo():
    eliza_chat()
demo()



pairs = [
    [
        r"나의 이름은 (.*)",
        ['안녕 %1', '%1 잘있었니'],
    ],
    [
        r"나의 이름은 (.*)(.*)",
        ['안녕 %1', '%1 잘있었니', '%1 안녕 %2 알아'],
    ],
    [
        r'안녕',
        ['안녕하세요','잘있었니','잘가',],
    ],
    [
        r'(.*) (배고파|자니|뭐해)',
        [
            '%1 %2'
        ]
    ],
    [
        r'(.*) (배고파|자니|뭐해) (.*)',
        [
            '%1 %2 %3'
        ]
    ],
    [
        r'(.*)(사랑해)(.*)',
        [
            "안돼 그러는거 아니야.",
            "나는 항상 기다리고 있어.",
            "웃기지마."
        ],
    ],
    [
        r'(사과|포도|참외|수박|맛있는|과일) (.*) (먹고)(.*)',
        [
            "작작먹어",
            "없어",
            "맛있게 먹어",
        ],
    ],
    [
        r'(.*)', #default response if no patterns from above is found
        [
            "미안 '%1'에 응답할 말이 없네, 너 혼자 놀아!",
        ],
    ]
]




reflections = {
    "나는": "당신은",
    "나의": "너의",
    "당신은": "나는",
    "너의": "나의"}

def hugot_bot():
    print("안녕 이름이 뭐니?")
    chat = Chat(pairs, reflections)
    chat.converse()
hugot_bot()




import nltk
from nltk.corpus import movie_reviews
movie_reviews.sents()
sentences = [list(s) for s in movie_reviews.sents()]
sentences[0]
sentences[1]


#도수 #가까이 근접 #벡터 사이의 코사인 유사도를 이용한 가까운 단어를 추출.
from gensim.models.word2vec import Word2Vec
model = Word2Vec(sentences)
model.init_sims(replace=True)
model.similarity('actor', 'actress')
model.similarity('he', 'she')
model.similarity('actor', 'she')
model.similarity('actress', 'she')
model.most_similar('accident', topn=10) #default 값이 10개

# Find the top-N most similar wordsself.
#Positive words contribute positively towards the similarity
#negative words negatively


model.most_similar(positive=['actor', 'she'], negative='actress', topn=1)
model.most_similar(positive=['accident'])




from collections import Counter
import urllib # crawling
import random
import webbrowser # 직접 웹브라우저 컨트롤
from konlpy.tag import Hannanum #품사에 대한 4가지 종류 중 하나
import pytagcloud
nouns = list()
nouns.extend(['대한민국' for t in range(8)]) #내장리스트
nouns.extend(['미국' for t in range(7)])
nouns.extend(['영국' for t in range(7)])
nouns.extend(['일본' for t in range(6)])
nouns.extend(['벨기에' for t in range(6)])
nouns.extend(['독일' for t in range(6)])
nouns.extend(['러시아' for t in range(6)])
nouns.extend(['베트남' for t in range(5)])
nouns.extend(['태국' for t in range(5)])
nouns.extend(['인도' for t in range(5)])
count = Counter(nouns)
tag2 = count.most_common(100)
taglist = pytagcloud.make_tags(tag2, maxsize=50)
pytagcloud.create_tag_image(taglist, 'wordcloud.jpg', size=(900, 600),
        fontname='Korean', rectangular=False)
webbrowser.open('wordcloud.jpg')



###

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Hyungi/Desktop/workplace/dataset/winemag.csv", index_col=0)
df.head()
country = df.groupby("country")
country.describe().head()

plt.figure(figsize=(15,10))
country.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Number of Wines")
plt.show()
text = df.description[0]
wordcloud = WordCloud(max_font_size = 50, max_words = 100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("first_review.png")

문제) description에 있는 모든 데이터를 한꺼번에 처리해서 시각화 해보시오.
text = " ".join(review for review in df.description)
wordcloud = WordCloud( background_color = "white", max_font_size=100, max_words=500).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

## 스톱워드 적용, 원하는 필터단어 추가, 출력하는 이미지 모양 제어.
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

mask = np.array(Image.open("C:/Users/Hyungi/Desktop/workplace/dataset/house.png"))
wc = WordCloud(background_color="white", max_words = 2000, mask=mask,
        stopwords=stopwords, contour_width=3, contour_color='steelblue')
wc.generate(text)
wc.to_file( "wine.png")
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()



# 나라별로 wordcloud를 생성해보시오

usa = " ".join(review for review in df[df["country"]=="US"].description)
fra = " ".join(review for review in df[df["country"]=="France"].description)
mask = np.array(Image.open("C:/Users/Hyungi/Desktop/workplace/dataset/france.png"))
wordcloud_usa = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA",
        max_words=1000, mask=mask).generate(fra)

image_colors = ImageColorGenerator(mask)
# color image를 기반으로 만든 컬러 = mask
plt.figure(figsize=[7,7])
plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()




from konlpy.corpus import kolaw
from konlpy.tag import Twitter
t = Twitter()
ko_con_text = kolaw.open('constitution.txt').read()
print(type(ko_con_text))
ko_con_text[:100]
tokens_ko = t.nouns(ko_con_text)
tokens_ko[:10]
stop_words = ['제', '월', '일', '조', '수', '때', '그', '이', '바', '및', '안']
tokens_ko = [each_word for each_word in tokens_ko if each_word not in stop_words]
#stop_words는 set으로 만듭니다.

ko = nltk.Text(tokens_ko, name="대한민국 헌법")
ko.vocab().most_common(10) #데이터 타입 => 튜플리스트

#딕셔너리로 변환 (해야지 여기서 쓸 수 있음.)
data = ko.vocab().most_common(500)
tmp_data = dict(data)

wordcloud = WordCloud(font_path='c:/Windows/Fonts/malgun.ttf', relative_scaling = 0.1,
        background_color='white',).generate_from_frequencies(tmp_data)
        # generate는 텍스트 사용할 때
        # 이렇게 정리해서 딕셔너리할때는 generate_from_frequencies를 사용한다.
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


nltk 텍스트가 R로 *치자면, corpus입니다.

한글 문서를 웹에서 크롤링한다음, wordcloud를 생성해서 출력해보시오.
(실제 프로젝트할 때 이렇게 합니다.)


import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen("http://www.pythonscraping.com/pages/warandpeace.html")
bsObj = BeautifulSoup(html, "html.parser")

allText = bsObj.findAll(id="text")
allText

text = allText[0].get_text()
wordcloud = WordCloud(max_font_size = 50, max_words = 100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()
