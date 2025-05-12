import pandas as pd 
from konlpy.tag import Okt #konlpy의 Okt 형태소 분석기를 받아온 후 문장이 들어오면 어근을 분리하는 함수 생성, konlpy패키지 별도 설치 필요
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt #konlpy의 Okt 형태소 분석기를 받아온 후 문장이 들어오면 어근을 분리하는 함수 생성, konlpy패키지 별도 설치 필요



def get_df(file): #파일명을 입력값으로 받는다.
    interview_data=pd.read_excel(file)
    return interview_data




def okt_tokenizer(sentence):
    okt=Okt() #Okt 형태소 분석기를 사용하기 위해 Okt()로 객체를 생성한다.
    okt_token=okt.morphs(sentence) #.morphs로 텍스트를 형태소 단위로 나눈다
    return okt_token


def get_tfidf(data_file,sentence_column,stop_words_file): #data_file=인터뷰 데이터 파일,sentence_column=텍스트 데이터에 해당하는 컬럼 인덱스,stop_words_file=불용어 파일
    
    interview_data=pd.read_excel(data_file)
    question_cleasinged=interview_data.iloc[:,sentence_column].values.astype('str')
    
    stop_words=pd.read_csv(stop_words_file)
    stop_words_interview=stop_words['불용어'].tolist()
    
    tfidf=TfidfVectorizer(tokenizer=okt_tokenizer,ngram_range=(1,2),max_df=1.0,min_df=0.0, stop_words=stop_words_interview,lowercase=False)
    tfidf.fit(question_cleasinged)
    tfidf_interview_data=tfidf.transform(question_cleasinged)
    tfidf_interview_data
    
    return tfidf_interview_data.toarray()

def category_encoding(file, category_index): #입력값으로 데이터 프레임과 인코딩할 컬럼의 인덱스를 받는다.
    df=pd.read_excel(file) 
    category=df.iloc[:,category_index].values
    encoder = LabelEncoder()
    encoder.fit(category)
    category_encoded = encoder.transform(category)
    return category_encoded

