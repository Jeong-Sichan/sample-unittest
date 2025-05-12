import pandas as pd 
from konlpy.tag import Okt #konlpy의 Okt 형태소 분석기를 받아온 후 문장이 들어오면 어근을 분리하는 함수 생성, konlpy패키지 별도 설치 필요
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt #konlpy의 Okt 형태소 분석기를 받아온 후 문장이 들어오면 어근을 분리하는 함수 생성, konlpy패키지 별도 설치 필요

import unittest
from my_preprocessing import get_df, okt_tokenizer, get_tfidf, category_encoding

class TestMyModule(unittest.TestCase):

  def test_get_df(self):
    self.assertEqual(get_df('merged_interview_category.xlsx').iloc[0,1], '데이터 분석가')

  def test_okt_tokenizer(self):
    self.assertEqual(okt_tokenizer('나는 밥을 먹었다'), ['나', '는', '밥', '을', '먹었다'])

  def test_get_tfidf(self):
    self.assertEqual(len(set(get_tfidf('merged_interview_category.xlsx',0,'korean_stopwords.txt')[0])), 19)

  def test_category_encoding(self):
    self.assertEqual(set(category_encoding('merged_interview_category.xlsx',1)), {0, 1, 2, 3, 4})

if __name__ == '__main__':
  unittest.main()