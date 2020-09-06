import nltk
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# print(os.listdir(nltk.data.find('corpora')))


from nltk.corpus import movie_reviews

neg_review=movie_reviews.fileids('neg')
pos_review=movie_reviews.fileids('pos')

# print(len(neg_review))
# print(len(pos_review))
# print(movie_reviews.fileids('pos'))
# print(' ')
# print(movie_reviews.fileids('pos'))

rev= nltk.corpus.movie_reviews.words('pos/cv000_29590.txt')
# print(rev)
rev_list=[]
for rev in neg_review:
    rev_text_neg=nltk.corpus.movie_reviews.words(rev)
    review_one_string=" ".join(rev_text_neg)
    review_one_string =review_one_string.replace(' ,',',')
    review_one_string =review_one_string.replace(' .','.')
    review_one_string =review_one_string.replace("\' ","'")
    review_one_string =review_one_string.replace(" \'","'")
    rev_list.append(review_one_string)
print(len(rev_list))
pos_review=movie_reviews.fileids('pos')