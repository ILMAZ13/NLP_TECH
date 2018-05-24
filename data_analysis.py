import csv, math
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

math.log2(1.124283133003378)
          
reviews = []
scores = []

doc = pd.read_csv('corpN1.csv', delimiter=',', encoding='utf-8', header=-1)
doc = doc[:100]

reviews = doc[0]
scores = doc[1]

#the total number of reviews
reviews_count = len(reviews)
reviews_count_pos = 0
reviews_count_neg = 0

#the total number of tokens in corpus, in positive and negative reviews
word_occurrence = []
word_occurrence_pos = []
word_occurrence_neg = []

#all unique words
words = []
words_in_pos =  0
words_in_neg =  0
words_amount = 0

tokenize_review = ' '
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words("russian"))
ps = PorterStemmer()

for index, review in enumerate(reviews):
    tokenize_review = tokenizer.tokenize(review.lower())
    tokenize_review = [w for w in tokenize_review if not w in stop_words]
    tokenize_review = [(ps.stem(w)) for w in tokenize_review]
    review_score = scores[index]
    if (review_score == 'positive'):
        reviews_count_pos += 1
    if (review_score == 'negative'):
        reviews_count_neg += 1
    for token in tokenize_review:
        if token not in words:
            words.append(token)
            word_occurrence.append(0)
            word_occurrence_pos.append(0)
            word_occurrence_neg.append(0)
        word_index = words.index(token)
        word_occurrence[word_index] += 1
        words_amount += 1
        if (review_score == 'positive'):
            word_occurrence_pos[word_index] += 1
            words_in_pos += 1
        if (review_score == 'negative'):
            word_occurrence_neg[word_index] += 1
            words_in_neg += 1
            
print('proccessing done')

local_dictionary_size = len(words)
pmi_pos = []
pmi_neg = []

f1 = open('data.txt', 'w+')
print('number of reviews: ', file=f1)
print(reviews_count, file=f1)
print('number of words: ', file=f1)
print(words_amount, file=f1)
print('size of dictionary: ', file=f1)
print(local_dictionary_size, file=f1)
print('number of positive reviews: ', file=f1)
print(reviews_count_pos, file=f1)
print('number of negative reviews: ', file=f1)
print(reviews_count_neg, file=f1)
f1.close()

for i in range(local_dictionary_size):
    value_pos = (word_occurrence_pos[i] * words_amount)/(word_occurrence[i] * words_in_pos)
    value_neg = (word_occurrence_neg[i] * words_amount)/(word_occurrence[i] * words_in_pos)
    if (value_pos != 0):
        pmi_pos.append(math.log2(value_pos))
    else:
        pmi_pos.append(0)
    if (value_neg != 0):
        pmi_neg.append(math.log2(value_neg))
    else:
        pmi_neg.append(0)

with open('pmi.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(words)
    writer.writerow(pmi_pos)
    writer.writerow(pmi_neg)

             
