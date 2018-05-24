import csv, sys
from analyze_methods import getStat, calcPMI
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import random

doc = []
csvreader = csv.reader(open(sys.argv[1]), delimiter=' ')
for line in csvreader:
    doc.append(line)

words, words_amount, reviews_count_pos, reviews_count_neg, words_in_pos, words_in_neg, word_occurrence, word_occurrence_pos, word_occurrence_neg = getStat(doc)
reviews_count = len(doc)
local_dictionary_size = len(words)
pmi_pos, pmi_neg = calcPMI(word_occurrence, word_occurrence_pos, word_occurrence_neg, words_in_pos, words_in_neg, local_dictionary_size, words_amount)

f1 = open('analyze.txt', 'w+')
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

with open('pmi.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(words)):
        writer.writerow([words[i],  pmi_pos[i] + pmi_neg[i]])

doc_lex = []
for i in range(int(len(doc) * 3 / 4)):
    rnd = random.choice(doc)
    doc_lex.append(rnd)
    doc.remove(rnd)

words, words_amount, reviews_count_pos, reviews_count_neg, words_in_pos, words_in_neg, word_occurrence, word_occurrence_pos, word_occurrence_neg = getStat(doc_lex) 
local_dictionary_size = len(words)
pmi_pos, pmi_neg = calcPMI(word_occurrence, word_occurrence_pos, word_occurrence_neg, words_in_pos, words_in_neg, local_dictionary_size, words_amount)

with open('chunk_' + sys.argv[1] + '.lex', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(words)):
        writer.writerow([words[i],  pmi_pos[i] + pmi_neg[i]])

with open('chunk_' + sys.argv[1], 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for d in doc:
        writer.writerow(d)
