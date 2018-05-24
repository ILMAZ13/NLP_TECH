import math
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import texterra

def getStat(docs):

    tokenizer = RegexpTokenizer(r'\w+')
    le = WordNetLemmatizer()
    te = texterra.API("9318d65e0fd6e25b22bdcd3f6adc7bfcce4c8280")
    language = "russian"
    lang = ""
    result = te.language_detection(docs[0][0])
    for item in result:
        lang = item
    if(item == 'ru'):
        language = 'russian'
    if(item == 'en'):
        language = 'english'

    stop_words = set(stopwords.words(language))

    words = []
    word_occurrence = []
    word_occurrence_pos = []
    word_occurrence_neg = []
    reviews_count_pos=0
    reviews_count_neg=0
    words_amount=0
    words_in_pos=0
    words_in_neg=0
    
    for doc in docs:
        review = doc[0]
        tokenize_review = tokenizer.tokenize(review.lower())
        tokenize_review = [w for w in tokenize_review if not w in stop_words]
        temp = []
        if(language == 'english'):
            tokenize_review = [(le.lemmatize(w)) for w in tokenize_review]
        if(language == 'russian'):
            for w in tokenize_review:
                if(w in string.punctuation):
                    temp.append(w)
                    continue
                result = te.lemmatization(w, rtype='lemma')
                for item in result:
                    lemma = item
                    temp.append(lemma)
            tokenize_review = temp
        review_score = doc[1]
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
        
    return words, words_amount, reviews_count_pos, reviews_count_neg, words_in_pos, words_in_neg, word_occurrence, word_occurrence_pos, word_occurrence_neg

def calcPMI(word_occurrence, word_occurrence_pos, word_occurrence_neg, words_in_pos, words_in_neg, local_dictionary_size, words_amount):
    
    pmi_pos = []
    pmi_neg = []

    for i in range(local_dictionary_size):
        value_pos = (word_occurrence_pos[i] * words_amount)/(word_occurrence[i] * words_in_pos)
        value_neg = (word_occurrence_neg[i] * words_amount)/(word_occurrence[i] * words_in_neg)
        if (value_pos != 0):
            pmi_pos.append(math.log2(value_pos))
        else:
            pmi_pos.append(0)
        if (value_neg != 0):
            pmi_neg.append(math.log2(value_neg))
        else:
            pmi_neg.append(0)

    return pmi_pos, pmi_neg
