import string
import texterra
import logging
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# from TrainModel import *


class TextAnalyser:
    features = "true"
    word_type = "surface_all"
    language = ""
    features_list = []

    # Global variables
    negation = ["no", "not", "не", "нет"]
    text = ""
    pos_change = 0
    neg_change = 0
    # te = texterra.API('9318d65e0fd6e25b22bdcd3f6adc7bfcce4c8280')
    te = texterra.API(host='http://localhost:8082/texterra/')

    # Lexicon
    eng_lex_pos = ""
    eng_lex_neg = ""
    eng_lex = ""
    rus_lex = ""

    def __init__(self, language, word_type, features):
        self.word_type = word_type
        self.features = features
        self.language = language
        if self.language == "english":
            self.eng_lex_pos = open("positive-words.txt").read()
            self.eng_lex_neg = open("negative-words.txt").read()
            self.eng_lex = open("mpqa.tff").readlines()
        if self.language == "russian":
            self.rus_lex = open("rusentilex_2017.txt").readlines()

    def text_process(self, text_raw):
        self.text = text_raw

        # language detection
        # result = self.te.language_detection(text)
        # for item in result:
        #     if item == "ru":
        #         self.language = 'russian'
        #     if item == 'en':
        #         self.language = 'english'

        if self.features == "true":
            for neg in self.negation:
                if re.search(r'\b%s\b' % neg, self.text) is not None:
                    self.neg_feature(self.text)
                    break

        words = word_tokenize(self.text)
        words = [word.lower() for word in words if word.lower() not in stopwords.words(self.language)]

        if self.word_type == 'surface_no_pm':
            punc = str.maketrans('', '', string.punctuation)
            words = [word.translate(punc) for word in words]
            words = [word for word in words if word.isalpha()]

        feature = self.pos_neg_features(words)
        feature[0] = feature[0] + self.pos_change
        feature[1] = feature[1] + self.neg_change

        if self.word_type == 'stem':
            stemmer = SnowballStemmer(self.language)
            words = [stemmer.stem(word) for word in words]

        # Тут возвращается обработанный тескт и для него лист из 6 (для русского
        # языка), либо 10 (для английского языка) фич
        self.features_list.append(feature)
        return words

    def neg_feature(self, text):
        try:
            result = self.te.syntax_detection(text)
            for item in result:
                tree = item.tree
                for key in tree.keys():
                    self.find_neg(tree, key)
        except BaseException:
            logging.warning('Error on text: {0}'.format(text))

    def find_neg(self, tree, key):
        node_list = tree[key]
        for node in node_list:
            node_key = list(node.keys())[0]
            if node_key[2] in self.negation:
                self.text = re.sub(r'\b%s\b' % key[2], node_key[2] + "_" + key[2], self.text, 1)
                self.text = re.sub(r'\b%s\b' % node_key[2], '', self.text, 1)
                self.change_metrics(key[2].lower())
            if node[node_key]:
                self.find_neg(node, node_key)

    def pos_neg(self, word):
        if word in string.punctuation:
            return None
        try:
            result = self.te.lemmatization(word, rtype='lemma')
            for item in result:
                lemma = item
            lemma = lemma[0]
            if self.language == "english":
                if re.search(r'\b%s\b' % lemma, self.eng_lex_neg) is not None:
                    return "negative"
                if re.search(r'\b%s\b' % lemma, self.eng_lex_pos) is not None:
                    return "positive"
            if self.language == "russian":
                for line in self.rus_lex:
                    if re.search(r'\b%s\b' % lemma, line.split(", ")[2]) is not None:
                        polarity = line.split(", ")[3]
                        if polarity == "positive" or polarity == "negative":
                            return polarity
        except BaseException:
            logging.warning('Error on word: {0}'.format(word))
        return None

    def pos_neg_additional(self, word):
        if word in string.punctuation:
            return None, None
        try:
            result = self.te.lemmatization(word, rtype='lemma')
            for item in result:
                lemma = item
            lemma = lemma[0]
            for line in self.eng_lex:
                if re.search(r'\bword1=%s\b' % lemma, line) is not None:
                    strength = re.search(r'type=(.+) l', line).group(1)
                    polarity = re.search(r'polarity=(.+)', line).group(1)
                    if polarity == "positive" or polarity == "negative":
                        return polarity, strength
        except BaseException:
            logging.warning('Error on word: {0}'.format(word))
        return None, None

    def pos_neg_features(self, words):
        features = [0] * 6
        if self.language == "english":
            features = [0] * 10
        count = {}
        count['positive'] = 0
        count['negative'] = 0
        for word in words:
            isNeg = False
            match = re.search(r'^(no|not|не|нет)_(.+)$', word)
            if match is not None:
                isNeg = True
                word = word_wo_neg = match.group(2)
            polarity = self.pos_neg(word)
            if polarity is not None:
                count[polarity] = count[polarity] + 1
            features = self.count_score(features, isNeg, polarity, word)

        features[0] = count['positive']
        features[1] = count['negative']

        return features

    def count_score(self, features, isNeg, pol, word):
        polarity = pol
        if not isNeg:
            if polarity == 'positive':
                features[2] += 1
            if polarity == 'negative':
                features[3] -= 1
        else:
            if polarity == 'positive':
                features[4] += 1
            if polarity == 'negative':
                features[5] -= 1

        if self.language == 'english':
            polarity, strength = self.pos_neg_additional(word)
            score = 1
            if strength == 'strongsubj':
                score = 2
            if not isNeg:
                if polarity == 'positive':
                    features[6] += score
                if polarity == 'negative':
                    features[7] -= score
            else:
                if polarity == 'positive':
                    features[8] += score
                if polarity == 'negative':
                    features[9] -= score

        return features

    def change_metrics(self, word):
        polarity = self.pos_neg(word)
        if polarity == "positive":
            self.pos_change = self.pos_change - 1
            self.neg_change = self.neg_change + 1
        if polarity == "negative":
            self.pos_change = self.pos_change + 1
            self.neg_change = self.neg_change - 1


if __name__ == '__main__':
    ta = TextAnalyser('russian', 'surface_all', 'true')
    print(ta.text_process('не облаять'))
