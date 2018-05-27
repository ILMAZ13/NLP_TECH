import string
import texterra
import logging
import re
import csv
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# from TrainModel import *


class TextAnalyser:
    features = "true"
    feature_3c = 'false'
    word_type = "surface_all"
    language = ""
    features_list = []
    features_names = []
    corpus_name = 'corpus_3c'

    # Global variables
    negation = ["no", "not", "не", "нет"]
    text = ""
    pos_change = 0
    neg_change = 0
    # te = texterra.API('9318d65e0fd6e25b22bdcd3f6adc7bfcce4c8280')
    te = texterra.API(host='http://localhost:8082/texterra/')
    lem = WordNetLemmatizer()

    # Lexicon
    eng_lex_pos = ""
    eng_lex_neg = ""
    eng_lex = ""
    rus_lex = ""
    auto_lex = []

    count = 0

    def __init__(self, language, word_type, features, features_c3):
        self.word_type = word_type
        self.features = features
        self.language = language
        self.feature_3c = features_c3
        if self.language == "english":
            self.eng_lex_pos = open("positive-words.txt").read()
            self.eng_lex_neg = open("negative-words.txt").read()
            self.eng_lex = open("mpqa.tff").readlines()
        if self.language == "russian":
            self.rus_lex = open("rusentilex_2017.txt").readlines()

    def init_c3(self, pmi_file):
        auto_lex_file = csv.reader(open(pmi_file, 'r'), delimiter=',')
        for line in auto_lex_file:
            self.auto_lex.append(line)

    def text_process(self, text_raw):
        self.text = text_raw

        self.count = self.count + 1
        print(self.count)

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

        feature = []
        if self.features == "true":
            feature = self.pos_neg_features(words)
            feature[0] = feature[0] + self.pos_change
            feature[1] = feature[1] + self.neg_change

        if self.feature_3c == 'true':
            feature_al = self.feature_auto_lex(words)
            self.features_names.append('not_zero_score')
            self.features_names.append('total_score')
            self.features_names.append('max_score')
            self.features_names.append('last_score')
            for f in feature_al:
                feature.append(f)

        if self.word_type == 'stem':
            stemmer = SnowballStemmer(self.language)
            words = [stemmer.stem(word) for word in words]

        # Тут возвращается обработанный тескт и для него лист из 6 (для русского
        # языка), либо 10 (для английского языка) фич
        self.features_list.append(feature)
        return words

    def neg_feature(self, text):
        if self.language == 'russian':
            try:
                result = self.te.syntax_detection(text)
                for item in result:
                    tree = item.tree
                    for key in tree.keys():
                        self.find_neg(tree, key)
            except BaseException:
                logging.warning('Error on text: {0}'.format(text))
        if self.language == 'english':
            words = word_tokenize(text)
            words = [word.lower() for word in words if word.lower() not in stopwords.words(self.language)]
            for index in range(len(words)):
                if words[index] in self.negation:
                    self.text = re.sub(r'\b%s\b' % words[index+1], words[index] + "_" + words[index+1], self.text, 1)
                    self.text = re.sub(r'\b%s\b' % words[index], '', self.text, 1)

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
            if self.language == "english":
                lemma = self.lem.lemmatize(word)
                if re.search(r'\b%s\b' % lemma, self.eng_lex_neg) is not None:
                    return "negative"
                if re.search(r'\b%s\b' % lemma, self.eng_lex_pos) is not None:
                    return "positive"
            if self.language == "russian":
                result = self.te.lemmatization(word, rtype='lemma')
                for item in result:
                    lemma = item
                lemma = lemma[0]
                for line in self.rus_lex:
                    if re.search(r'\b%s\b' % lemma, line.split(", ")[2]) is not None:
                        polarity = line.split(", ")[3]
                        if polarity == "positive" or polarity == "negative":
                            return polarity
        except BaseException as e:
            logging.warning('pos_neg: Error on word: {0} {1}'.format(word, str(e)))
        return None

    def pos_neg_additional(self, word):
        if word in string.punctuation:
            return None, None
        try:
            lemma = ''
            lemma = self.lem.lemmatize(word)
            for line in self.eng_lex:
                if re.search(r'\bword1=%s\b' % lemma, line) is not None:
                    strength = re.search(r'type=(.+) l', line).group(1)
                    polarity = re.search(r'polarity=(.+)', line).group(1)
                    if polarity == "positive" or polarity == "negative":
                        return polarity, strength
        except BaseException:
            logging.warning('additional: Error on word: {0}'.format(word))
        return None, None

    def pos_neg_features(self, words):
        features = [0] * 6
        self.features_names = ['count_pos', 'count_neg', 'sum_pos_aff', 'sum_neg_aff', 'sum_pos_neg', 'sum_neg_neg']
        if self.language == "english":
            features = [0] * 10
            self.features_names.append('sum_pos_aff_mpqa')
            self.features_names.append('sum_neg_aff_mpqa')
            self.features_names.append('sum_pos_neg_mpqa')
            self.features_names.append('sum_neg_neg_mpqa')
        count = {}
        count['positive'] = 0
        count['negative'] = 0
        for word in words:
            isNeg = False
            match = re.search(r'^(no|not|не|нет)_(.+)$', word)
            if match is not None:
                isNeg = True
                word = match.group(2)
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

    def auto_lex_search(self, word):
        for line in self.auto_lex:
            if line[0] == word:
                return line[1]
        return None

    def feature_auto_lex(self, words):
        features = [0, 0, -1, 0]
        for word in words:
            score = self.auto_lex_search(word)
            if score is not None:
                score = float(score)
                features[0] += 1
                features[1] += score
                if score > features[2]:
                    features[2] = score
        try:
            last = self.auto_lex_search(words[len(words)-1])
            if last is not None:
                features[3] = last
            return features
        except IndexError:
            return features


if __name__ == '__main__':
    ta = TextAnalyser('russian', 'surface_all', 'true')
    print(ta.text_process('не облаять'))
