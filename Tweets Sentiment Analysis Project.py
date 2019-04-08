# -*- coding: utf-8 -*-
"""
Created on Fri May  4 02:23:45 2018

@author: maiga
"""

###################---------------Libraries Importing---------------###################

import re 
import csv
import pandas as pd
from nltk.corpus import wordnet
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
import timeit
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFpr
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from autocorrect import spell
import ftfy
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.tag import pos_tag
from sklearn.model_selection import StratifiedKFold
from gensim.models import KeyedVectors
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

###################---------------Lexicons and Models Preparation---------------###################

#Reading slang lexicon document
slang_lexicon = []
with open('Slang_Lexicon.txt', 'r') as f:
    for line in f:
        items = line.casefold().strip().split()
        word_1 = ' '.join(items[1:len(items)])
        items = [items[0], word_1]
        slang_lexicon.append(items)
f.close()

#Reading positive and negative words files
pos_file = open("positive-words.txt","r") 
neg_file = open("negative-words.txt","r")
pos_words = []
neg_words = []
#Each line in positive and negative text file is word + \n
#so i split the line with '\n' to extract the word only
for line in pos_file:
    line = line.split('\n')
    pos_words.append (line[0])
for line in neg_file: 
    line = line.split('\n')
    neg_words.append (line[0])
    
#Emoticons
emoticons = \
	[	('EMOT_POS',	[':-)', ':)', '(:', '(-:', ] )	,\
		('EMOT_POS',		[':-D', ':D', 'X-D', 'XD', 'xD', ] )	,\
		('EMOT_POS',		['<3', ':\*', ] )	,\
		('EMOT_POS',		[';-)', ';)', ';-D', ';D', '(;', '(-;', ] )	,\
		('EMOT_NEG',		[':-(', ':(', '(:', '(-:', ] )	,\
		('EMOT_NEG',		[':,(', ':\'(', ':"(', ':(('] )	,\
	]

   
#For emoticon regexes
def escape_paren(arr):
	return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

def regex_union(arr):
	return '(' + '|'.join( arr ) + ')'

emoticons_regex = [ (repl, re.compile(regex_union(escape_paren(regx))) ) \
					for (repl, regx) in emoticons ]

#Word Embeddings Model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 

#Vader Model
analyser = SentimentIntensityAnalyzer()

###################---------------Techniques and Features Functions---------------###################

def lower_case(tweet):
    tweet = tweet.casefold()
    return tweet

def stemDocs(doc):
    stemmed_docs = []

    curr = ""
    for word in doc.split():
        curr = curr + PorterStemmer().stem(word) + " "
    curr = curr.strip()
    stemmed_docs.append(curr)

    return curr

def lemmatize_tweet(tweet):
    tweet = tweet.strip().split()
    tweet = [WordNetLemmatizer().lemmatize(word, pos="v") for word in tweet]
    tweet = ' '.join(tweet)
    return tweet

def stopWordRemoval(doc):
    stops = set(stopwords.words("english"))
    curr = ""
    for word in re.split("\W+", doc):
        if word not in stops:
            curr = curr + word + " "
    curr = curr.strip()
    return curr

def removeUnicode(text):
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

def removeEmoticons(text):
    for (repl, regx) in emoticons_regex:        
        text = re.sub(regx, r'', text)
    return text

def countEmoticons(text):
    pos_emo = 0
    neg_emo = 0
    for (repl, regx) in emoticons_regex:
        text = re.sub(regx, ' '+repl+' ', text)
    text = text.strip().split()
    if 'EMOT_POS' in text:
        for word in text:
            if word == 'EMOT_POS':
                pos_emo += 1
    if 'EMOT_NEG' in text:
        for word in text:
            if word == 'EMOT_NEG':
                neg_emo += 1
    return pos_emo, neg_emo

def removeURL(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|([\w\/\.]*\.com[\w\/\.\?\=\#]*))',r'',text)
    return text

def removeHashTag(text):
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def countURL(text):
    count_url_train = 0
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|([\w\/\.]*\.com[\w\/\.\?\=\#]*))','url',text)
    text = text.strip().split()
    if 'url' in text:
        for word in text:
            if word == 'url':
                count_url_train += 1
    return count_url_train

def countHashTag(text):
    count_hash_train = 0
    text = re.sub(r'#([^\s]+)', r'hash \1', text)
    text = text.strip().split()
    if 'hash' in text:
        for word in text:
            if word == 'hash':
                count_hash_train += 1
    return count_hash_train

def removeAtUser(text):
    text = re.sub('@([^\s]+)',r'\1',text)
    return text

def countAtUser(text):
    count_at = 0
    text = re.sub('@([^\s]+)',r'atUser \1',text)
    text = text.strip().split()
    if 'atUser' in text:
        for word in text:
            if word == 'atUser':
                count_at += 1
    return count_at

def removeNumbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

def replace_punc(row):
    row = row.strip().split()
    for i in range(0, len(row)):
        if "!!" in row[i]:
            word = row[i]
            index = word.find('!')
            word_temp = list(word)   
            word_temp[index] = ' MultiExclamationMarks'
            word = ''.join(word_temp)
            word = word.replace('!', '')
            row[i] = word
        if "??" in row[i]:
            word = row[i]
            index = word.find('?')
            word_temp = list(word)   
            word_temp[index] = ' MultiQuestionMarks'
            word = ''.join(word_temp)
            word = word.replace('?', '')
            row[i] = word
        if ".." in row[i]:
            word = row[i]
            index = word.find('.')
            word_temp = list(word)   
            word_temp[index] = ' MultiStopMarks'
            word = ''.join(word_temp)
            word = word.replace('.', '')
            row[i] = word
    row = ' '.join(row)
    return row

def removePunct(text):
    freePunct = []
    s = text.translate(string.punctuation)
    translate_table = dict((ord(char), None) for char in string.punctuation)
    freePunct.append(s.translate(translate_table))
    return s.translate(translate_table)

def count_punc(tweet):
    count_e = 0
    count_q = 0
    count_s = 0
    for token in tweet:
        if token == '!':
            count_e = count_e + 1
        if token == '?':
            count_q = count_q + 1
        if token == '.':
            count_s = count_s + 1
    return count_e, count_q, count_s

def count_cap_trainital(tweet):
    num = 0
    tweet = tweet.strip().split()
    for i in range(0, len(tweet)):
        word = tweet[i]
        if len(word) > 2:
            if word.isupper():
                num += 1
    return num

def replace_slang(tweet, slang_lexicon):
    tweet = tweet.strip().split()
    tweet_join = ' '.join(tweet)
    tweet_lower = lower_case(tweet_join)
    tweet_lower = tweet_lower.strip().split()
    for abr in slang_lexicon:
        if abr[0] in tweet_lower:
            for i in range(0, len(tweet_lower)):
                if tweet_lower[i] == abr[0]:
                    word = tweet_lower[i]
                    word = word.replace(abr[0], abr[1])
                    tweet[i] = word
    tweet = ' '.join(tweet)
    return tweet

def count_slang(tweet, slang_lexicon):
    tweet = tweet.strip().split()
    tweet_join = ' '.join(tweet)
    tweet_lower = lower_case(tweet_join)
    tweet_lower = tweet_lower.strip().split()
    count = 0
    for abr in slang_lexicon:
        if abr[0] in tweet_lower:
            for i in range(0, len(tweet_lower)):
                if tweet_lower[i] == abr[0]:
                    count = count + 1
    return count

def replaceElongated_word(word, flag):
    """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """

    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:   
        flag = 1
        [repl_word, flag] = replaceElongated_word(repl_word, flag)
    return repl_word, flag

def replaceElongated_tweet(row):
    row = row.strip().split()
    for i in range (0, len(row)):
        word = row[i]
        [word, elongated_flag] = replaceElongated_word(word, flag=0)
        row[i] = word
    row = ' '.join(row)
    return row

def countElongated(tweet):
    tweet = tweet.strip().split()
    count = 0
    long_check = re.compile("([a-zA-Z])\\1{2,}")
    for i in range (0, len(tweet)):
        word = tweet[i]
        elongated_flag = bool(long_check.search(word))
        if elongated_flag == True:
            count = count + 1
    return count

def count_pos_neg(tweet):
    tweet = tweet.strip().split()
    count_pos_train = 0
    count_neg_train = 0
    for i in range (len(tweet)):
        if tweet[i] in pos_words:
            count_pos_train += 1
        elif tweet[i] in neg_words:
            count_neg_train += 1
    return count_pos_train, count_neg_train

def pos_tag_count(tweet):
    tag_list = Counter([j for i,j in pos_tag(word_tokenize(tweet))])
    count_nn = tag_list['NN']
    count_vb = tag_list['VB']
    count_jj = tag_list['JJ']
    count_rb = tag_list['RB']
    return count_nn, count_vb, count_jj, count_rb

def word_embeddings(tweet):
    count = 0
    word2vec_model_sum = np.zeros([1, 300])
    for i in range(len(tweet)):
        try:
            word2vec_model_sum = np.add(word2vec_model_sum , model.wv[tweet[i]])
            count += 1
        except KeyError:
            count += 0
    word2vec_model_sum = np.divide(word2vec_model_sum , count) #average word2vec per tweet
    return word2vec_model_sum

def vader_scores(sentence):
    vader_dict = analyser.polarity_scores(sentence)
    vader_list = list(vader_dict.values())
    return vader_list
    
def word_correct(tweet):
    tweet = tweet.strip().split()
    for i in range (0, len(tweet)):
        word = tweet[i]
        if not wordnet.synsets(word):
            word = spell(word)
            tweet[i] = word
    tweet = ' '.join(tweet)
    return tweet

def replace(word, pos=None):
    antonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
    if len(antonyms) == 1:
        return antonyms.pop()
    else:
        return "not exists"

def replaceNegations(text):
    text = text.strip().split()
    i, l = 0, len(text)
    words_r = []
    while i < l:
      word = text[i]
      if word == 'not' and i+1 < l:
        ant = replace(text[i+1])
        ant = str(ant)
        if ant != "not exists":
          words_r.append(ant)
          i += 2
          continue
      words_r.append(word)
      i += 1
    words_r = ' '.join(words_r)
    return words_r

###################---------------Features Addition Function---------------###################

def features_append(features_list, elongated_f, exclamation_f, question_f, stop_f,
                    cap_f, posWord_f, negWord_f, word_f, slang_f, posEmo_f, negEmo_f,
                    url_f, hash_f, atUser_f, noun_f, adj_f, adverb_f, verb_f, 
                    vader_f, wordEmb_f, elongated_count, count_exclamation, 
                    count_question, count_stop, count_cap, count_pos, count_neg, 
                    count_words, count_slg, count_neg_emo, count_pos_emo, count_url, 
                    count_hash, count_user, count_noun, count_adj, count_adv, 
                    count_verb, vader_matrix, word2vec_models_sum):
    
    for row in range(0, len(features_list)):
        
            if elongated_f == 1:
                features_list[row].append(elongated_count[row])
    
            if exclamation_f == 1:
                features_list[row].append(count_exclamation[row])
                
            if question_f == 1:
                features_list[row].append(count_question[row])
                
            if stop_f == 1:
                features_list[row].append(count_stop[row])
                
            if cap_f == 1:
                features_list[row].append(count_cap[row])
            
            if posWord_f == 1:
                features_list[row].append(count_pos[row])
                
            if negWord_f == 1:
                features_list[row].append(count_neg[row])
                
            if word_f == 1:
                features_list[row].append(count_words[row])
            
            if slang_f == 1:
                features_list[row].append(count_slg[row])
                
            if posEmo_f == 1:
                features_list[row].append(count_neg_emo[row])
                
            if negEmo_f == 1:
                features_list[row].append(count_pos_emo[row])
                
            if url_f == 1:
                features_list[row].append(count_url[row])
                
            if hash_f == 1:
                features_list[row].append(count_hash[row])
                
            if atUser_f == 1:
                features_list[row].append(count_user[row])
                
            if noun_f == 1:
                features_list[row].append(count_noun[row])
                
            if adj_f == 1:
                features_list[row].append(count_adj[row])
                
            if adverb_f == 1:
                features_list[row].append(count_adv[row])
                
            if verb_f == 1:
                features_list[row].append(count_verb[row])
    
    features_list = pd.DataFrame(features_list) 

    if vader_f == 1:
        vader_matrix = pd.DataFrame(vader_matrix)
        features_list = pd.concat([vader_matrix, features_list], axis=1)
       
    if  wordEmb_f == 1:
        word2vec_models_sum = pd.DataFrame(word2vec_models_sum)
        features_list = pd.concat([word2vec_models_sum, features_list], axis=1)
        
    return features_list

###################---------------Variables Initialization---------------###################

#1 is given for the methods used in the model, and 0 for non-used methods

#Techniques List 
fix_t = 0
lower_t = 0
stem_t = 0
lemmatize_t = 0
removeStopword_t = 0
removeEmoticons_t = 0
removeUnicode_t = 0
removeNumbers_t = 1
removeURL_t = 0
removeHashTag_t = 0
removeAtUser_t = 0
replaceContraction_t = 0
replacePunc_t = 0
removePunct_t = 0
replaceSlang_t = 0
replaceElongated_t = 0
replaceNegations_t = 1
wordCorrect_t = 0

#Features List
elongated_f = 1
exclamation_f = 1
question_f = 1
stop_f = 0
cap_f = 0
posWord_f = 1
negWord_f = 1
word_f = 0
slang_f = 0
negEmo_f = 0
posEmo_f = 1
url_f = 0
hash_f = 0
atUser_f = 0
noun_f = 1
adj_f = 0
adverb_f = 0
verb_f = 0
vader_f = 1
wordEmb_f = 0

#Models and Methods List
count_vect = 1
tfidf_vect = 0
wordEmb_vect = 0
reduction_1 = 0
reduction_2 = 1
normalization = 0

###################---------------Classification Pipeline Implementation---------------###################

#Start Computation Time Calculation
start = timeit.default_timer()

#Training Lists Initialization
raw_tweets_train = []
tweets_split_train = []
correct_decision = []
word2vec_models_sum_train = []
vader_matrix_train = []
count_exclamation_train = []
count_question_train = []
count_stop_train = []
elongated_count_train = []
count_cap_train = []
count_pos_train = []
count_neg_train = []
count_words_train = []
count_slg_train = []
count_pos_emo_train = []
count_neg_emo_train = []
count_url_train = []
count_hash_train = []
count_user_train = []
count_noun_train = []
count_verb_train = []
count_adj_train = []
count_adv_train = []
dataset_weka_train = []

#Test Lists Initialization
raw_tweets_test = []
tweets_split_test = []
tweets_id_test = []
word2vec_models_sum_test = []
vader_matrix_test = []
count_exclamation_test = []
count_question_test = []
count_stop_test = []
elongated_count_test = []
count_cap_test = []
count_pos_test = []
count_neg_test = []
count_words_test = []
count_slg_test = []
count_pos_emo_test = []
count_neg_emo_test = []
count_url_test = []
count_hash_test = []
count_user_test = []
count_noun_test = []
count_verb_test = []
count_adj_test = []
count_adv_test = []
dataset_weka_test = []

#Training data preprocessing techniques and features implementation
for i in range (0, 3):
    if i == 0:
        f = open("twitter-2013train.txt", 'r')
    if i == 1:
        f = open("twitter-2015train.txt", 'r')
    if i == 2:
        f = open("twitter-2016train.txt", 'r')
        
    for line in f.readlines():
    
        correct_decision.append(line.split('\t')[1])
        line_weka = line.strip().split(None, 1)[1]
        dataset_weka_train.append(line_weka.strip().split(None, 1))
    
        line = line.split('\t')[2]
        line_split = line.strip().split()
        
        if elongated_f == 1:
            count_elong = countElongated(line)
            elongated_count_train.append(count_elong)

        if exclamation_f == 1 or question_f == 1 or stop_f == 1:
            [count_e, count_q, count_s] = count_punc(line)
            count_exclamation_train.append(count_e)
            count_question_train.append(count_q)
            count_stop_train.append(count_s)
            
        if cap_f == 1:
            count_c = count_cap_trainital(line)
            count_cap_train.append(count_c)
        
        if posWord_f == 1 or negWord_f == 1:
            [count_p, count_n] = count_pos_neg(line)
            #count_p = count_p*vader_vec[3]
            #count_n = count_n*vader_vec[1]
            count_pos_train.append(count_p)
            count_neg_train.append(count_n)
            
        if word_f == 1:
            count_w = len(line_split)
            count_words_train.append(count_w)
        
        if slang_f == 1:
            count_s = count_slang(line, slang_lexicon)
            count_slg_train.append(count_s)
            
        if posEmo_f == 1 or negEmo_f == 1:
            [count_p_e, count_n_e] = countEmoticons(line)
            count_pos_emo_train.append(count_p_e)
            count_neg_emo_train.append(count_n_e)
            
        if url_f == 1:
            count_ur = countURL(line)
            count_url_train.append(count_ur)
            
        if hash_f == 1:
            count_h = countHashTag(line)
            count_hash_train.append(count_h)
            
        if atUser_f == 1:
            count_us = countAtUser(line)
            count_user_train.append(count_us)
            
        if noun_f == 1 or adj_f == 1 or adverb_f == 1 or verb_f == 1:
            [count_nn, count_vb, count_jj, count_rb] = pos_tag_count(line)
            count_noun_train.append(count_nn)
            count_verb_train.append(count_vb)
            count_adj_train.append(count_jj)
            count_adv_train.append(count_rb)
                
        if fix_t == 1:
            line = ftfy.fix_text(line)
            
        if lower_t == 1:
            line = lower_case(line)
            
        if stem_t == 1:
            line = stemDocs(line)
        
        if lemmatize_t == 1:
            line = lemmatize_tweet(line)   
            
        if removeStopword_t == 1:
            line = stopWordRemoval(line)
        
        if removeEmoticons_t == 1:
            line = removeEmoticons(line)

        if removeUnicode_t == 1:
            line = removeUnicode(line)
            
        if removeNumbers_t == 1:
            line = removeNumbers(line)

        if removeURL_t == 1:
            line = removeURL(line)

        if removeHashTag_t == 1:
            line = removeHashTag(line)   
            
        if removeAtUser_t == 1:
            line = removeAtUser(line)
        
        if replaceContraction_t == 1:
            line = replaceContraction(line)

        #There is dependency between replace_punc & removePunc
        if replacePunc_t == 1:
            line = replace_punc(line)
            
        if removePunct_t == 1:
            line = removePunct(line)
            
        if replaceSlang_t == 1:
            line = replace_slang(line, slang_lexicon)
            
        if replaceElongated_t == 1:
            line = replaceElongated_tweet(line)
            
        if replaceNegations_t == 1:
            line = replaceNegations(line)
            
        if wordCorrect_t == 1:
            line = word_correct(line)

        if vader_f == 1:
           vader_vec = vader_scores(line)
           vader_matrix_train.append(vader_vec)
           
        if  wordEmb_f == 1 or wordEmb_vect == 1:
            word_emb_vec = word_embeddings(line)
            word_emb_vec = word_emb_vec.tolist()
            word_emb_vec = [item for sublist in  word_emb_vec for item in sublist]
            word2vec_models_sum_train.append(word_emb_vec)
            
        raw_tweets_train.append(line)
        tweets_split_train.append(line_split)

correct_decision = pd.DataFrame(correct_decision) #Labels list

#Testing data preprocessing techniques and features implementation
f = open("new_english_test.txt", 'r')

for line in f.readlines():

    tweets_id_test.append(line.strip().split(',', 1)[0])
    line = line.strip().split(',', 1)[1]
    #line = line.strip('"')
    line_split = line.strip().split()
    dataset_weka_test.append(line)
    
    if elongated_f == 1:
        count_elong = countElongated(line)
        elongated_count_test.append(count_elong)

    if exclamation_f == 1 or question_f == 1 or stop_f == 1:
        [count_e, count_q, count_s] = count_punc(line)
        count_exclamation_test.append(count_e)
        count_question_test.append(count_q)
        count_stop_test.append(count_s)
        
    if cap_f == 1:
        count_c = count_cap_trainital(line)
        count_cap_test.append(count_c)
    
    if posWord_f == 1 or negWord_f == 1:
        [count_p, count_n] = count_pos_neg(line)
        #count_p = count_p*vader_vec[3]
        #count_n = count_n*vader_vec[1]
        count_pos_test.append(count_p)
        count_neg_test.append(count_n)
        
    if word_f == 1:
        count_w = len(line_split)
        count_words_test.append(count_w)
    
    if slang_f == 1:
        count_s = count_slang(line, slang_lexicon)
        count_slg_test.append(count_s)
        
    if posEmo_f == 1 or negEmo_f == 1:
        [count_p_e, count_n_e] = countEmoticons(line)
        count_pos_emo_test.append(count_p_e)
        count_neg_emo_test.append(count_n_e)
        
    if url_f == 1:
        count_ur = countURL(line)
        count_url_test.append(count_ur)
        
    if hash_f == 1:
        count_h = countHashTag(line)
        count_hash_test.append(count_h)
        
    if atUser_f == 1:
        count_us = countAtUser(line)
        count_user_test.append(count_us)
        
    if noun_f == 1 or adj_f == 1 or adverb_f == 1 or verb_f == 1:
        [count_nn, count_vb, count_jj, count_rb] = pos_tag_count(line)
        count_noun_test.append(count_nn)
        count_verb_test.append(count_vb)
        count_adj_test.append(count_jj)
        count_adv_test.append(count_rb)
            
    if fix_t == 1:
        line = ftfy.fix_text(line)
        
    if lower_t == 1:
        line = lower_case(line)
        
    if stem_t == 1:
        line = stemDocs(line)
    
    if lemmatize_t == 1:
        line = lemmatize_tweet(line)   
        
    if removeStopword_t == 1:
        line = stopWordRemoval(line)
    
    if removeEmoticons_t == 1:
        line = removeEmoticons(line)

    if removeUnicode_t == 1:
        line = removeUnicode(line)
        
    if removeNumbers_t == 1:
        line = removeNumbers(line)

    if removeURL_t == 1:
        line = removeURL(line)

    if removeHashTag_t == 1:
        line = removeHashTag(line)   
        
    if removeAtUser_t == 1:
        line = removeAtUser(line)
    
    if replaceContraction_t == 1:
        line = replaceContraction(line)

    #There is dependency between replace_punc & removePunc
    if replacePunc_t == 1:
        line = replace_punc(line)
        
    if removePunct_t == 1:
        line = removePunct(line)
        
    if replaceSlang_t == 1:
        line = replace_slang(line, slang_lexicon)
        
    if replaceElongated_t == 1:
        line = replaceElongated_tweet(line)
        
    if replaceNegations_t == 1:
        line = replaceNegations(line)
        
    if wordCorrect_t == 1:
        line = word_correct(line)

    if vader_f == 1:
       vader_vec = vader_scores(line)
       vader_matrix_test.append(vader_vec)
       
    if  wordEmb_f == 1 or wordEmb_vect == 1:
        word_emb_vec = word_embeddings(line)
        word_emb_vec = word_emb_vec.tolist()
        word_emb_vec = [item for sublist in  word_emb_vec for item in sublist]
        word2vec_models_sum_test.append(word_emb_vec)
        
    raw_tweets_test.append(line)
    tweets_split_test.append(line_split)

#Preparing dataset for weka
for row in dataset_weka_train:
    if '"' in row[1]:
        word = row[1]
        word = word.replace('"', '')
        row[1] = word
        
for row in dataset_weka_test:
    if '"' in row:
        word = row
        word = word.replace('"', '')
        row = word
        
dataset_weka_train_dataframe = pd.DataFrame(dataset_weka_train)
dataset_weka_train_dataframe.to_csv("processed_tweets_train.csv", index=False, quoting= csv.QUOTE_ALL)
dataset_weka_test_dataframe = pd.DataFrame(dataset_weka_test)
dataset_weka_test_dataframe.to_csv("processed_tweets_test.csv", index=False, quoting= csv.QUOTE_ALL)

#Models Implementation
if count_vect == 1:    
    vectorizer = CountVectorizer(ngram_range=(1,3), strip_accents='unicode')
    tweets_vectorized_train = vectorizer.fit_transform(raw_tweets_train)
    tweets_vectorized_test = vectorizer.transform(raw_tweets_test)
    
if tfidf_vect == 1:
    vectorizer = TfidfVectorizer()
    tweets_vectorized_train = vectorizer.fit_transform(raw_tweets_train)
    tweets_vectorized_test = vectorizer.transform(raw_tweets_test)
    
if wordEmb_vect == 1:
    tweets_vectorized_train = pd.DataFrame(word2vec_models_sum_train)
    tweets_vectorized_test = pd.DataFrame(word2vec_models_sum_test)
    
features_list_train = tweets_vectorized_train
features_list_test = tweets_vectorized_test

#Word Embeddings vector dimensionality reduction
if reduction_1 == 1:
    ch1 = SelectKBest(f_classif, k=50)
    word2vec_models_sum_train = ch1.fit_transform(word2vec_models_sum_train, correct_decision)
    word2vec_models_sum_test = ch1.transform(word2vec_models_sum_test)

#Document vector dimensionality reduction
if reduction_2 == 1:
    ch2 = SelectKBest(chi2, k=5000)
    #ch2 = SelectKBest(f_classif, k=6000)
    #ch2 = SelectFpr(chi2)
    features_list_train = ch2.fit_transform(tweets_vectorized_train, correct_decision)
    features_list_test = ch2.transform(tweets_vectorized_test)
    #Dimensionality reduction using PCA
    #reduce_pca = PCA()
    #features_list_train = reduce_pca.fit_transform(features_list_train.toarray())
    #features_list_test = reduce_pca.transform(features_list_test.toarray())

if wordEmb_vect == 0:
    features_list_train = features_list_train.toarray().tolist() 
    features_list_test = features_list_test.toarray().tolist() 

#Appending training features to the training document vector
features_list_train = features_append(features_list_train, elongated_f, exclamation_f, question_f, stop_f,
                    cap_f, posWord_f, negWord_f, word_f, slang_f, posEmo_f, negEmo_f,
                    url_f, hash_f, atUser_f, noun_f, adj_f, adverb_f, verb_f, 
                    vader_f, wordEmb_f, elongated_count_train, count_exclamation_train, 
                    count_question_train, count_stop_train, count_cap_train, 
                    count_pos_train, count_neg_train, count_words_train, count_slg_train,
                    count_neg_emo_train, count_pos_emo_train, count_url_train, 
                    count_hash_train, count_user_train, count_noun_train, count_adj_train,
                    count_adv_train, count_verb_train, vader_matrix_train, word2vec_models_sum_train)

#Appending testing features to the testing document vector
features_list_test = features_append(features_list_test, elongated_f, exclamation_f, question_f, stop_f,
                    cap_f, posWord_f, negWord_f, word_f, slang_f, posEmo_f, negEmo_f,
                    url_f, hash_f, atUser_f, noun_f, adj_f, adverb_f, verb_f, 
                    vader_f, wordEmb_f, elongated_count_test, count_exclamation_test, 
                    count_question_test, count_stop_test, count_cap_test, 
                    count_pos_test, count_neg_test, count_words_test, count_slg_test,
                    count_neg_emo_test, count_pos_emo_test, count_url_test, 
                    count_hash_test, count_user_test, count_noun_test, count_adj_test,
                    count_adv_test, count_verb_test, vader_matrix_test, word2vec_models_sum_test)

#Document vector normalization
if normalization == 1:
    tweets_list_unormal_train = features_list_train
    tweets_list_unormal_test = features_list_test
    scaler = MinMaxScaler().fit(features_list_train)
    features_list_train = scaler.transform(features_list_train).tolist()
    features_list_test = scaler.transform(features_list_test).tolist()
    features_list_train = pd.DataFrame(features_list_train) 
    features_list_test = pd.DataFrame(features_list_test) 
    tweets_list_normal_train = features_list_train
    tweets_list_normal_test = features_list_test
    
features_list_train = pd.DataFrame(features_list_train) 
features_list_test = pd.DataFrame(features_list_test)

#Single Classifiers
clfr = LogisticRegression(class_weight='balanced', tol=0.000001, random_state=1)
#clfr = LinearSVC(class_weight='balanced', tol=0.000001, random_state=1)
#clfr = naive_bayes.MultinomialNB()
#clfr = naive_bayes.GaussianNB()
#clfr = naive_bayes.BernoulliNB()
#clfr = DecisionTreeClassifier(random_state=0)
#clfr = XGBClassifier()
#clfr = RandomForestClassifier(n_estimators=100)
#clfr = ExtraTreesClassifier(n_estimators=100)
#clfr = svm.SVC(kernel='linear')
#clfr = GradientBoostingClassifier()

#Ensemble Classifier
#estimators = []
#model1 = LogisticRegression()
#estimators.append(('logistic', model1))
#model2 = DecisionTreeClassifier()
#estimators.append(('cart', model2))
#model3 = SVC()
#estimators.append(('svm', model3))
#clfr = VotingClassifier(estimators)

#Random Search Classifier
#clfr1 = RandomForestClassifier()
#param_dist = {"max_depth": [3, None],
#              "max_features": sp_randint(1, 11),
#              "min_samples_split": sp_randint(2, 11),
#              "min_samples_leaf": sp_randint(1, 11),
#              "bootstrap": [True, False],
#              "criterion": ["gini", "entropy"]}
#n_iter_search = 20
#clfr = RandomizedSearchCV(clfr1, param_distributions=param_dist)

#Cross validation
kf = StratifiedKFold(n_splits=10, shuffle=True)
arr_acc = []
arr_f1score = []
for train_idx, val_idx in kf.split(features_list_train, correct_decision):
    X_train = features_list_train.iloc[train_idx]
    train_label = correct_decision.iloc[train_idx]
    X_val = features_list_train.iloc[val_idx]
    val_label = correct_decision.iloc[val_idx]
    clfr.fit(X_train,train_label)
    predicted = clfr.predict(X_val)
    acc = metrics.accuracy_score(val_label,predicted)  
    f_score = metrics.f1_score(val_label,predicted,average='weighted') 
    arr_acc.append(acc)
    arr_f1score.append(f_score)
    print('accuracy = '+str(acc*100)+'%')
    print('f1_score = '+str(f_score*100)+'%')
print('avg accuracy = '+str(sum(arr_acc)/10*100)+'%')
print('avg f1_score = '+str(sum(arr_f1score)/10*100)+'%')

#Fitting the classifier with the traininf data
clfr.fit(features_list_train, correct_decision)

#Predicting the testing data
predicted_labels = clfr.predict(features_list_test)

#Saving the predicted labels in a csv file
with open('test_labels.csv', 'w', newline='') as csvfile:
    labelwriter = csv.writer(csvfile, delimiter=',')
    labelwriter.writerow(['id'] + ['sentiment'])
    for i in range(0, len(tweets_id_test)):    
        labelwriter.writerow([tweets_id_test[i]] + [predicted_labels[i]])
    
#End of computation time
stop = timeit.default_timer()

print ('Computation Time =', stop - start) 