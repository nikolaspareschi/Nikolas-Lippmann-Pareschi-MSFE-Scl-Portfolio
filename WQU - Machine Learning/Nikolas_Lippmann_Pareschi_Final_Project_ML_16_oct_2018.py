# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:01:00 2018

@author: nikol
"""

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import random
import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

moviedirectory = r'C:\Users\nikol\txt_sentoken'


movie_train = load_files(moviedirectory,shuffle=True)


# The classes are generated from subfolder names

# Please copy the files from positive and negative subfolders to the main directory
# which will be read

movie_train.target_names


foovec = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)


# Make sure we have all the ntlk package

nltk.download()

# initialize movie_vector 
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)        
movie_counts = movie_vec.fit_transform(movie_train.data)

# Let[s check if we can search for words

movie_vec.vocabulary_.get('deniro')

# Convert raw frequency counts into TF-IDF values
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)

# Training and test sets

docs_train, docs_test, y_train, y_test = train_test_split(
    movie_tfidf, movie_train.target, test_size = 0.20, random_state = 12)

# Trainining -Naive Bayes classifier

clf = MultinomialNB().fit(docs_train, y_train)

# Predictions

y_pred = clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm



stop_words = stopwords.words("english")
import pickle

movie_reviews.categories()

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)
            ]
random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
#making a frequency distribution of the words
all_words = nltk.FreqDist(all_words)
all_words.most_common(25)


#train first 5000 words on the list
feature_words = list(all_words.keys())[:5000]

def find_features(document):
    words = set(document)
    feature = {}
    for w in feature_words:
        feature[w] = (w in words)
    return feature

feature_sets = [(find_features(rev), category) for (rev, category) in documents]

#Training 

training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]

## classifier NB
classifier = nltk.NaiveBayesClassifier.train(training_set)

## Accuracy
print("Naive bayes classifier accuracy percentage : ", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(25)
