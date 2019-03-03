#!/usr/bin/python3
# Dylan Waters CS3580 Assignment 7

from __future__ import division
import numpy as np
import json
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import sys

# author books
booksAusten = []
booksBaum = []
booksVerne = []
bookToPredict = []

# Books by Austen
wordsStr = ""
loadBook = open('./Austen/Persuasion.txt', 'r')
for line in loadBook:
    wordsStr += line
booksAusten.append(wordsStr)

wordsStr = ""
loadBook = open('./Austen/Mansfield_Park.txt', 'r')
for line in loadBook:
    wordsStr += line
booksAusten.append(wordsStr)

wordsStr = ""
loadBook = open('./Austen/Northanger_Abbey.txt', 'r')
for line in loadBook:
    wordsStr += line
booksAusten.append(wordsStr)

wordsStr = ""
loadBook = open('./Austen/Pride_and_Prejudice.txt', 'r')
for line in loadBook:
    wordsStr += line
booksAusten.append(wordsStr)

# books by Baum

loadBook = open('./Baum/Dorothy_and_the_Wizard_in_Oz.txt', 'r')
for line in loadBook:
    wordsStr += line
booksBaum.append(wordsStr)

wordsStr = ""
loadBook = open('./Baum/Ozma_of_Oz.txt', 'r')
for line in loadBook:
    wordsStr += line
booksBaum.append(wordsStr)

wordsStr = ""
loadBook = open('./Baum/The_Emerald_City_of_Oz.txt', 'r')
for line in loadBook:
    wordsStr += line
booksBaum.append(wordsStr)

wordsStr = ""
loadBook = open('./Baum/The_Wonderful_Wizard_of_Oz.txt', 'r')
for line in loadBook:
    wordsStr += line
booksBaum.append(wordsStr)

# books by Verne

wordsStr = ""
loadBook = open('./Verne/All_Around_the_Moon.txt', 'r')
for line in loadBook:
    wordsStr += line
booksVerne.append(wordsStr)

wordsStr = ""
loadBook = open('./Verne/Around_the_World.txt', 'r')
for line in loadBook:
    wordsStr += line
booksVerne.append(wordsStr)

wordsStr = ""
loadBook = open('./Verne/From_the_Earth_to_the_Moon.txt', 'r')
for line in loadBook:
    wordsStr += line
booksVerne.append(wordsStr)

wordsStr = ""
loadBook = open('./Verne/Journey_to_the_Centre_of_the_Earth.txt', 'r')
for line in loadBook:
    wordsStr += line
booksVerne.append(wordsStr)

# numWorksAusten = len(booksAusten)
# numWorksBaum = len(booksBaum)
# numWorksVerne = len(booksVerne)

# get bags of words

authorArray = ["Austen","Austen","Austen","Austen","Baum","Baum","Baum","Baum","Verne","Verne","Verne","Verne"]

stop_words = text.ENGLISH_STOP_WORDS.union({'VERNE','BAUM','AUSTEN','PROJECT','GUTENBERG'})

vectorizer = text.CountVectorizer(stop_words=stop_words,min_df=10)
X_train = vectorizer.fit_transform(booksAusten+booksBaum+booksVerne)
tfidfTransform = TfidfTransformer()
X_train_tfidf = tfidfTransform.fit_transform(X_train)
naiveBayesClassifier = MultinomialNB().fit(X_train_tfidf, authorArray)

userSelectedPath = ""
while(userSelectedPath != '0'):
    # user predicted book

    print("Enter the name of a book to read in (type 0 to quit): ")
    userSelectedPath = input()
    if(userSelectedPath == '0'):
        sys.exit()
    loadBook = open(userSelectedPath, 'r')
    wordsStr = ""
    for line in loadBook:
        wordsStr += line
    bookToPredict.append(wordsStr)

    # numWorksUserBook = len(bookToPredict)

    X_test = vectorizer.transform(bookToPredict)
    X_test_tfidf = tfidfTransform.transform(X_test)

    predicted = naiveBayesClassifier.predict(X_test_tfidf)

    print(predicted)
