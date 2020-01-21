#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:19:45 2018

@author: kevin
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

files = open('allfilesinone', 'r')

# convert the text to a tf-idf weighted term-document matrix

vectorizer = TfidfVectorizer(max_features=4000, min_df=10, stop_words='english')

X = vectorizer.fit_transform(files)

idx_to_word = np.array(vectorizer.get_feature_names())

# apply NMF

nmf = NMF(n_components=20, solver="mu")

W = nmf.fit_transform(X)

H = nmf.components_

# print the topics

for i, topic in enumerate(H):
    print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in idx_to_word[topic.argsort()[-10:]]])))