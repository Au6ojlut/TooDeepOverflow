import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


TFIDF_MAX_FEATURES = 10000


def question2vec(text):
    pass


def build_tfidf_model(text):
    vectorizer = TfidfVectorizer(input='content', max_features=TFIDF_MAX_FEATURES, use_idf=True, smooth_idf=True)
    vectorizer.fit(raw_documents=text)
    return vectorizer