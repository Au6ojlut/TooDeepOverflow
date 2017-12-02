import os.path
import pickle
from annoy import AnnoyIndex
from .cleaner import clean
from .similar import TFIDF_MAX_FEATURES


class Application:
    def __init__(self, data_root):
        self.data_root = data_root

        with open(os.path.join(self.data_root, 'computed', 'tfidf.pickle'), 'rb') as f:
            self.tfidf = pickle.load(f)

        self.index = AnnoyIndex(TFIDF_MAX_FEATURES)
        self.index.load(os.path.join(self.data_root, 'computed', 'index.ann'))

    def question2vec(self, question):
        cleaned_question = clean(question)
        return self.tfidf.transform([cleaned_question])[0]

    def find_similar(self, top=100):
