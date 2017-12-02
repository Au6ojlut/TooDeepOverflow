import os.path
import pickle
import nmslib
from .cleaner import clean


class Application:
    def __init__(self, data_root):
        self.data_root = data_root

        with open(os.path.join(self.data_root, 'computed', 'tfidf.pickle'), 'rb') as f:
            self.tfidf = pickle.load(f)

        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.loadIndex(os.path.join(self.data_root, 'computed', 'index.nmslib'))

        with open(os.path.join(self.data_root, 'computed', 'pca.pickle'), 'rb') as f:
            self.pca = pickle.load(f)

        with open(os.path.join(self.data_root, 'computed', 'answers_tree.pickle'), 'rb') as f:
            self.answers_tree = pickle.load(f)

        with open(os.path.join(self.data_root, 'computed', 'answers.pickle'), 'rb') as f:
            self.answers = pickle.load(f)

    def question2vec(self, question):
        cleaned_question = list(clean([question]))[0]
        tfidf = self.tfidf.transform([cleaned_question]).todense()
        tfidf_compressed = self.pca.transform(tfidf).reshape(-1)

        return tfidf_compressed

    def process(self, question):
        vec = self.question2vec(question)
        knn_ids, _ = self.index.knnQuery(vec, k=100)

        answers = []
        for id in knn_ids:
            if id in self.answers_tree:
                answers.extend(self.answers_tree[id])

        answers_body = []

        for ans_id in answers:
            answers_body.append(self.answers[ans_id])

        return '\n'.join(answers_body)