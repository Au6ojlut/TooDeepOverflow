import os.path
import pickle
import nmslib
import gensim
import re
import math
from .cleaner import clean


GENSIM_WORDS_COUNT = 384
MAX_ANSWERS = 100
NUMBER_OF_NEIGHBOURS = 500
USE_PCA = True
DEBUG = False


def score(score, views, dist, votes_range=None):
    return dist


class Application:
    def __init__(self, data_root):
        self.data_root = data_root

        with open(os.path.join(self.data_root, 'computed', 'tfidf.pickle'), 'rb') as f:
            self.tfidf = pickle.load(f)

        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.loadIndex(os.path.join(self.data_root, 'computed', 'index.nmslib'))

        if USE_PCA:
            with open(os.path.join(self.data_root, 'computed', 'pca.pickle'), 'rb') as f:
                self.pca = pickle.load(f)

        with open(os.path.join(self.data_root, 'computed', 'answers_tree.pickle'), 'rb') as f:
            self.answers_tree = pickle.load(f)

        with open(os.path.join(self.data_root, 'computed', 'answers.pickle'), 'rb') as f:
            self.answers = pickle.load(f)

        if not DEBUG:
            with open(os.path.join(self.data_root, 'computed', 'entities.pickle'), 'rb') as f:
                self.entities = pickle.load(f)

    def question2vec(self, question):
        cleaned_question = list(clean([question]))[0]
        vec = self.tfidf.transform([cleaned_question]).todense()
        if USE_PCA:
            vec = self.pca.transform(vec).reshape(-1)

        return vec

    def summarize_v1(self, answers):
        return gensim.summarization.summarize(' '.join(answers), word_count=GENSIM_WORDS_COUNT)

    def replace_entities(self, text):
        def replacer(match):
            name = match['name']

            category, *_ = name[1:].split('_')

            if name not in self.entities:
                return 'NOT_FOUND_ENTITY'

            text = self.entities[name]

            if category == 'code':
                if '\n' in text:
                    text = '<br><code>{}</code></br>'.format(text)
                else:
                    text = '<code>{}</code>'.format(text)
            elif category == 'url':
                text = '<a href="{0}" target="blank">{0}</a>'.format(text)
                print(name)
                print(text)

            return text

        return re.sub(r'(?P<name>@\w+)', replacer, text)

    def debug_process(self, question):
        vec = self.question2vec(question)
        knn_ids, dists = self.index.knnQuery(vec, k=NUMBER_OF_NEIGHBOURS)

        text = 'SIMILIAR QUESTIONS IDS: {}<br/>'.format(', '.join(map(str, knn_ids)))

        answer_ids = []
        for id, dist in zip(knn_ids, dists):
            if id in self.answers_tree:
                for ans_id in self.answers_tree[id]:
                    r = score(
                        self.answers[ans_id][2],
                        self.answers[ans_id][1],
                        dist
                    )
                    answer_ids.append((r, ans_id))

        text += '<br/>ANSWERS ({}):<br/>'.format(len(answer_ids))

        answer_ids.sort(key=lambda p: p[0], reverse=True)
        answer_ids = answer_ids[:MAX_ANSWERS]

        for r, ans_id in answer_ids:
            text += 'answer: {} rank: {}<br/>'.format(ans_id, r)
            text += self.answers[ans_id][0]
            text += '<br/><br/>'

        return text

    def process(self, question):
        if DEBUG:
            return self.debug_process(question)

        vec = self.question2vec(question)
        knn_ids, dists = self.index.knnQuery(vec, k=NUMBER_OF_NEIGHBOURS)

        answer_ids = []
        for id, dist in zip(knn_ids, dists):
            if id in self.answers_tree:
                for ans_id in self.answers_tree[id]:
                    r = score(
                        self.answers[ans_id][2],
                        self.answers[ans_id][1],
                        dist
                    )
                    answer_ids.append((r, ans_id))

        print('answers:', len(answer_ids))

        answer_ids.sort(key=lambda p: p[0], reverse=True)
        answer_ids = answer_ids[:MAX_ANSWERS]

        answer_bodies = []

        for _, ans_id in answer_ids:
            answer_bodies.append(self.answers[ans_id][0])

        summarization = self.summarize_v1(answer_bodies)
        summarization = self.replace_entities(summarization)

        return summarization