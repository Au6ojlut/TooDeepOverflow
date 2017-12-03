import os.path
import pickle
import nmslib
import gensim
import re
import math
from .cleaner import clean


GENSIM_WORDS_COUNT = 256
MAX_ANSWERS = 100


def wilson_score(score, views, votes_range=None):
    if votes_range is None:
        votes_range = [-1, 1]

    views = views * 0.1

    z = 1.64485
    v_min = min(votes_range)
    v_width = float(max(votes_range) - v_min)
    phat = (score - views * v_min) / v_width / float(views)
    rating = (phat + z * z / (2 * views) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * views)) / views)) / (1 + z * z / views)
    return rating * v_width + v_min


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

        with open(os.path.join(self.data_root, 'computed', 'answer_properties.pickle'), 'rb') as f:
            self.answer_properties = pickle.load(f)

        with open(os.path.join(self.data_root, 'computed', 'entities.pickle'), 'rb') as f:
            self.entities = pickle.load(f)

    def question2vec(self, question):
        cleaned_question = list(clean([question]))[0]
        tfidf = self.tfidf.transform([cleaned_question]).todense()
        tfidf_compressed = self.pca.transform(tfidf).reshape(-1)

        return tfidf_compressed

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

    def process(self, question):
        vec = self.question2vec(question)
        knn_ids, dists = self.index.knnQuery(vec, k=100)

        answer_ids = []
        for id, dist in zip(knn_ids, dists):
            if id in self.answers_tree:
                for ans_id in self.answers_tree[id]:
                    r = dist * wilson_score(
                        self.answer_properties[ans_id]['score'],
                        self.answer_properties[ans_id]['views']
                    )
                    answer_ids.append((r, ans_id))

        answer_ids.sort(key=lambda p: p[0], reverse=True)
        answer_ids = answer_ids[:MAX_ANSWERS]

        answer_bodies = []

        for _, ans_id in answer_ids:
            answer_bodies.append(self.answers[ans_id])

        summarization = self.summarize_v1(answer_bodies)
        summarization = self.replace_entities(summarization)

        return summarization