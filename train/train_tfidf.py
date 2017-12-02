import os.path
from annoy import AnnoyIndex
import pickle
import tqdm
from sklearn.decomposition import PCA

from deepoverflow.similar import build_tfidf_model, TFIDF_MAX_FEATURES

DATA_ROOT = '/home/egor/ml/DeepOverflow/data'
PCA_COMPONENTS = 1024

with open(os.path.join(DATA_ROOT, 'computed', 'cleaned_questions.pickle'), 'rb') as f:
    xs = pickle.load(f)

ids, cleaned_questions = zip(*xs)

total = len(cleaned_questions)

print('loaded cleaned questions')

tfidf_path = os.path.join(DATA_ROOT, 'computed', 'tfidf.pickle')

if os.path.exists(tfidf_path):
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
        print('loaded tfidf')
else:
    tfidf = build_tfidf_model(cleaned_questions)

    print('built tfidf')

    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf, f)
        print('saved tfidf')

transformed_questions = tfidf.transform(cleaned_questions)

print('transformed questions')

pca = PCA(n_components=PCA_COMPONENTS)
transformed_questions = pca.fit_transform(transformed_questions)
print('performed PCA')
print(pca.explained_variance_ratio_.sum())


t = AnnoyIndex(PCA_COMPONENTS)
for idx, q in tqdm.tqdm(enumerate(transformed_questions), total=total):
    t.add_item(idx, q.reshape(-1))

t.build(10)

t.save(os.path.join(DATA_ROOT, 'computed', 'index.ann'))

print('saved Annoy Index')

