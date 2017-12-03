import os.path
import nmslib
import pickle
from sklearn.decomposition import TruncatedSVD
import scipy

from deepoverflow.similar import build_tfidf_model

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

tags_path = os.path.join(DATA_ROOT, 'incoming', 'Tags.csv')
tags_df = pd.read_csv(tags_file,  encoding = "ISO-8859-1", index_col=2)
top_tags = tags_df.sort_values(['Count'], ascending=False).TagName[:256]

top_tags_path = os.path.join(DATA_ROOT, 'computed', 'top_tags.pickle')
with open(top_tags_path, 'wb') as f:
	pickle.dump(top_tags, f)
	print('saved top_tags')

additional_matrix = np.zeros((total, len(top_tags)), dtype='int64')
temp_tags = posts_df[questions_mask]['Tags'].apply(lambda x: x.replace('<', '').replace('>', ' '))
for i in range(len(top_tags)):
    additional_matrix[:,i] = temp_tags.apply(lambda x: top_tags.iloc[i] in x).values
additional_matrix = scipy.sparse.csr_matrix(additional_matrix, shape=additional_matrix.shape)
transformed_questions = scipy.sparse.hstack(transformed_questions, scipy.sparse.csr_matrix(additional_matrix, shape=additional_matrix.shape)

print('added tags')

pca_path = os.path.join(DATA_ROOT, 'computed', 'pca.pickle')

if os.path.exists(pca_path):
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)

    print('loaded PCA')

    transformed_questions = pca.transform(transformed_questions)
    print('performed PCA')
else:
    pca = TruncatedSVD(n_components=PCA_COMPONENTS, n_iter=10, random_state=42)
    transformed_questions = pca.fit_transform(transformed_questions)
    print('performed PCA')
    print(pca.explained_variance_ratio_.sum())

    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)

    print('saved PCA')

index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data=transformed_questions, ids=ids)
index.createIndex({'post': 2}, print_progress=True)
index.saveIndex(os.path.join(DATA_ROOT, 'computed', 'index.nmslib'))

print('saved nmslib index')
