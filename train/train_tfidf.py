import os.path
import pandas as pd
import pickle
import itertools

from deepoverflow.cleaner import clean
from deepoverflow.similiar import build_tfidf_model

DATA_ROOT = '/home/egor/ml/DeepOverflow/data'

posts_path = os.path.join(DATA_ROOT, 'incoming', 'Posts.csv')

posts_df = pd.read_csv(posts_path, encoding='ISO-8859-1', low_memory=False)
questions = posts_df['Body'].tolist()

print('loaded questions')

cleaned_questions = list(map(clean, questions))

print('cleaned questions')

tfidf = build_tfidf_model(cleaned_questions[:1000])

print('built tfidf')

pickle.dump(tfidf, os.path.join(DATA_ROOT, 'computed', 'tfidf.pickle'))

print('saved tfidf')

transformed_questions = map(tfidf.transform, cleaned_questions)

print('transformed questions')

pickle.dump(zip(itertools.count(), transformed_questions, questions), os.path.join(DATA_ROOT, 'computed', 'questions.pickle'))

print('saved transformed questions')

