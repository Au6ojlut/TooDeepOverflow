import os.path
import pandas as pd
import pickle
from tqdm import tqdm
from deepoverflow.cleaner import clean

DATA_ROOT = '/home/egor/ml/DeepOverflow/data'

posts_path = os.path.join(DATA_ROOT, 'incoming', 'Posts.csv')

posts_df = pd.read_csv(posts_path, encoding='ISO-8859-1', low_memory=False)
questions = posts_df['Body'].tolist()

total = len(questions)

print('loaded questions')

cleaned_questions = list(tqdm(clean(questions), total=total))

print('cleaned questions')

with open(os.path.join(DATA_ROOT, 'computed', 'cleaned_questions.pickle'), 'wb') as f:
    pickle.dump(cleaned_questions, f)

print('saved cleaned questions')