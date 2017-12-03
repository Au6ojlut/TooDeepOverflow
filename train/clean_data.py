import os.path
import pandas as pd
import pickle
from tqdm import tqdm
from deepoverflow.cleaner import clean

from deepoverflow.config import DATA_ROOT

posts_path = os.path.join(DATA_ROOT, 'incoming', 'Posts.csv')

posts_df = pd.read_csv(posts_path, encoding='ISO-8859-1', low_memory=False)
questions_mask = posts_df['PostTypeId'] == 1
ids = posts_df[questions_mask]['Id'].tolist()
questions = posts_df[questions_mask]['Body'].tolist()

total = len(questions)

print('loaded questions')

cleaned_questions = list(tqdm(clean(questions), total=total))

print('cleaned questions')

with open(os.path.join(DATA_ROOT, 'computed', 'cleaned_questions.pickle'), 'wb') as f:
    pickle.dump(list(zip(ids, cleaned_questions)), f)

print('saved cleaned questions')