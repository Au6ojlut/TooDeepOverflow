import os.path
import pandas as pd
import pickle
import tqdm
from bs4 import BeautifulSoup


DATA_ROOT = '/home/egor/ml/DeepOverflow/data'

answers = {}


posts_df = pd.read_csv(os.path.join(DATA_ROOT, 'incoming', 'Posts.csv'), encoding='ISO-8859-1', low_memory=False, dtype='str')

total = len(posts_df.index)

entities_dict = {}

for idx, row in tqdm.tqdm(posts_df.iterrows(), total=total):
    if int(row['PostTypeId']) == 2:
        id = int(row['Id'])
        text = row['Body']

        soup = BeautifulSoup(row['Body'], 'lxml')

        counter = 0
        for tag in soup.find_all('code'):
            value = tag.string
            if value != '' and value is not None:
                code = '@code_' + str(row['Id']) + '_' + str(counter)
                entities_dict[code] = value
                tag.string = code
                counter += 1

        counter = 0
        for tag in soup.find_all('a', href=True):
            value = tag.href
            if value != '' and value is not None:
                code = '@url_' + str(counter)
                entities_dict[code] = value
                tag.string = code
                counter += 1

        answers[id] = ''.join(soup.findAll(text=True))


with open(os.path.join(DATA_ROOT, 'computed', 'entities.pickle'), 'wb') as handle:
    pickle.dump(entities_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(DATA_ROOT, 'computed', 'answers.pickle'), 'wb') as f:
    pickle.dump(answers, f)

