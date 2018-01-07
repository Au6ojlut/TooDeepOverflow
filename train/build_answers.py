import os.path
import pandas as pd
import pickle
import tqdm
from bs4 import BeautifulSoup
from deepoverflow.config import DATA_ROOT

answers = {}

posts_path = os.path.join(DATA_ROOT, 'incoming', 'Posts.csv')
posts_df = pd.read_csv(posts_path, encoding='ISO-8859-1', low_memory=False, dtype='str')

posts_df.loc[:, 'ViewCount'] = posts_df['ViewCount'].fillna(100)
posts_df.loc[:, 'Score'] = posts_df['Score'].fillna(0)

total = len(posts_df.index)

entities_dict = {}

for idx, row in tqdm.tqdm(posts_df.iterrows(), total=total):
    if int(row['PostTypeId']) == 2:
        id = int(row['Id'])
        text = row['Body']
        views = int(row['ViewCount'])
        score = int(row['Score'])

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
                code = '@url_' + str(row['Id']) + '_' + str(counter)
                entities_dict[code] = value
                tag.string = code
                counter += 1

        answers[id] = (''.join(soup.findAll(text=True)), views, score)


with open(os.path.join(DATA_ROOT, 'computed', 'entities.pickle'), 'wb') as handle:
    pickle.dump(entities_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(DATA_ROOT, 'computed', 'answers.pickle'), 'wb') as f:
    pickle.dump(answers, f)

