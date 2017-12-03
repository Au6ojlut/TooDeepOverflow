import pickle
import pandas as pd
import os
from bs4 import BeautifulSoup


def get_entities(posts_df):
    entities_dict = {}
    processed_answers = {}
    counter = 0

    for index, row in posts_df.iterrows():
        if int(row['PostTypeId']) == 1:
            continue
        soup = BeautifulSoup(row['Body'], 'lxml')

        for tag in soup.find_all('code'):
            code = '@code_' + str(row['Id']) + '_' + str(counter)
            entities_dict[code] = tag.string
            tag.string = code
            counter += 1
        for tag in soup.find_all('a', href=True):
            code = '@url_' + str(row['Id']) + '_' + str(counter)
            entities_dict[code] = tag.string
            tag.string = code
            counter += 1

        # text = ''.join(soup.findAll(text=True)).strip('/n')
        text = ''.join(soup.findAll(text=True))
        processed_answers[str(row['Id'])] = text

    return entities_dict, processed_answers


def save_entities(ent_dict, proc_answers):
    with open('entities.pickle', 'wb') as handle:
        pickle.dump(ent_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('processed_answers.pickle', 'wb') as handle:
        pickle.dump(proc_answers, handle, protocol=pickle.HIGHEST_PROTOCOL)


posts_file = "Posts.csv"
path = "D:/data"
posts_info = pd.read_csv(os.path.join(path, posts_file), encoding="ISO-8859-1")

posts = posts_info[['Id', 'Body', 'PostTypeId']]
gen_dict, proc_answers = get_entities(posts)
save_entities(gen_dict, proc_answers)