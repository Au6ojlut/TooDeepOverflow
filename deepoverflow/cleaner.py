import re
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from bs4 import  BeautifulSoup
from multiprocessing import Pool


def clean(texts):
    st = LancasterStemmer()
    sw = set(stopwords.words('english'))
    token_re = re.compile(r'[a-zA-Z]{3,}')
    url_re = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')

    for text in texts:
        soup = BeautifulSoup(text, 'lxml')

        for tag in soup.find_all('code'):
            tag.clear()

        text = ''.join(soup.findAll(text=True)).strip('/n')
        text = url_re.sub('', text)
        tokens = token_re.findall(text.lower())
        tokens = [st.stem(j) for j in tokens if (j not in sw)]

        yield ' '.join(tokens)