import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.lancaster import LancasterStemmer
from bs4 import  BeautifulSoup


def clean(texts):
    trans = str.maketrans(dict.fromkeys(string.punctuation))
    st = LancasterStemmer()
    sw = set(stopwords.words('english'))

    for text in texts:
        soup = BeautifulSoup(text, 'lxml')
        text = ''.join(soup.findAll(text=True)).strip('/n')
        text = text.translate(trans)
        tokens = nltk.word_tokenize(text.lower())
        tokens = [st.stem(j) for j in tokens if (j not in sw)]

        yield ' '.join(tokens)
