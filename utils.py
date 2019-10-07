import os
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from collections import defaultdict
from bs4 import BeautifulSoup

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import  TfidfVectorizer

class Document:

    """
    Each .sgm file has following tags related for classification task:
    <REUTERS  LEWISSPLIT="TRAIN"  NEWID="01">
    <TOPICS><D>...</D><D>...</D>/TOPICS>
    <TEXT>....
    <TITLE>...</TITLE>
    <BODY>...</BODY>
    </TEXT>
    """
    def __init__(self,article):
        #print (article.prettify())
        self.topics= [topic.text for topic in article.topics.children]
        #self.body = [body.get_text() for body in article.find_all('body')]
        self.newid= article['newid']
        self.lewissplit=article['lewissplit']
        self.title=[]
        self.body=[]

    def _tokenize_title_body(self,article):
        # tokenize, cleanig , lemmatize and stemming
        if article.title:
            self.title = self.__tokenize(article.title.string)
        if article.body:
            self.body = self.__tokenize(article.body.string)
        if len(self.title) == 0  and len(self.body) == 0:
            self.body= self.__tokenize(article.text)

    def __tokenize(self,text):
        # remove punctuation # remove digits
        text=text.translate(str.maketrans('', '', string.punctuation+string.digits))
        # separate text into tokens
        tokens = nltk.word_tokenize(text)
        # remove stopwords,
        tokens = [w for w in tokens if not w in stopwords.words('english')]

        # lemmatization
        lemmas = []
        lmtzr = WordNetLemmatizer()
        for token in tokens:
           lemmas.append(lmtzr.lemmatize(token))

        # stemming
        stems = []
        stemmer = PorterStemmer()
        for token in lemmas:
            stem = stemmer.stem(token).encode('ascii', 'ignore')
            if len(stem) >= 4:
                stems.append(stem)
        return stems


def parse_documents(datapath):
    dataset = defaultdict(list)
    topics= defaultdict(dict)
    for file in os.listdir(datapath):
        # open '.sgm' file
        if not file.endswith(".sgm"):
            continue
        path = os.path.join(datapath, file)
        with open(path, 'rb') as data:
            text = data.read()
        tree = BeautifulSoup(text.lower(), "html.parser")
        for reuter in tree.find_all("reuters"):
            document = Document(reuter)
            document._tokenize_title_body(reuter)
            if len(document.topics)>0:
                dataset[document.lewissplit].append(document)
                for topic in document.topics:
                    if topic in topics[document.lewissplit]:
                        topics[document.lewissplit][topic]+=1
                    else:
                        topics[document.lewissplit][topic]= 1
        print("Finished extracting information from file {}".format(file))
    return {'dataset': dataset, 'topics': topics}



def dummy_fun(doc):
    return doc

def transformer(dataset,labels):
    vectorizer = TfidfVectorizer(analyzer='word',
                                tokenizer=dummy_fun,
                                preprocessor=dummy_fun,
                                token_pattern=None, min_df=1)

    # binarize the class ( multi-class, multi-label)
    mlb = MultiLabelBinarizer(classes=list(labels))
    docs = {}
    # apply filtering
    docs['train'] = [doc.body + doc.title for doc in dataset['train'] if len(set(doc.topics)- set(labels))==0]
    docs['test'] = [doc.body + doc.title for doc in dataset['test'] if len(set(doc.topics) - set(labels))==0]
    xs = {'train': [], 'test': []}
    # vectorize
    xs['train'] = vectorizer.fit_transform(docs['train']).toarray()
    xs['test'] = vectorizer.transform(docs['test']).toarray()
    ys = {'train': [], 'test': []}


    ys['train'] = mlb.fit_transform([doc.topics
                                     for doc in dataset['train'] if len(set(doc.topics) - set(labels))==0])
    l=list(mlb.classes_)
    #print (l)
    ys['test'] = mlb.fit_transform([doc.topics
                                     for doc in dataset['test'] if len(set(doc.topics)- set(labels))==0])

    train_dataset=(xs['train'], ys['train'])
    test_dataset=(xs['test'], ys['test'])
    return train_dataset, test_dataset
