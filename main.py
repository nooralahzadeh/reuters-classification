import os
from utils import parse_documents, Document, transformer
import pickle
import matplotlib.pyplot as plt
import operator
from collections import OrderedDict
from itertools import chain
from classifer import mdl

# ignore warning in sklearn
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='classifier')
    parser.add_argument('--path', type=str, help='Input dir')
    parser.add_argument('--n', type=int, help='n most topic')
    args = parser.parse_args()

    datapath = args.path
    # get_topics(datapath,"all-topics-strings.lc.txt")
    corpus_name = os.path.join(datapath,"dataset.pkl")
    if os.path.exists(corpus_name):
        with open(os.path.normpath(corpus_name), "rb") as pkl:
            corpus = pickle.load(pkl)
    else:
        corpus = parse_documents(datapath)
        with open(corpus_name, "wb") as _file:
            pickle.dump(corpus, _file)

    print("Finished Extraction step")

    # some statistics
    topics_in_train=OrderedDict(sorted(corpus['topics']['train'].items(), key=operator.itemgetter(1),reverse=True))

    plt.bar(range(len(topics_in_train)), list(topics_in_train.values()), align='center')
    plt.xticks(range(len(topics_in_train)), list(topics_in_train.keys()))
    plt.xlabel('xlabel', fontsize=8)
    plt.show()
    # highly skewed distribution of documents over topics
    # all topics :
    print("Running feature extraction step")
    topics=list(set(chain(corpus['topics']['train'].keys(),corpus['topics']['test'].keys())))
    train_dataset, test_dataset=transformer(corpus['dataset'],labels= topics)
    print("Running classifier")
    # classifer
    metrics=mdl(train_dataset,test_dataset)

    print("Micro-average")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
          .format(metrics['micro']['P'], metrics['micro']['R'], metrics['micro']['f1']))
    ####################################
    print("Macro-average")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
          .format(metrics['macro']['P'], metrics['macro']['R'], metrics['macro']['f1']))

    print("Running classifier by filtering {} most topic".format(args.n))
    topics_in_test=list(corpus['topics']['test'].keys())
    # filter dataset based on ten most topics
    n=args.n
    n_top_most_topic=[]
    for t in topics_in_train.keys():
        if n==0:
            break
        if t in topics_in_test:
            n_top_most_topic.append(t)
            n-=1

    train_dataset, test_dataset = transformer(corpus['dataset'], labels=n_top_most_topic)

    # classifer

    metrics = mdl(train_dataset, test_dataset)
    print("Micro-average after filtering")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
          .format(metrics['micro']['P'], metrics['micro']['R'], metrics['micro']['f1']))
    ####################################
    print("Macro-average after filtering")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
          .format(metrics['macro']['P'], metrics['macro']['R'], metrics['macro']['f1']))


# We have 120 topics and some of them has very small training and testing instances.
# it seems considering N-most topics, yeilds a better performance in Micro mode.
# I am wondering, if we use doc2vec method and use CNN or LSTM model as classifier,
# how would be the result?