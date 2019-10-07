from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

def mdl(train,test):
    # One-vs-the-rest (OvR) multiclass/multilabel strategy
    # train
    classifier = OneVsRestClassifier(LinearSVC(random_state=1000), n_jobs=-1)
    classifier.fit(train[0], train[1])
    y_pred = classifier.predict(test[0])
    y_test = test[1]

    # Evaluation
    ####################################
    metrics={'macro':{'f1':0,'R':0, 'P':0},'micro':{'f1':0,'R':0, 'P':0}}
    metrics['micro']['P'] = precision_score(y_test, y_pred,
                                average='micro')
    metrics['micro']['R'] = recall_score(y_test, y_pred,
                          average='micro')
    metrics['micro']['f1'] = f1_score(y_test, y_pred, average='micro')


    ####################################
    metrics['macro']['P'] = precision_score(y_test, y_pred,
                                average='macro')
    metrics['macro']['R'] = recall_score(y_test, y_pred,
                          average='macro')
    metrics['macro']['f1'] = f1_score(y_test, y_pred, average='macro')

    return metrics