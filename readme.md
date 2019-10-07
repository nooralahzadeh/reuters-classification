
- Requirements: Python 3.6, matplotlib, sklearn

`python model.py --path <path to the dataset> --n <classify with n most topic>
`

Example: 

1- Put the reuters dataset in the root 

2- Run classifier with 10 most topics

`python model.py --path  reuters21578/  --n 10
`
- First results:

`Micro-average: Precision: 0.9377, Recall: 0.7867, F1-measure: 0.8556`

`Macro-average: Precision: 0.4759, Recall: 0.2913, F1-measure: 0.3442`


- Results with 10 most topics:

`Micro-average: Precision: 0.9654, Recall: 0.9215, F1-measure: 0.9429`

`Micro-average : Precision: 0.9364, Recall: 0.8420, F1-measure: 0.8852`

We have 120 topics and some of them have a very small training and testing instances.
it seems considering N-most topics, yeilds a better performance in the Micro mode.
I am wondering, if we use doc2vec method and use CNN or LSTM model as a classifier, what would be the results?