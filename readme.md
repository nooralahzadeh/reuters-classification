
- Requirements: Python 3.6, matplotlib, sklearn

`python model.py --path <path to the dataset> --n <classify with n most topic>
`

Example: 

1- Put the reuters dataset in the root 

2- Run classifier with 10 most topics

`python model.py --path  reuters21578/  --n 10
`

We have 120 topics and some of them have a very small training and testing instances.
it seems considering N-most topics, yeilds a better performance in the Micro mode.
I am wondering, if we use doc2vec method and use CNN or LSTM model as a classifier, what would be the results?