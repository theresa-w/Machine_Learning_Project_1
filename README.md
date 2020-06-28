# Machine_Learning_Project_1

## Idea and Model

This notebook uses Python programming language to build a machine learning model to identify the sentiment of customer reviews in a E-Commerce store for Women's Clothing. The information used is a real commercial and anoymised data. Each review has an associated feature on whether the customer who wrote the review would recommend the product, in other words, whether the review is positive or negative. The model would predict whether a review is positive or negative.

The Python Notebook can be found [here](final-Theresa-Wang.ipynb)

## Environment setup and dependencies

To run the Python Notebook, need to install all the relevant packages first:
- [Seaborn](https://seaborn.pydata.org/installing.html)
- [Textblob](https://textblob.readthedocs.io/en/dev/index.html)
- [nltk](http://www.nltk.org/index.html#)
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [tensorflow](https://www.tensorflow.org/install/pip)
- [Keras](https://keras.io/)


## Step-by-step how to train/test the model

Each review is firstly cleaned by removing punctuations, white spaces, special characters. Commonly used words that does not provide value to sentiment analysis (e.g. 'the', 'a', 'is') is removed. Then lemmatisation is performed to normalise the words with the context of vocabulary and morphological analysis of the text.

The two most commonly used method for feature extraction are Bag of Words (BOW) or Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF is used in this analysis. The words of the review are given weights in this method, with the most frequent word is given the least weight and the least frequent word is given the most weight. The purpose is to pick out the distinct words between the reviews that can be used to differentiate each review.

The reviews are split into the training set and test set with approximately 80% trainig and 20% testing. The split is done using the 5-fold cross validation method.

The values in the 'tf_idf' column are the feature vectors, which are the input of the machine learning model. The binary value of the column 'Recommended' is the result that the model try to predict.

The accuracy of the prediction from the test data and a confusion matrix are computed.

Four models are tested - Complement Naive Bayes, SVM, Random Forest, and Deep Learning. SVM predicts the data the best.
