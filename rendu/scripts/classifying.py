#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


# # Classifying
# Here we just concat the processed text to have different option to test


df = pd.read_json("dataframe.json")
df['processed_text_only'] = df['processed_text'].apply(lambda x: " ".join(x[1]))
df['processed_desc_only'] = df['processed_desc'].apply(lambda x: " ".join(x[1]) if x else "")
df['annotated_text_only'] = df['processed_text'].apply(lambda x: " ".join([f"{z}({y})" for z, y in x[0]]))
df['annotated_desc_only'] = df['processed_desc'].apply(lambda x: " ".join([f"{z}({y})" for z, y in x[0]] if x else ""))
df['processed_all'] = df[["processed_text_only", "processed_desc_only"]].apply(" ".join, axis=1)
df['annotated_all'] = df[["annotated_text_only", "annotated_desc_only"]].apply(" ".join, axis=1)
df['all'] = df[["annotated_all", "processed_all"]].apply(" ".join, axis=1)
df.head()



# This method will return the test categories and prediction
def compute(df, x="processed_all", y='cat', max_iter=15):
    X = df[x]
    Y = df[y]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42) # Split the data
    tfidf_vectorizer = TfidfVectorizer(max_features=8000,
                                       use_idf=True,
                                       stop_words='english',
                                       tokenizer=nltk.word_tokenize,
                                       ngram_range=(1, 3))
    #transform  the dataset
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)
    # Create a Perceptron object
    classifier = Perceptron(max_iter=max_iter)
    # Train the model on the training data
    classifier.fit(X_train_vec, Y_train)
    # Test the model on the test data
    Y_pred = classifier.predict(X_test_vec)
    return Y_test, Y_pred

# This method return all the stats we need for a given test category and prediction
def stats(Y_test, Y_pred):
    # Calculate the precision,recall and fscore form metrics library
    prec, rec, f1, _ = precision_recall_fscore_support(
        Y_test,
        Y_pred,
        average="macro",
        zero_division=0
    )
    # We create 2 confusion_matrix here
    # First one has the name of the categories as the header of the cm
    cm = pd.crosstab(Y_test, Y_pred)
    # Second one has index of the categories as the header of the cm
    conf = confusion_matrix(Y_test, Y_pred)
    cat_acc = {}
    # We use the second one for calculating the accuracy per categories
    for idx, cls in enumerate(set(Y_test)):
        tn = np.sum(np.delete(np.delete(conf, idx, axis=0), idx, axis=1)) # get the true negatives
        tp = conf[idx, idx] # Get true positives
        cat_acc[cls] = (tp + tn) / np.sum(conf) # Calculate the accuracy

    return {"precision": prec, "recall": rec, "f1_score": f1, "accuracy": accuracy_score(Y_test, Y_pred),
            "confusion_matrix": cm, "category_accuracy": cat_acc}

# This method is here to visualize the heatmap of the confusion matrix  and to output the statistics
def see_matrix(stat,Y_pred):
    cm = stat['confusion_matrix']
    unique = len(set(Y_pred))
    plt.figure(figsize=(unique, unique))
    sns.heatmap(cm, annot=True, square=True, cmap="Set1")
    plt.ylabel("Test")
    plt.xlabel("Prediction")
    plt.title("Confusion matrix", size=15)
    print(f"Precision : {stat['precision']}")
    print(f"Recall : {stat['recall']}")
    print(f"F1 Score : {stat['f1_score']}")
    print(f"Accuracy : {stat['accuracy']}")
    print(f"Accuracy per categories : {stat['category_accuracy']}")
    plt.show()

# This method regroup all the above method
def complete(df, x="processed_all", y='cat', max_iter=20):
    Y_test, Y_pred = compute(df, x, y, max_iter)
    stat = stats(Y_test, Y_pred)
    see_matrix(stat,Y_pred)


# ### We test text + description


complete(df)


# ### We test text not preprocessed


complete(df, x='text')


# ### We test text preprocessed only


complete(df, x='processed_text_only')


# ### We test everything (text + description + text annotated + description annotated)


complete(df, x='all')


# ### We test everything annotated (text + description)


complete(df, x='annotated_all', max_iter=20)

