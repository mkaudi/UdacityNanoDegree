# import libraries

import sys
import nltk
from sqlalchemy import create_engine
import pandas as pd
import re
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    The function inputs a database file, connects to it with the sqlalchemy.
    the database is then converted to a pandas dataframe from where the model variables are then extracted.

    :param database_filepath: the path of the database to be loaded and read from.
    :return: the variables for training and testing the model( X and Y) and categories' column names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("Messages", engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    cols = df[df.columns[4:]].columns

    return X, Y, cols

def tokenize(text):
    """
    The functions takes text as input and tokenizes it by removing punctuations, converting all letters to lower case,
    removing the stop words and tokenizing it.

    :param text: the text to be tokenized
    :return: tokenized text that is lower case, no punctuation and stopwords.
    """
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text))

    # Convert to lowercase
    try:
        text = text.str.lower()
    except:
        text = text.lower()

    # tokenize
    words = word_tokenize(text)

   #remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]

    return words


def build_model():
    """
    A machine learning pipeline which uses the MultiOutputClassifier with the RandomForestClassifier
    the function goes further to tune/improve the model with GridSearch and returns the tuned model


    :return: ML model pipeline that has been tuned with GridSearch
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function evaluates the model, prints out the accuracy and then iterates through the columns and prints out the
    classification reports for each column.

    :param model: the ML model to evaluate
    :param X_test: the X test variables
    :param Y_test: the Y_test variables
    :param category_names: the names of the categories columns
    """

    Y_pred= model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()
    # print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)

    i = 0
    for col in category_names:
        report = classification_report(Y_test[i], Y_pred[i])
        print(f"Classification report for column '{col}':\n{report}\n")
        i += 1


def save_model(model, model_filepath):
    """
     The function exports the model as a pickle file

    :param model: the model to be exported
    :param model_filepath: the name of the export file
    """

    with open(model_filepath, "wb") as pickle_file:
        pickle.dump(model, pickle_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()