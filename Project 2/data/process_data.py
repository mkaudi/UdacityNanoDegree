import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This functions reads in the data from the messages and the categories files.
    It formats the data into a combined dataframe.
    The dataframe is then returned.

    :param messages_filepath: the path to the file with the messages
    :param categories_filepath: the path to the file with the categories
    :return: df, a combined dataframe that has both the messages and the categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype("str").str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype("int")

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    # df.reset_index(drop=True, inplace=True)
    # categories.reset_index(drop=True, inplace=True)
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    """
    the function clean the dataframe as it readies it to be output to the database.

    :param df: the dataframe containing the messages with the categories.
    :return: a cleaned dataframe with no duplicates.
    """
    # remove the duplicated rows
    df.drop_duplicates(inplace=True)

    # replace the 2's with 1's
    df[df['related'] == 2] = 1

    return df


def save_data(df, database_filename):
    """
    The function takes a dataframe and creates a SQL database.
    it further creates a table in the database where the data is stored.

    :param df: the dataframe containing the messages and the categories.
    :param database_filename: name of the output database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
