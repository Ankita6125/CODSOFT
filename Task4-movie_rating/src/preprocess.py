import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

cv = CountVectorizer(tokenizer=lambda x: x.split('|'), token_pattern=None)


def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    #  Drop rows where Rating is missing (we're predicting Rating)
    df = df.dropna(subset=['Rating'])

    #  Fill Votes missing with 0
    df['Votes'] = df['Votes'].fillna(0)
    #  Clean Votes column: remove commas, fill NaNs
    df['Votes'] = df['Votes'].astype(str).str.replace(',', '', regex=False)
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0)


    #  Clean Year column like "(2019)" → 2019
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Year'] = df['Year'].fillna(df['Year'].median())

    #  Convert Duration from "109 min" → 109
    df['Duration'] = df['Duration'].astype(str).str.replace('min', '', regex=False)
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    df['Duration'] = df['Duration'].fillna(df['Duration'].median())

    #  Handle Genre (multi-label → one-hot encoding using CountVectorizer)
    df['Genre'] = df['Genre'].fillna('')
    df['Genre'] = df['Genre'].str.replace(', ', '|')
    cv = CountVectorizer(tokenizer=lambda x: x.split('|'), token_pattern=None)
    genre_encoded = cv.fit_transform(df['Genre'])
    genre_df = pd.DataFrame(genre_encoded.toarray(), columns=cv.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), genre_df.reset_index(drop=True)], axis=1)

    #  Encode Director (top 10 only, rest as "Other")
    df['Director'] = df['Director'].fillna('Unknown')
    top_directors = df['Director'].value_counts().nlargest(10).index
    df['Director'] = df['Director'].apply(lambda x: x if x in top_directors else 'Other')
    df = pd.get_dummies(df, columns=['Director'], drop_first=True)

    #  Combine actors and encode top 10 frequent actors
    df['Actors'] = (
        df['Actor 1'].fillna('') + '|' +
        df['Actor 2'].fillna('') + '|' +
        df['Actor 3'].fillna('')
    )

    top_actors = pd.Series('|'.join(df['Actors']).split('|')).value_counts().nlargest(10).index
    for actor in top_actors:
        df[f'Actor_{actor}'] = df['Actors'].apply(lambda x: int(actor in x))

    #  Drop unnecessary columns
    df.drop(['Name', 'Genre', 'Actor 1', 'Actor 2', 'Actor 3', 'Actors'], axis=1, inplace=True, errors='ignore')

    return df
