import pandas as pd

def preprocess_ml_1m():
    ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', encoding="iso-8859-1",
                             names=['userId', 'itemId', 'rating', 'timestamp'])
    movies_df = pd.read_csv('ml-1m/movies.dat', sep='::', encoding="iso-8859-1", header=None)
    movies_df.columns = ['movieId', 'title', 'genre']
    rm_df = pd.merge(movies_df, ratings_df, left_on='movieId', right_on='itemId')
   #print(rm_df.head())
    #rm_df = rm_df.drop(['itemId'], axis=1)

    features = rm_df.drop(['rating', 'title', 'genre'], axis=1)
    label = pd.DataFrame(rm_df['rating'])

    #rm_df = rm_df.drop(['title', 'genre'])

    return rm_df, features, label, rm_df.userId.unique().shape[0], rm_df.movieId.unique().shape[0]
