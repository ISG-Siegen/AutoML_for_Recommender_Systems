import pandas as pd

def preprocess_ml_100k():
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', encoding='iso-8859-1', names=['userId', 'itemId',
                                                                                       'rating', 'timestamp'])
    movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding="iso-8859-1", header=None)
    movies_df.columns = ['movieId', 'title', 'releaseDate', 'videoReleaseDate', 'imdbUrl', 'unknown', 'action',
                         'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
                         'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller',
                         'war', 'western']


    rm_df = pd.merge(movies_df, ratings_df, left_on='movieId', right_on='itemId')
    features = rm_df.drop(['rating', 'title', 'releaseDate', 'imdbUrl', 'videoReleaseDate'], axis=1)
    label = pd.DataFrame(rm_df['rating'])
    rm_df = rm_df.drop(['title', 'releaseDate', 'imdbUrl', 'videoReleaseDate'], axis=1)
    return rm_df, features, label, rm_df.userId.unique().shape[0], rm_df.movieId.unique().shape[0]