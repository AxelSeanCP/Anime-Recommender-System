# -*- coding: utf-8 -*-
"""anime_recommender_system.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-AEj9qhb12G_k5B4ppbudTmeICYVgNF0

# Setup
"""

!pip install opendatasets -q

"""## Import Libraries"""

import pandas as pd
import numpy as np
import opendatasets as od
import matplotlib.pyplot as plt

"""## Data Loading"""

od.download("https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database")

anime = pd.read_csv("/content/anime-recommendations-database/anime.csv")
rating = pd.read_csv("/content/anime-recommendations-database/rating.csv")

print("Total # of samples in anime dataframe: ", len(anime.anime_id.unique()))
print("Total # of samples in rating dataframe: ", len(rating))

"""# Data Understanding
- dataset link: [click here!](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

## EDA - Variable Description

- anime.csv:
  - anime_id: myanimelist.net's unique id identifying an anime.
  - name: full name of anime.
  - genre: comma separated list of genres for this anime.
  - type: type of the anime. movie, TV, OVA, etc.
  - episodes: number of episodes. (1 if movie).
  - rating: average rating out of 10 for this anime.
  - members: number of community members that are in this anime's
"group".
- rating.csv
  - user_id: randomly generated user_id
  - anime_id:  the anime that this user has rated.
  - rating: rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating).

## DataFrame Anime
"""

anime.info()

"""As shown below, the genres in the 'genre' column are in comma-separated values format. This needs to be changed so that the machine can identify the genre of each anime.

The dataset is not clean, so it will be difficult to identify each genre available in the dataset. This will be explained in the **Data Preparation** section.
"""

anime.head()

print(anime.shape)

"""## DataFrame Rating"""

rating.info()

"""Rating dataframe has a lot of samples. This can be computationally expensive to train, to simplify this project the size will be reduced."""

print(rating.shape)

rating.describe()

print("Lowest rating: ", min(rating.rating))
print("Biggest rating: ", max(rating.rating))

print("Total # of user: ", len(rating.user_id.unique()))

"""# Data Preparation

## Anime Data Preparation

### Convert genre from each anime to list
"""

anime['genre'] = anime['genre'].str.split(', ')

anime.head()

"""### Handle missing values for anime dataframe"""

anime.isnull().sum()

anime_clean = anime.dropna()

anime_clean.isnull().sum()

"""### Check unique genres"""

genre_flatten = [genre for sublist in anime_clean['genre'] for genre in sublist]

unique_genres = pd.Series(genre_flatten).unique()
print("Total # of genre: ", len(unique_genres))
print("List of all genre availabel: ", unique_genres)

"""### Drop unused columns"""

anime_new = anime_clean[['anime_id', 'name', 'genre']]
anime_new

"""### Drop rows with "R-rated" genres
- the "R-rated" genres i decided to drop is:
  - Yaoi
  - Yuri
  - Hentai
  - Shounen Ai
  - Shoujo Ai
"""

r_rated_genres = ['Yaoi', 'Yuri', 'Hentai', 'Shounen Ai', 'Shoujo Ai']

mask = anime_new['genre'].apply(lambda x: any(genre in x for genre in r_rated_genres))

anime_final = anime_new[~mask]
anime_final

"""### Convert genre list to string
separates the genre list from each rows with space, and preventing the genre's with space from being separated
"""

anime_final['genre_str'] = anime_final['genre'].apply(lambda x: ' '.join(g.replace(' ', '') for g in x))
anime_final

"""## Rating Data Preparation

### Reduce the size of rating dataframe
"""



"""### Change -1 rating to 0"""

rating['rating'] = rating['rating'].replace(-1, 0)
print("Rating paling kecil: ", min(rating.rating))
print("Rating paling besar: ", max(rating.rating))

"""# Model Development with Content-Based Filtering"""

data = anime_final
data.sample(5)

"""## TF-IDF Vectorizer"""

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()

tf.fit(data['genre_str'])

tf.get_feature_names_out()

tfidf_matrix = tf.fit_transform(data['genre_str'])

tfidf_matrix.shape

"""## View DataFrame"""

# Create dataframe to view tfidf_matrix
# Column is filled with genres
# Row is filled with anime names

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=data['name']
).sample(10, axis=1).sample(5, axis=0)

"""## Cosine Similarity"""

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

"""## Cosine Similarity DataFrame"""

cosine_sim_df = pd.DataFrame(cosine_sim, index=data['name'], columns=data['name'])
print("Shape: ", cosine_sim_df.shape)

cosine_sim_df.sample(10, axis=1).sample(10, axis=0)

"""## Getting top-N Recommendations"""

def anime_recommendations(nama_anime, similarity_data=cosine_sim_df, items=data[['name', 'genre']], k=5):
  """
  Rekomendasi anime berdasarkan kemiripan di dataframe

  Parameter:
  nama_anime: tipe data string (str)
  similarity_data: tipe data pd.DataFrame (object), kesamaan dataframe dengan anime sebagai index dan kolom
  items: tipe data pd.DataFrame (object), mengandung kedua nama dan fitur lainnya untuk mendefinisikan kemiripan
  k: tipe data integer (int), jumlah rekomendasi yang ingin didapatkan
  """

  index = similarity_data.loc[:, nama_anime].to_numpy().argpartition(
      range(-1, -k, -1)
  )

  closest = similarity_data.columns[index[-1:-(k+2):-1]]

  closest = closest.drop(nama_anime, errors='ignore')

  pd.set_option('display.max_columns', None)
  return pd.DataFrame(closest).merge(items).head(k)

anime_input = input("Input anime name: ")
data[data['name'].str.contains(anime_input, case=False)]

# Get top-N Recommendations based from anime input list
anime_recommendations('Kizumonogatari I: Tekketsu-hen', k=10)