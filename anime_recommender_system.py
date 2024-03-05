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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

plt.figure(figsize=(10,6))
plt.hist(rating['rating'], bins=range(-1, 11), color='skyblue', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Jumlah masing-masing skor rating')
plt.grid(axis='y', alpha=0.75)
plt.show()

num_users = len(rating.user_id.unique())
num_anime = len(rating.anime_id.unique())

print("Total # of user: ", num_users)
print("Total # of anime: ", num_anime)

plt.figure(figsize=(8, 6))
plt.bar(['Users', 'Anime'], [num_users, num_anime], color=['blue', 'green'])
plt.title("Total numbers of Users and Anime")
plt.show()

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

rating = rating[:100000]
print("The size of rating dataset: ", len(rating))
rating

num_users = len(rating.user_id.unique())
num_anime = len(rating.anime_id.unique())

print("Total # of users after reduction: ", num_users)
print("Total # of anime after reduction: ", num_anime)

rating.isnull().sum()

"""### Encode the user id and anime id"""

user_ids = rating['user_id'].unique().tolist()
print("List user_id: ", user_ids)

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print("Encoded user_id: ", user_to_user_encoded)

user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print("Decoded user_id: ", user_encoded_to_user)

anime_ids = rating['anime_id'].unique().tolist()
print("List anime_id: ", anime_ids)

anime_to_anime_encoded = {x: i for i, x in enumerate(anime_ids)}
print("Encoded anime_id: ", anime_to_anime_encoded)

anime_encoded_to_anime = {i: x for i, x in enumerate(anime_ids)}
print("Decoded anime_id: ", anime_encoded_to_anime)

# Map the encoded user_id and anime_id into new columns
rating['user'] = rating['user_id'].map(user_to_user_encoded)
rating['anime'] = rating['anime_id'].map(anime_to_anime_encoded)

"""### Change -1 rating to 0"""

rating['rating'] = rating['rating'].replace(-1, 0)

min_rating = min(rating['rating'])
max_rating = max(rating['rating'])

print("Lowest rating: ", min_rating)
print("Highest rating: ", max_rating)

# change to float
rating['rating'] = rating['rating'].values.astype(np.float32)

"""### Split into train and validation

before splitting, randomize the data first
"""

rating = rating.sample(frac=1, random_state=69)
rating

x = rating[['user', 'anime']].values

# Normalization using min max scaler
y = rating['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * rating.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print("Training data: ", x)
print("Validation data: ", y)

"""# Model Development with Content-Based Filtering"""

data = anime_final
data.sample(5)

"""## TF-IDF Vectorizer"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer()

tfv.fit(data['genre_str'])

tfv.get_feature_names_out()

tfidf_matrix = tfv.fit_transform(data['genre_str'])

tfidf_matrix.shape

"""## View DataFrame"""

# Create dataframe to view tfidf_matrix
# Column is filled with genres
# Row is filled with anime names

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfv.get_feature_names_out(),
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

"""# Model Development with Collaborative Filtering

## Create a RecommenderNet Class
"""

class RecommenderNet(tf.keras.Model):

  # Function initialization
  def __init__(self, num_users, num_anime, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_anime = num_anime
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)
    self.anime_embedding = layers.Embedding(
        num_anime,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.anime_bias = layers.Embedding(num_anime, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:, 0])
    user_bias = self.user_bias(inputs[:, 0])
    anime_vector = self.anime_embedding(inputs[:, 1])
    anime_bias = self.anime_bias(inputs[:, 1])

    dot_user_anime = tf.tensordot(user_vector, anime_vector, 2)

    x = dot_user_anime + user_bias + anime_bias

    return tf.nn.sigmoid(x)

"""## Model Compile"""

model = RecommenderNet(num_users, num_anime, 100)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

"""## Model Training"""

modelku = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 48,
    epochs=100,
    validation_data=(x_val, y_val)
)

"""## Metrics Visualization"""

plt.plot(modelku.history['mean_absolute_error'])
plt.plot(modelku.history['val_mean_absolute_error'])
plt.title('model_metrics')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""## Getting top-N Recommendations"""

anime_df = anime_final

# get sample user
user_id = rating.user_id.sample(1).iloc[0]
anime_watched_by_user = rating[rating.user_id == user_id]

anime_not_watched = anime_df[~anime_df['anime_id'].isin(anime_watched_by_user.anime_id.values)]['anime_id']
anime_not_watched = list(
    set(anime_not_watched)
    .intersection(set(anime_to_anime_encoded.keys()))
)

anime_not_watched = [[anime_to_anime_encoded.get(x)] for x in anime_not_watched]
user_encoder = user_to_user_encoded.get(user_id)
user_anime_array = np.hstack(
    ([[user_encoder]] * len(anime_not_watched), anime_not_watched)
)

ratings = model.predict(user_anime_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::1]
recommended_anime_ids = [
    anime_encoded_to_anime.get(anime_not_watched[x][0]) for x in top_ratings_indices
]

print(f"Showing recommendations for user: {user_id}")
print("=" * 40)

print("Anime with high ratings from user")
print("-" * 40)

top_anime_user = (
    anime_watched_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .anime_id.values
)

anime_df_rows = anime_df[anime_df['anime_id'].isin(top_anime_user)]
for row in anime_df_rows.itertuples():
  print(f"{row.name} : {', '.join(row.genre)}")

print('-' * 40)
print("Top 10 anime recommendations")
print('-' * 40)

recommended_anime = anime_df[anime_df['anime_id'].isin(recommended_anime_ids)]
for row in recommended_anime.itertuples():
  print(f"{row.name} : {', '.join(row.genre)}")