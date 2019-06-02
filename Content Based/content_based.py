import pandas as pd
import numpy as np

col_user = ['user_id' , 'age' , 'gender' , 'occupation' , 'zip code']
users = pd.read_csv('/gdrive/My Drive/ml-100k/u.user',sep='|',names=col_user,encoding='latin-1')

col_data = ['user_id' , 'movie id', 'rating' , 'timestamp']
ratings = pd.read_csv('/gdrive/My Drive/ml-100k/u.data',sep='\t',names=col_data,encoding='latin-1')


col_items = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('/gdrive/My Drive/ml-100k/u.item', sep='|', names=col_items,encoding='latin-1')

#print(ratings.head())
#print(items.head())

movie_ratings = pd.merge(ratings, items, on='movie id')  
#print(movie_ratings.head())

#print(movie_ratings.groupby('movie title')['rating'].mean().sort_values(ascending=False).head())
print(movie_ratings.groupby('movie title')['rating'].count().sort_values(ascending=False).head())

ratings_mean_count = pd.DataFrame(movie_ratings.groupby('movie title')['rating'].mean()) 
ratings_mean_count['rating_counts'] = pd.DataFrame(movie_ratings.groupby('movie title')['rating'].count())
#print(ratings_mean_count)

user_movie_rating = movie_ratings.pivot_table(index='user_id', columns='movie title', values='rating')
#print(user_movie_rating)

Star_Wars_ratings = user_movie_rating['Star Wars (1977)'] 
print(Star_Wars_ratings.head())

movies_like_Star_Wars = user_movie_rating.corrwith(Star_Wars_ratings)

corr_Star_Wars = pd.DataFrame(movies_like_Star_Wars, columns=['Correlation'])  
corr_Star_Wars.dropna(inplace=True)  
corr_Star_Wars.head()  
