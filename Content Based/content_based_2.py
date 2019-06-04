col_user = ['user_id' , 'age' , 'gender' , 'occupation' , 'zip code']
users = pd.read_csv('/gdrive/My Drive/ml-100k/u.user',sep='|',names=col_user,encoding='latin-1')

col_data = ['user_id' , 'movie id', 'rating' , 'timestamp']
ratings = pd.read_csv('/gdrive/My Drive/ml-100k/u.data',sep='\t',names=col_data,encoding='latin-1')


col_items = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('/gdrive/My Drive/ml-100k/u.item', sep='|', names=col_items,encoding='latin-1')
n_users = ratings.user_id.unique()

user_ratings = pd.merge(ratings, items, on='movie id')  

#recommendation is being generated for user_id == 1
user1_ratings = user_ratings.loc[ratings['user_id'] == 1]

#making extra column for providing the labeling
user1_ratings['decision'] = 'na'

#labeling the dataset
for index,row in user1_ratings.iterrows():
  if row["rating"] >= 4:
      user1_ratings.at[index, 'decision'] = 1
  else:
      user1_ratings.at[index,'decision'] = 0
  
user1_ratings.drop('timestamp',inplace=True,axis=1)

columns = ['movie title','release date','video release date', 'IMDb URL', 'unknown']
user1_ratings.drop(columns,inplace=True,axis=1)
#print(user1_ratings)

output = user1_ratings['decision']
output = output.to_frame() 
intake = user1_ratings.copy()
intake.drop(['user_id','movie id','decision'],axis=1,inplace=True)

#merging the two dataframe and then finding the corelation
#only taking the first row which is relevant
#print the corelation in order to understand why!!
features = pd.concat([intake,output],axis=1).corr()
features.drop('rating',axis=1,inplace=True)
feature_vector = features[:1]
#print(feature_vector)
#print(feature_vector.T.sort_values(ascending=False,by=['rating']).shape)

items.drop(['movie id','movie title' ,'release date','video release date', 'IMDb URL', 'unknown'],axis=1,inplace=True)
#print(items.head())

#doing the matrix multiplication of the two matrices.
result = feature_vector.dot(items.T)
#print(result.T.sort_values(ascending=False,by=['rating']).head())
#print(result.T)
