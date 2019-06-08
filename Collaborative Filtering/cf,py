#Tensorflow library. Used to implement machine learning models
import tensorflow as tf
#Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
#Dataframe manipulation library
import pandas as pd
#Graph plotting library
import matplotlib.pyplot as plt
%matplotlib inline


import numpy as np
import pandas as pd

col_user = ['user_id' , 'age' , 'gender' , 'occupation' , 'zip code']
users = pd.read_csv('/gdrive/My Drive/ml-100k/u.user',sep='|',names=col_user,encoding='latin-1')

col_data = ['user_id' , 'movie id', 'rating' , 'timestamp']
ratings = pd.read_csv('/gdrive/My Drive/ml-100k/u.data',sep='\t',names=col_data,encoding='latin-1')
#print(len(ratings))

col_items = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('/gdrive/My Drive/ml-100k/u.item', sep='|', names=col_items,encoding='latin-1')

user_ratings = ratings.pivot(index = 'user_id',columns = 'movie id',values = 'rating')

#normalize the data
norm_user_ratings = user_ratings.fillna(0)/5.0
trX = norm_user_ratings.values

"""
Memory-Based Collaborative Filtering approaches can be divided into two main sections:
user-item filtering and item-item filtering.
A user-item filtering takes a particular user, find users that are similar to that user based on similarity of ratings, and recommend items that those similar users liked.
In contrast, item-item filtering will take an item, find users who liked that item, and find other items that those users or similar users also liked. 
It takes items and outputs other ivtems as recommendations.

Item-Item Collaborative Filtering: “Users who liked this item also liked …”
User-Item Collaborative Filtering: “Users who are similar to you also liked …”
"""

#setting the parameters for the model
n_hidden_units = 30
n_visible_units = len(user_ratings.columns)
print(n_visible_units)
#creating placeholders
visible_neurons_bias = tf.placeholder("float",[n_visible_units])
hidden_neurons_bias = tf.placeholder("float",[n_hidden_units])
weights = tf.placeholder("float",[n_visible_units,n_hidden_units])

# FORWARD PASS
visible_neurons = tf.placeholder("float",[None,n_visible_units])
#finding the conditional probability
hidden_neurons_prob =tf.nn.sigmoid(tf.matmul(visible_neurons,weights)+hidden_neurons_bias)

#sample a hidden activation vector probability
hidden_activation = tf.nn.relu(hidden_neurons_prob - tf.random_uniform(tf.shape(hidden_neurons_prob)))

#BACKWARD PASS
reconstruct_visible_prob = tf.nn.sigmoid(tf.matmul(hidden_activation,tf.transpose(weights))+visible_neurons_bias)
#finding the conditional probability
reconstruct_activation = tf.nn.relu(reconstruct_visible_prob - tf.random_uniform(tf.shape(reconstruct_visible_prob)))
                              
#generate sample
next_hidden_neurons_prob = tf.nn.sigmoid(tf.matmul(reconstruct_activation,weights)+hidden_neurons_bias)

"""
note on Contrastive divergence
https://www.quora.com/What-is-contrastive-divergence
"""

alpha = 0.1

pos_grad = tf.matmul(tf.transpose(visible_neurons),hidden_activation)
neg_grad =  tf.matmul(tf.transpose(reconstruct_activation),next_hidden_neurons_prob)

#calculate Contrastive divergence for matrix updation
cd = (pos_grad - neg_grad)/(tf.to_float(tf.shape(visible_neurons)[0]))

#update of the weight matrix by contrastive divergence (cd)
#Create methods to update the weights and biases
update_weight = weights + alpha * cd
update_visible_neurons_bias= visible_neurons_bias + alpha * tf.reduce_mean(visible_neurons - reconstruct_activation, 0)
update_hidden_neurons_bias = hidden_neurons_bias + alpha * tf.reduce_mean(hidden_activation - next_hidden_neurons_prob, 0)

error = visible_neurons - reconstruct_activation
error_sum =  tf.reduce_mean(error * error)

#Current weight
cur_w = np.zeros([n_visible_units, n_hidden_units], np.float32)

#Current visible unit biases
cur_vb = np.zeros([n_visible_units], np.float32)

#Current hidden unit biases
cur_hb = np.zeros([n_hidden_units], np.float32)

#Previous weight
prv_w = np.zeros([n_visible_units, n_hidden_units], np.float32)

#Previous visible unit biases
prv_vb = np.zeros([n_visible_units], np.float32)

#Previous hidden unit biases
prv_hb = np.zeros([n_hidden_units], np.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 15
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        sess.run(hidden_neurons_bias,feed_dict={hidden_neurons_bias:prv_hb})
        cur_w = sess.run(update_weight, feed_dict={visible_neurons: batch, weights: prv_w, visible_neurons_bias: prv_vb, hidden_neurons_bias: prv_hb})
        cur_vb = sess.run(update_visible_neurons_bias, feed_dict={visible_neurons: batch, weights: prv_w, visible_neurons_bias: prv_vb, hidden_neurons_bias: prv_hb})
        cur_nb = sess.run(update_hidden_neurons_bias, feed_dict={visible_neurons: batch, weights: prv_w, visible_neurons_bias: prv_vb, hidden_neurons_bias: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(error_sum, feed_dict={visible_neurons: trX, weights: cur_w, visible_neurons_bias: cur_vb, hidden_neurons_bias: cur_hb}))
    print (errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()
 
#Recommendation Starts now

mock_user_id = 215

#Selecting the input user
inputUser = trX[mock_user_id-1].reshape(1, -1)
inputUser[0:5]

#Feeding in the user and reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(visible_neurons, weights) + hidden_neurons_bias)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(weights)) + visible_neurons_bias)
feed = sess.run(hh0, feed_dict={ visible_neurons: inputUser, weights: prv_w, hidden_neurons_bias: prv_hb})
rec = sess.run(vv1, feed_dict={ hh0: feed, weights: prv_w, visible_neurons_bias: prv_vb})
print(rec)

scored_movies_df_mock = items[items['movie id'].isin(user_ratings.columns)]
scored_movies_df_mock = scored_movies_df_mock.assign(RecommendationScore = rec[0])
scored_movies_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20)


movies_df_mock = ratings[ratings['user_id'] == mock_user_id]
movies_df_mock.head()

#Merging movies_df with ratings_df by MovieID
merged_df_mock = scored_movies_df_mock.merge(movies_df_mock, on='movie id', how='outer')

merged_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20)
