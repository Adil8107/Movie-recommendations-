import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dot

# Load dataset
data = pd.read_csv("dataset.csv")

# Encode userId and movieId
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

data['user'] = user_encoder.fit_transform(data['userId'])
data['movie'] = movie_encoder.fit_transform(data['movieId'])

num_users = data['user'].nunique()
num_movies = data['movie'].nunique()

# Split dataset
X = data[['user', 'movie']]
y = data['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ANN Model
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

user_embedding = Embedding(num_users, 10)(user_input)
movie_embedding = Embedding(num_movies, 10)(movie_input)

user_vec = Flatten()(user_embedding)
movie_vec = Flatten()(movie_embedding)

dot_product = Dot(axes=1)([user_vec, movie_vec])

output = Dense(1, activation='relu')(dot_product)

model = Model([user_input, movie_input], output)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit([X_train['user'], X_train['movie']], y_train,
          epochs=20, batch_size=4)

# Evaluate
loss, mae = model.evaluate([X_test['user'], X_test['movie']], y_test)
print("Test MAE:", mae)

# Predict for a user
user_id = 1
encoded_user = user_encoder.transform([user_id])[0]

movie_ids = data['movie'].unique()
predictions = []

for movie in movie_ids:
    pred = model.predict([[encoded_user], [movie]])
    predictions.append((movie, pred[0][0]))

# Sort recommendations
predictions.sort(key=lambda x: x[1], reverse=True)

print("Top Recommendations:")
for movie, rating in predictions[:3]:
    print("Movie:", movie_encoder.inverse_transform([movie])[0],
          "Predicted Rating:", rating)
