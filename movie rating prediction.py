#movie rating prediction 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = {
    'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'user_id': [101, 102, 103, 101, 102, 103, 101, 102, 103, 101],
    'rating': [4.5, 4.2, 4.8, 4.1, 4.6, 4.9, 4.3, 4.7, 4.0, 4.4]
}

df = pd.DataFrame(data)

X = df[['movie_id', 'user_id']]
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
movie_id = int(input("Enter movie ID: "))
user_id = int(input("Enter user ID: "))


user_input = pd.DataFrame({'movie_id': [movie_id], 'user_id': [user_id]})
prediction = rf.predict(user_input)
print(f"Predicted rating: {prediction[0]:.2f}")
