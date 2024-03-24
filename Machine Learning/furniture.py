import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load dataset CSV
dataset = pd.read_csv('house_prices.csv')

# 2. Preprocessing data
# Misalnya, kita akan menghilangkan kolom 'id' jika ada
dataset = dataset.drop('id', axis=1)

# 3. Pisahkan data menjadi data latih dan data uji
X = dataset.drop('price', axis=1)
y = dataset['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Bangun model Machine Learning
model = RandomForestRegressor()

# 5. Latih model menggunakan data latih
model.fit(X_train, y_train)

# 6. Evaluasi model menggunakan data uji
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 7. Gunakan model untuk membuat prediksi
# Misalnya, kita akan mencoba memprediksi harga rumah untuk data baru
new_data = pd.DataFrame({'bedrooms': [3], 'bathrooms': [2], 'sqft_living': [2000]})
predicted_price = model.predict(new_data)
print(f"Predicted Price for new data: ${predicted_price[0]}")
