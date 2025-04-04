import pandas as pd
from sklearn.linear_model import LinearRegression

data = {'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'Sales': [100, 120, 110, 130, 140, 150, 160, 170, 180, 190, 200, 210]}

try:
    df = pd.DataFrame(data)

    x = df[['Month']]
    y = df['Sales']

    model = LinearRegression()
    model.fit(x, y)

    month = int(input("Enter month (1-12): "))
    sales = model.predict([[month]])

    print(f"Predicted sales for month {month}: {sales[0]}")
except Exception as e:
    print()
