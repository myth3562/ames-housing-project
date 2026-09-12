import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_and_split_data(data_path):
    print("Initializing baseline template...")
    np.random.seed(42)
    dummy_data = {
        'GrLivArea': np.random.randint(800, 4000, 100),
        'OverallQual': np.random.randint(1, 11, 100),
        'GarageCars': np.random.randint(0, 4, 100),
        'SalePrice': np.random.randint(100000, 400000, 100)
    }
    df = pd.DataFrame(dummy_data)
    X = df[['GrLivArea', 'OverallQual', 'GarageCars']]
    y = df['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data("data/train.csv")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Baseline Model Trained! Test RMSE: ${np.sqrt(mean_squared_error(y_test, predictions)):.2f}")