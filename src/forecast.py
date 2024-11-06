import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(file_path):
    """
    Load sales data for demand forecasting.
    """
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

def preprocess_data(df):
    """
    Preprocess data by filling missing values and adding time-based features.
    """
    df.fillna(method='ffill', inplace=True)
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    return df

def train_demand_forecasting_model(df, target_column='sales'):
    """
    Train a linear regression model for demand forecasting.
    """
    X = df[['month', 'day_of_week']]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")
    return model, rmse

def forecast_demand(model, X_future):
    """
    Use the trained model to forecast demand on new data.
    """
    return model.predict(X_future)

