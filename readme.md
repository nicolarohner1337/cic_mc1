# Paralelized Code

```python
import pandas as pd
import numpy as np
from dask.distributed import Client, wait
from dask import delayed, dataframe as dd
from dask import compute
import time
import glob
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_symbol(filename):
    return os.path.splitext(os.path.basename(filename))[0].split('.')[0]

@delayed
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_class, **kwargs):
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse, model.__class__.__name__

@delayed
def process_and_analyze_file(filename):
    try:
        if os.path.getsize(filename) == 0:
            logging.warning(f"Skipping empty file: {filename}")
            return None
        
        symbol = extract_symbol(filename)
        df = pd.read_csv(filename, parse_dates=['Date'], usecols=['Date', 'Close', 'High', 'Low', 'Volume'])
        
        if df.empty:
            logging.warning(f"File {filename} is empty after reading.")
            return None
        
        # Calculate daily returns
        df['Return'] = df['Close'].pct_change()
        
        # Feature engineering
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Return'].rolling(window=20).std()
        
        # Prepare data for ML models
        df = df.dropna()
        features = ['Close', 'High', 'Low', 'Volume', 'Return', 'MA5', 'MA20', 'Volatility']
        X = df[features]
        y = df['Close'].shift(-1).dropna()
        X = X[:-1]  # Remove last row to align with y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate models in parallel
        models = [
            (RandomForestRegressor, {'n_estimators': 100, 'random_state': 42}),
            (ElasticNet, {'alpha': 1.0, 'random_state': 42}),
            (SVR, {'kernel': 'rbf'}),
            (XGBRegressor, {'n_estimators': 100, 'random_state': 42})
        ]
        
        model_results = [train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, model_class, **params) 
                         for model_class, params in models]
        
        # Compute model results
        model_performances = compute(*model_results)
        
        # Find best model
        best_rmse, best_model = min(model_performances, key=lambda x: x[0])
        
        # Calculate summary statistics
        summary = {
            'symbol': symbol,
            'avg_daily_return': df['Return'].mean(),
            'volatility': df['Return'].std() * np.sqrt(252),  # Annualized volatility
            'sharpe_ratio': df['Return'].mean() / df['Return'].std() * np.sqrt(252),
            'max_drawdown': (df['Close'] / df['Close'].cummax() - 1).min(),
            'avg_volume': df['Volume'].mean(),
            'best_model': best_model,
            'best_model_rmse': best_rmse
        }
        
        return pd.Series(summary)
    except Exception as e:
        logging.error(f"Error processing file {filename}: {str(e)}")
        return None

# Initialize Dask client
client = Client()
logging.info(f"Dask dashboard available at: {client.dashboard_link}")

# Get list of filenames data/Stocks/*.txt and data/ETFs/*.txt
file_pattern = 'data/**/*.txt'
filenames = glob.glob(file_pattern)
print(f"Processing {len(filenames)} files")

# Create delayed objects for each file processing
delayed_results = [process_and_analyze_file(f) for f in filenames]

# Compute all results
futures = client.compute(delayed_results)

# Wait for computations to complete
results = wait(futures)

# Collect results
valid_results = [r.result() for r in results.done if r.result() is not None]

# Create a Dask DataFrame from the results
ddf = dd.from_pandas(pd.DataFrame(valid_results), chunksize=1000)

# Write results to CSV in parallel
write_future = client.compute(ddf.to_csv('export/stock_analysis_results_*.csv', index=False))

# Wait for write operation to complete
wait(write_future)

# Shut down the client
client.close()
````
This took 32:41.8 Minutes to run.