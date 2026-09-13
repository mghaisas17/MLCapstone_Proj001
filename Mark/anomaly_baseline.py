import numpy as np
import pandas as pd
from fetch_binance import fetch_klines

class BaselineDetector:
    def __init__(self, target_col="close", window=60):
        """
        Args:
            target_col: The single dataframe column to analyze.
            window: Number of periods for the rolling baseline.
        """
        self.target_col = target_col
        self.window = window
        self.rolling_mean = None
        self.rolling_std = None
        
    def fit(self, df):
        # Baseline doesn't strictly need to 'fit' in a machine learning sense, 
        # but we implement it to maintain the uniform interface.
        return self

    def predict(self, df):
        # Compute pct change just for the target column
        # Adding a small epsilon to avoid division by zero if target is volume and hits 0
        returns = df[self.target_col].pct_change().fillna(0)
        
        rolling_mean = returns.rolling(window=self.window).mean()
        rolling_std = returns.rolling(window=self.window).std()
        
        upper_bound = rolling_mean + 2 * rolling_std
        lower_bound = rolling_mean - 2 * rolling_std
        
        anomaly_mask = (returns > upper_bound) | (returns < lower_bound)
        
        # Return list of timestamps
        anomaly_timestamps = df.index[anomaly_mask].tolist()
        return anomaly_timestamps

if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE ANOMALY DETECTION: REAL BINANCE DATA (BTCUSDT)")
    print("=" * 60)
    
    try:
        print("Fetching 1-minute klines for BTCUSDT (2023-10-01 to 2023-10-07)...")
        df_real = fetch_klines("BTCUSDT", "1m", "2023-10-01", "2023-10-07")
        print(f"Fetched {len(df_real)} real samples.")
        
        detector = BaselineDetector(target_col="close", window=60)
        detector.fit(df_real)
        anomalies = detector.predict(df_real)
        
        print(f"Detected {len(anomalies)} price anomalies out of {len(df_real)} samples.")
        print("\nIndices (timestamps) where anomalies were detected (first 20):")
        for idx in anomalies[:20]:
            print(f" - {idx}")
            
    except Exception as e:
        print(f"Failed to process: {e}")
        
    print("=" * 60)
