import numpy as np
import pandas as pd
import iisignature
from sklearn.svm import OneClassSVM
from fetch_binance import fetch_klines

class SignatureOCSVMDetector:
    def __init__(self, target_col="close", window=60, sig_level=2, nu=0.05):
        """
        Args:
            target_col: The single dataframe column to analyze (e.g., 'close' or 'volume').
            window: Number of periods in the rolling path.
            sig_level: Truncation level for the path signature.
            nu: Expected anomaly rate for the OCSVM.
        """
        self.target_col = target_col
        self.window = window
        self.sig_level = sig_level
        self.nu = nu
        self.model = OneClassSVM(nu=self.nu, kernel='rbf', gamma='scale')
        
    def _extract_signatures(self, df):
        series = df[self.target_col].values
        n_samples = len(series)
        
        signatures = []
        valid_indices = []
        
        for i in range(self.window, n_samples):
            window_vals = series[i - self.window : i]
            
            # Z-score normalize the path to make it scale-invariant and avoid div-by-zero
            mean_val = np.mean(window_vals)
            std_val = np.std(window_vals) + 1e-8
            p = (window_vals - mean_val) / std_val
            
            t = np.linspace(0, 1, self.window)
            path = np.column_stack((t, p))
            
            sig = iisignature.sig(path, self.sig_level)
            signatures.append(sig)
            valid_indices.append(df.index[i-1])
            
        return np.array(signatures), valid_indices

    def fit(self, df):
        print(f"Extracting signatures for '{self.target_col}' (level {self.sig_level}, window {self.window})...")
        X, _ = self._extract_signatures(df)
        print("Fitting One-Class SVM...")
        self.model.fit(X)
        return self

    def predict(self, df):
        X, valid_indices = self._extract_signatures(df)
        preds = self.model.predict(X)
        
        anomaly_mask = (preds == -1)
        anomaly_timestamps = [valid_indices[i] for i, is_anomaly in enumerate(anomaly_mask) if is_anomaly]
        return anomaly_timestamps

if __name__ == "__main__":
    print("=" * 60)
    print("SIGNATURE OCSVM ANOMALY DETECTION: REAL BINANCE DATA (BTCUSDT)")
    print("=" * 60)
    
    try:
        print("Fetching 1-minute klines for BTCUSDT (2023-10-01 to 2023-10-07)...")
        df_real = fetch_klines("BTCUSDT", "1m", "2023-10-01", "2023-10-07")
        print(f"Fetched {len(df_real)} real samples.")
        
        # Now we can easily pivot to test Volume instead of Price!
        detector = SignatureOCSVMDetector(target_col="volume", window=60, sig_level=2, nu=0.05)
        detector.fit(df_real)
        anomalies = detector.predict(df_real)
        
        print(f"Detected {len(anomalies)} volume anomalies out of {len(df_real)} samples.")
        print("\nIndices (timestamps) where volume anomalies were detected (first 20):")
        for idx in anomalies[:20]:
            print(f" - {idx}")
            
    except Exception as e:
        print(f"Failed to process: {e}")
        
    print("=" * 60)
