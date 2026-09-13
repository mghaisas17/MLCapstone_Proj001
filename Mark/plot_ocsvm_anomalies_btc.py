import matplotlib.pyplot as plt
import pandas as pd
from fetch_binance import fetch_klines
from signature_ocsvm import SignatureOCSVMDetector

def plot_anomalies():
    print("Fetching data (Oct 1-3 for a clearer zoomed-in view)...")
    # Fetching 3 days instead of 7 so the chart isn't too squished
    df = fetch_klines("BTCUSDT", "1m", "2023-10-01", "2023-10-03")
    
    print("Running Signature OCSVM...")
    detector = SignatureOCSVMDetector(window=60, sig_level=2, nu=0.05)
    detector.fit(df)
    anomalies = detector.predict(df)
    
    print("Plotting...")
    plt.figure(figsize=(15, 7))
    
    # Plot normal price line
    plt.plot(df.index, df['close'], label='BTCUSDT Price', color='black', linewidth=1, alpha=0.7)
    
    # Overlay anomalies as red dots
    anomaly_prices = df.loc[anomalies, 'close']
    plt.scatter(anomaly_prices.index, anomaly_prices.values, color='red', label='Path Signature Anomaly', zorder=5, s=20)
    
    plt.title("Path Signature Anomalies (Rolling 60m Window)")
    plt.xlabel("Time")
    plt.ylabel("Price (USDT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "Mark/plots/ocsvm_btc_1m_oct1_oct3_window60.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_anomalies()
