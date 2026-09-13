import pandas as pd
import numpy as np

class AnomalyEvaluator:
    def __init__(self, df, lags=[1, 5, 15, 60], debounce_window=60):
        self.df = df.copy()
        self.lags = lags
        self.debounce_window = debounce_window
        self._prepare_baseline()
        
    def _prepare_baseline(self):
        self.baseline_metrics = {}
        for lag in self.lags:
            # Absolute return over the lag window
            fwd_ret = self.df['close'].shift(-lag) / self.df['close'] - 1.0
            fwd_abs_ret = fwd_ret.abs()
            self.df[f'{lag}m_fwd_abs_ret'] = fwd_abs_ret
            
            # Maximum Excursion (Max absolute deviation from entry price during window)
            fwd_max_high = self.df['high'].rolling(window=lag).max().shift(-lag)
            fwd_min_low = self.df['low'].rolling(window=lag).min().shift(-lag)
            
            max_up = (fwd_max_high / self.df['close']) - 1.0
            max_down = 1.0 - (fwd_min_low / self.df['close'])
            
            fwd_max_exc = pd.concat([max_up, max_down], axis=1).max(axis=1)
            self.df[f'{lag}m_fwd_max_exc'] = fwd_max_exc
            
            self.baseline_metrics[lag] = {
                'mean_abs_ret': fwd_abs_ret.mean()
            }
            
    def _debounce_anomalies(self, anomaly_timestamps):
        if not anomaly_timestamps:
            return []
        anomaly_timestamps = pd.Series(anomaly_timestamps).sort_values().tolist()
        debounced = [anomaly_timestamps[0]]
        threshold = pd.Timedelta(minutes=self.debounce_window)
        for ts in anomaly_timestamps[1:]:
            if ts - debounced[-1] >= threshold:
                debounced.append(ts)
        return debounced

    def evaluate(self, anomaly_timestamps, n_bootstraps=1000):
        debounced_ts = self._debounce_anomalies(anomaly_timestamps)
        n_events = len(debounced_ts)
        
        print(f"Total raw anomalies passed: {len(anomaly_timestamps)}")
        print(f"Total debounced 'Events': {n_events}")
        print("-" * 65)
        
        if n_events == 0:
            print("No anomalies to evaluate.")
            return None
            
        events_df = self.df.loc[debounced_ts]
        
        results = []
        for lag in self.lags:
            fwd_abs_ret = events_df[f'{lag}m_fwd_abs_ret'].dropna()
            fwd_max_exc = events_df[f'{lag}m_fwd_max_exc'].dropna()
            
            if len(fwd_abs_ret) == 0:
                continue
                
            mean_abs_ret = fwd_abs_ret.mean()
            mean_max_exc = fwd_max_exc.mean()
            
            base = self.baseline_metrics[lag]
            vol_multiplier = mean_abs_ret / base['mean_abs_ret'] if base['mean_abs_ret'] > 0 else 1.0
            
            # --- BOOTSTRAP P-VALUE ON ABSOLUTE RETURN ---
            all_fwd_abs_ret = self.df[f'{lag}m_fwd_abs_ret'].dropna().values
            random_means = np.random.choice(all_fwd_abs_ret, size=(n_bootstraps, n_events), replace=True).mean(axis=1)
            # P-value is proportion of random portfolios that beat our model's mean absolute return
            p_value = (random_means >= mean_abs_ret).mean()
            
            results.append({
                'Lag': f"{lag}m",
                'Mean Abs Ret': f"{mean_abs_ret*100:.4f}%",
                'Vol Mult': f"{vol_multiplier:.2f}x",
                'Max Exc': f"{mean_max_exc*100:.4f}%",
                'P-Value': f"{p_value:.3f}"
            })
            
        results_df = pd.DataFrame(results).set_index('Lag')
        print(results_df.to_string())
        print("-" * 65)
        return results_df

def get_random_intervals(n=10):
    intervals = []
    start_date = pd.to_datetime("2023-01-01")
    end_date = pd.to_datetime("2023-12-01")
    
    for _ in range(n):
        random_days = np.random.randint(0, (end_date - start_date).days)
        start_ts = start_date + pd.Timedelta(days=random_days)
        end_ts = start_ts + pd.Timedelta(days=7)
        intervals.append((start_ts.strftime('%Y-%m-%d'), end_ts.strftime('%Y-%m-%d')))
    return intervals

if __name__ == "__main__":
    from fetch_binance import fetch_klines
    from anomaly_baseline import BaselineDetector
    from signature_ocsvm import SignatureOCSVMDetector
    import builtins
    
    _original_print = builtins.print
    def _quiet_print(*args, **kwargs):
        pass

    intervals = get_random_intervals(10)
    _original_print("Generated 10 random 7-day intervals across 2023:")
    for inv in intervals:
        _original_print(f"  {inv[0]} to {inv[1]}")
    
    configs = [
        ("BaselineDetector", BaselineDetector, "close"),
        ("BaselineDetector", BaselineDetector, "taker_buy_base"),
        ("SignatureOCSVMDetector", SignatureOCSVMDetector, "close"),
        ("SignatureOCSVMDetector", SignatureOCSVMDetector, "taker_buy_base"),
    ]
    
    raw_results = {f"{name}_{col}": {lag: {'Mean Abs Ret': [], 'Vol Mult': [], 'Max Exc': [], 'P-Value': []} 
                                     for lag in [1, 5, 15, 60]} 
                   for name, _, col in configs}
                   
    for i, (start, end) in enumerate(intervals):
        _original_print(f"\n--- RUN {i+1}/10: Fetching data for {start} to {end} ---")
        try:
            df = fetch_klines("BTCUSDT", "1m", start, end)
        except Exception as e:
            _original_print(f"Failed to fetch {start}-{end}: {e}")
            continue
            
        builtins.print = _quiet_print
        
        for name, cls, col in configs:
            config_key = f"{name}_{col}"
            detector = cls(target_col=col, window=60)
            detector.fit(df)
            anomalies = detector.predict(df)
            
            evaluator = AnomalyEvaluator(df, lags=[1, 5, 15, 60], debounce_window=60)
            eval_res = evaluator.evaluate(anomalies, n_bootstraps=1000)
            if eval_res is None:
                continue
                
            for lag in [1, 5, 15, 60]:
                try:
                    row = eval_res.loc[f"{lag}m"]
                    raw_results[config_key][lag]['Mean Abs Ret'].append(float(row['Mean Abs Ret'].replace('%', '')) / 100.0)
                    raw_results[config_key][lag]['Vol Mult'].append(float(row['Vol Mult'].replace('x', '')))
                    raw_results[config_key][lag]['Max Exc'].append(float(row['Max Exc'].replace('%', '')) / 100.0)
                    raw_results[config_key][lag]['P-Value'].append(float(row['P-Value']))
                except KeyError:
                    pass

        builtins.print = _original_print

    print("\n" + "="*65)
    print("MONTE CARLO AGGREGATED VOLATILITY RESULTS")
    print("="*65)
    
    for config_key in raw_results:
        print(f"\nUNIFIED TESTING FRAMEWORK: {config_key.replace('_', ' on ')}")
        print("-" * 65)
        aggregated = []
        for lag in [1, 5, 15, 60]:
            lag_metrics = {}
            for metric in ['Mean Abs Ret', 'Vol Mult', 'Max Exc', 'P-Value']:
                vals = raw_results[config_key][lag][metric]
                if len(vals) < 3:
                    lag_metrics[metric] = np.nan
                    continue
                vals = sorted(vals)[1:-1]
                lag_metrics[metric] = np.mean(vals)
            
            aggregated.append({
                'Lag': f"{lag}m",
                'Mean Abs Ret': f"{lag_metrics.get('Mean Abs Ret', 0)*100:.4f}%" if pd.notna(lag_metrics.get('Mean Abs Ret')) else "N/A",
                'Vol Mult': f"{lag_metrics.get('Vol Mult', 0):.2f}x" if pd.notna(lag_metrics.get('Vol Mult')) else "N/A",
                'Max Exc': f"{lag_metrics.get('Max Exc', 0)*100:.4f}%" if pd.notna(lag_metrics.get('Max Exc')) else "N/A",
                'P-Value': f"{lag_metrics.get('P-Value', 0):.3f}" if pd.notna(lag_metrics.get('P-Value')) else "N/A"
            })
            
        print(pd.DataFrame(aggregated).set_index('Lag').to_string())
        print("-" * 65)
