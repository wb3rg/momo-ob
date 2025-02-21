import numpy as np
import pandas as pd

class AmplitudeBasedLabeler:
    def __init__(self, minamp: float = 20, Tinactive: int = 10):
        self.minamp = minamp / 10000.0  # Convert bps to decimal
        self.Tinactive = Tinactive
        
    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2:
            return pd.DataFrame({'label': [0]})
            
        prices = df['close'].values
        times = pd.to_datetime(df['time']).values
        
        labels = np.zeros(len(prices))
        last_signal_time = times[0]
        last_signal_price = prices[0]
        current_direction = 0
        
        for i in range(1, len(prices)):
            price_change = (prices[i] - last_signal_price) / last_signal_price
            time_diff = (times[i] - last_signal_time).astype('timedelta64[m]').astype(int)
            
            if abs(price_change) >= self.minamp:
                current_direction = 1 if price_change > 0 else -1
                last_signal_time = times[i]
                last_signal_price = prices[i]
            elif time_diff >= self.Tinactive:
                current_direction = 0
                last_signal_time = times[i]
                last_signal_price = prices[i]
                
            labels[i] = current_direction
            
        return pd.DataFrame({'label': labels}) 