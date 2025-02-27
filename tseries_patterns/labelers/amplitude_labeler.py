import numpy as np
import pandas as pd
from datetime import datetime

class AmplitudeBasedLabeler:
    def __init__(self, minamp: float = 20, Tinactive: int = 10):
        self.minamp = minamp / 10000.0  # Convert bps to decimal
        self.Tinactive = Tinactive
        
    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2:
            return pd.DataFrame({'label': [0]})
            
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Print debug information
        print(f"DataFrame columns: {df_copy.columns.tolist()}")
        print(f"DataFrame index name: {df_copy.index.name}")
        print(f"DataFrame index type: {type(df_copy.index)}")
        
        # Ensure we have the 'close' column
        if 'close' not in df_copy.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        
        prices = df_copy['close'].values
        
        # Find a suitable time column
        time_col = None
        
        # Check for common time column names
        for col in ['timestamp', 'time', 'date', 'datetime']:
            if col in df_copy.columns:
                time_col = col
                print(f"Found time column: {col}")
                break
        
        # If no time column found, check if the index is a datetime
        if time_col is None:
            if isinstance(df_copy.index, pd.DatetimeIndex):
                # Create a time column from the index
                print("Using DatetimeIndex as time column")
                df_copy = df_copy.reset_index()
                time_col = df_copy.columns[0]  # First column after reset_index
                print(f"Reset index, new time column: {time_col}")
            else:
                # If we still don't have a time column, raise an error
                raise ValueError(f"Could not find a suitable time column in the DataFrame. Columns: {df_copy.columns.tolist()}")
        
        # Ensure the time column is in datetime format
        try:
            # Check if the time column contains datetime objects or timestamps
            if pd.api.types.is_datetime64_any_dtype(df_copy[time_col]):
                times = df_copy[time_col].values
            else:
                # Try to convert to datetime, handling both string dates and unix timestamps
                try:
                    # First try standard conversion
                    times = pd.to_datetime(df_copy[time_col]).values
                except:
                    # If that fails, try assuming it's a unix timestamp in milliseconds
                    try:
                        times = pd.to_datetime(df_copy[time_col], unit='ms').values
                    except:
                        # Last resort: create a dummy time series
                        print(f"WARNING: Could not convert {time_col} to datetime. Using dummy time values.")
                        times = np.array([pd.Timestamp(datetime.now()) + pd.Timedelta(minutes=i) for i in range(len(df_copy))])
        except Exception as e:
            raise ValueError(f"Could not convert '{time_col}' column to datetime: {str(e)}")
        
        labels = np.zeros(len(prices))
        last_signal_time = times[0]
        last_signal_price = prices[0]
        current_direction = 0
        
        for i in range(1, len(prices)):
            price_change = (prices[i] - last_signal_price) / last_signal_price
            
            # Safely calculate time difference in minutes
            try:
                # Check if we're dealing with numpy datetime64 objects
                if isinstance(times[i], np.datetime64) and isinstance(last_signal_time, np.datetime64):
                    time_diff = (times[i] - last_signal_time).astype('timedelta64[m]').astype(int)
                # Check if we're dealing with pandas Timestamp objects
                elif isinstance(times[i], pd.Timestamp) and isinstance(last_signal_time, pd.Timestamp):
                    time_diff = int((times[i] - last_signal_time).total_seconds() / 60)
                # If types don't match or aren't datetime objects, use a fallback
                else:
                    # Convert both to pandas Timestamp if possible
                    try:
                        current_time = pd.Timestamp(times[i])
                        last_time = pd.Timestamp(last_signal_time)
                        time_diff = int((current_time - last_time).total_seconds() / 60)
                    except:
                        # If conversion fails, assume 1 minute difference
                        print(f"WARNING: Could not calculate time difference at position {i}. Using default 1 minute.")
                        time_diff = 1
            except Exception as e:
                print(f"WARNING: Error calculating time difference: {str(e)}. Using default 1 minute.")
                time_diff = 1
            
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