import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter, date2num, AutoDateLocator
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
# Import directly from GitHub repository
import sys
import os
import subprocess

# Check if tseries-patterns is installed, if not clone from GitHub
if not os.path.exists('tseries-patterns'):
    subprocess.check_call([
        'git', 'clone', 'https://github.com/tr8dr/tseries-patterns.git'
    ])
    
# Add the repository to the Python path
if 'tseries-patterns' not in sys.path:
    sys.path.append('tseries-patterns')

# Import the AmplitudeBasedLabeler
try:
    # Try different import paths to handle various project structures
    try:
        from tseries_patterns.labelers.amplitude_labeler import AmplitudeBasedLabeler
    except ImportError:
        try:
            from tseries_patterns import AmplitudeBasedLabeler
        except ImportError:
            # Last resort: try to import from the local directory
            sys.path.append('.')
            from tseries_patterns.labelers.amplitude_labeler import AmplitudeBasedLabeler
except ImportError as e:
    st.error(f"Error importing AmplitudeBasedLabeler: {str(e)}")
    st.info("Make sure the tseries-patterns repository is properly cloned and accessible.")

from typing import Dict, List, Optional, Tuple, Union
import copy

# Configure page
st.set_page_config(
    page_title="Quantavius Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    .Widget>label {
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# Configuration settings
CONFIG = {
    'trading': {
        'timeframe': '1m',
        'exchange': 'coinbase',
    },
    'visualization': {
        'figure_size': (16, 8),
        'price_color': '#c0c0c0',
        'vwap_above_color': '#3399ff',
        'vwap_below_color': '#ff4d4d',
        'vwma_color': '#90EE90',
        'up_color': '#3399ff',
        'down_color': '#ff4d4d',
        'bullish_color': '#3399ff',  # Blue for bullish momentum
        'bearish_color': '#ff4d4d',   # Red for bearish momentum
        'neutral_color': '#cccccc',   # Gray for neutral momentum
        'volume_colors': {
            'high': '#3399ff',
            'medium': '#cccccc',
            'low': '#ff4d4d'
        },
        'base_bubble_size': 35,
        'volume_bins': 50,
        'watermark': {
            'size': 15,
            'color': '#999999',
            'alpha': 0.25
        }
    },
    'analysis': {
        'amplitude_threshold': 20,
        'inactive_period': 10,
        'vwma_period': 20
    }
}

def initialize_exchange(price_exchange_name='coinbase', orderbook_exchange_name='kraken'):
    """Initialize the cryptocurrency exchange connections.
    
    Args:
        price_exchange_name: Name of the exchange to use for price data
        orderbook_exchange_name: Name of the exchange to use for orderbook data
    
    Returns:
        Dictionary with initialized exchange objects
    """
    # Common exchange settings
    common_settings = {
        'enableRateLimit': True,
        'timeout': 30000,
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        },
        'options': {
            'fetchOHLCVWarning': False
        }
    }
    
    # Initialize price exchange
    if price_exchange_name == 'coinbase':
        price_exchange = ccxt.coinbase({
            **common_settings,
            'options': {
                'fetchOHLCVWarning': False,
                'createMarketBuyOrderRequiresPrice': False
            }
        })
    elif price_exchange_name == 'binance':
        price_exchange = ccxt.binance(common_settings)
    elif price_exchange_name == 'kraken':
        price_exchange = ccxt.kraken(common_settings)
    elif price_exchange_name == 'kucoin':
        price_exchange = ccxt.kucoin(common_settings)
    elif price_exchange_name == 'bitfinex':
        price_exchange = ccxt.bitfinex(common_settings)
    else:
        # Default to coinbase if exchange not supported
        price_exchange = ccxt.coinbase({
            **common_settings,
            'options': {
                'fetchOHLCVWarning': False,
                'createMarketBuyOrderRequiresPrice': False
            }
        })
    
    # Initialize orderbook exchange
    if orderbook_exchange_name == 'kraken':
        orderbook_exchange = ccxt.kraken(common_settings)
    elif orderbook_exchange_name == 'binance':
        orderbook_exchange = ccxt.binance(common_settings)
    elif orderbook_exchange_name == 'coinbase':
        orderbook_exchange = ccxt.coinbase({
            **common_settings,
            'options': {
                'fetchOHLCVWarning': False,
                'createMarketBuyOrderRequiresPrice': False
            }
        })
    elif orderbook_exchange_name == 'kucoin':
        orderbook_exchange = ccxt.kucoin(common_settings)
    elif orderbook_exchange_name == 'bitfinex':
        orderbook_exchange = ccxt.bitfinex(common_settings)
    else:
        # Default to kraken if exchange not supported
        orderbook_exchange = ccxt.kraken(common_settings)
    
    return {'price': price_exchange, 'orderbook': orderbook_exchange}

def fetch_market_data(exchange: ccxt.Exchange, symbol: str, timeframe: str = '1m', limit: int = 1000, lookback_hours: int = 24) -> List[List]:
    """
    Fetch OHLCV data from exchange with retry logic and proper timestamp handling
    """
    # Reset retry count if this is a new call or if more than 60 seconds have passed
    current_time = time.time()
    if not hasattr(fetch_market_data, 'last_call_time') or current_time - fetch_market_data.last_call_time > 60:
        fetch_market_data.retry_count = 0
    fetch_market_data.last_call_time = current_time
    
    # Ensure markets are loaded
    if not exchange.markets:
        try:
            exchange.load_markets()
        except Exception as e:
            st.error(f"Error loading markets: {str(e)}")
            return []
    
    # Special handling for Coinbase
    if exchange.id.lower() == 'coinbase':
        # Force BTC/USD format for Coinbase regardless of input
        if 'btc' in symbol.lower() and ('usd' in symbol.lower() or 'usdt' in symbol.lower()):
            formatted_symbol = 'BTC/USD'
        elif 'eth' in symbol.lower() and ('usd' in symbol.lower() or 'usdt' in symbol.lower()):
            formatted_symbol = 'ETH/USD'
        else:
            # Use the convert_symbol_format as fallback
            formatted_symbol = convert_symbol_format(symbol, exchange.id)
        
        # Coinbase API expects symbols in format like "BTC-USD" not "BTC/USD"
        formatted_symbol = formatted_symbol.replace('/', '-')
    else:
        # For other exchanges, use the normal conversion
        formatted_symbol = convert_symbol_format(symbol, exchange.id)
    
    # Debug output
    print(f"Using symbol {formatted_symbol} for {exchange.id}")
    
    # Check if symbol exists in exchange
    if formatted_symbol not in exchange.markets:
        # Try to find similar symbols or suggest default symbols
        similar_symbols = [s for s in exchange.markets.keys() if formatted_symbol.split('/')[0] in s if '/' in formatted_symbol][:5]
        default_symbols = ['BTC/USD', 'BTC/USDT', 'ETH/USD', 'ETH/USDT']
        available_defaults = [s for s in default_symbols if s in exchange.markets]
        
        error_msg = f"Symbol '{formatted_symbol}' not found in {exchange.id} markets."
        if similar_symbols:
            error_msg += f" Similar symbols: {', '.join(similar_symbols)}"
        if available_defaults:
            error_msg += f" Available default symbols: {', '.join(available_defaults)}"
        
        st.error(error_msg)
        return []
    
    # Calculate the start time for fetching data
    # For 4-hour timeframe, ensure we're getting the most recent data
    if lookback_hours <= 4:
        # Use current time to ensure we get the most recent data
        since = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)
    else:
        # For longer timeframes, use the standard approach
        since = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)
    
    # Add cache invalidation by adding a unique parameter
    cache_buster = int(time.time())
    
    try:
        # Fetch OHLCV data with explicit params to force fresh data
        params = {
            '_': cache_buster,
            'limit': limit
        }
        
        # Add exchange-specific parameters
        if exchange.id.lower() == 'coinbase':
            granularity = timeframe_to_seconds(timeframe)
            # Coinbase expects granularity as an integer, not a string
            params['granularity'] = int(granularity)
            st.info(f"Using Coinbase with granularity: {granularity} seconds")
        
        try:
            data = exchange.fetch_ohlcv(formatted_symbol, timeframe, since, limit, params=params)
        except Exception as specific_error:
            error_message = str(specific_error).lower()
            if exchange.id.lower() == 'coinbase' and 'granularity' in error_message:
                st.error(f"Coinbase granularity error: {specific_error}")
                st.info("Trying with default granularity of 60 seconds (1 minute)")
                # Ensure granularity is an integer
                params['granularity'] = int(60)
                data = exchange.fetch_ohlcv(formatted_symbol, timeframe, since, limit, params=params)
            else:
                raise  # Re-raise the exception if it's not a granularity issue
        
        if not data:
            st.warning(f"No data returned for {formatted_symbol} on {exchange.id}")
            return []
        
        # Check the timestamp of the most recent data point
        if data and len(data) > 0:
            latest_timestamp = data[-1][0]
            latest_time = datetime.fromtimestamp(latest_timestamp / 1000)
            current_time = datetime.now()
            time_diff = current_time - latest_time
            minutes_old = time_diff.total_seconds() / 60
            
            # Display warning if data is more than 5 minutes old
            if minutes_old > 5:
                st.warning(f"Data may be delayed. Latest data point: {latest_time.strftime('%Y-%m-%d %H:%M:%S')} ({minutes_old:.1f} minutes ago)")
            else:
                st.success(f"Using fresh data. Latest point: {latest_time.strftime('%Y-%m-%d %H:%M:%S')} ({minutes_old:.1f} minutes ago)")
        
        # Convert OHLCV data from list to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Ensure timestamp is properly converted to datetime
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            except (TypeError, ValueError) as e:
                st.error(f"Error converting timestamp: {str(e)}")
                # Try alternative conversion methods
                try:
                    # If timestamps are already datetime objects or strings
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except:
                    # Create synthetic timestamps as last resort
                    st.warning("Could not convert timestamps. Creating synthetic timestamps.")
                    df['timestamp'] = pd.date_range(
                        start=datetime.now() - timedelta(minutes=len(df)), 
                        periods=len(df), 
                        freq='1min'
                    )
        else:
            df = data
        
        return df
    
    except Exception as e:
        # Increment retry count
        if not hasattr(fetch_market_data, 'retry_count'):
            fetch_market_data.retry_count = 0
        fetch_market_data.retry_count += 1
        
        # Log the error
        st.error(f"Error in fetch_market_data: {str(e)}")
        
        # Retry logic with exponential backoff (up to 3 retries)
        if fetch_market_data.retry_count <= 3:
            wait_time = 2 ** (fetch_market_data.retry_count - 1)
            st.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            return fetch_market_data(exchange, symbol, timeframe, limit, lookback_hours)
        else:
            st.error("Maximum retry attempts reached. Please try again later.")
            return []

def timeframe_to_seconds(timeframe):
    """Convert timeframe string to seconds for Coinbase API
    
    Coinbase only accepts specific granularity values:
    - 60 (1 minute)
    - 300 (5 minutes)
    - 900 (15 minutes)
    - 3600 (1 hour)
    - 21600 (6 hours)
    - 86400 (1 day)
    """
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    
    seconds = 0
    if unit == 'm':
        seconds = value * 60
    elif unit == 'h':
        seconds = value * 60 * 60
    elif unit == 'd':
        seconds = value * 60 * 60 * 24
    else:
        seconds = 60  # Default to 1 minute
    
    # Map to closest valid Coinbase granularity
    valid_granularities = [60, 300, 900, 3600, 21600, 86400]
    
    # Find the closest valid granularity
    closest = min(valid_granularities, key=lambda x: abs(x - seconds))
    
    print(f"Converting timeframe {timeframe} to Coinbase granularity: {closest} seconds")
    # Ensure we return an integer
    return int(closest)

def calculate_vwma(df, period):
    """Calculate Volume Weighted Moving Average"""
    df['cum_vol'] = df['volume'].rolling(window=period, min_periods=1).sum()
    df['cum_vol_price'] = (df['close'] * df['volume']).rolling(window=period, min_periods=1).sum()
    df['vwma'] = df['cum_vol_price'] / df['cum_vol']
    return df

def calculate_metrics(df, ccxt_client, symbol, orderbook_depth):
    """Calculate various market metrics including VWAP and order book imbalance."""
    try:
        # Convert DataFrame to proper format if it's not already
        if isinstance(df, list):
            # Convert OHLCV data from list to DataFrame
            df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Ensure timestamp is properly converted to datetime
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            except (TypeError, ValueError) as e:
                st.error(f"Error converting timestamp: {str(e)}")
                # Try alternative conversion methods
                try:
                    # If timestamps are already datetime objects or strings
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except:
                    # Create synthetic timestamps as last resort
                    st.warning("Could not convert timestamps. Creating synthetic timestamps.")
                    df['timestamp'] = pd.date_range(
                        start=datetime.now() - timedelta(minutes=len(df)), 
                        periods=len(df), 
                        freq='1min'
                    )
        
        # If df is not a DataFrame after conversion, return empty DataFrame
        if not isinstance(df, pd.DataFrame):
            st.error(f"Invalid data format: {type(df)}")
            return pd.DataFrame()
        
        # Print DataFrame information for debugging
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame index name: {df.index.name}")
        print(f"DataFrame index type: {type(df.index)}")
        
        # Check if timestamp is in columns or index
        if 'timestamp' in df.columns:
            print("Found time column: timestamp")
            # Keep timestamp as a column for now, we'll set it as index later
        elif df.index.name == 'timestamp':
            # If timestamp is already the index, reset it to make it a column
            df = df.reset_index()
            print("Reset index to make timestamp a column")
        else:
            st.error("No timestamp column or index found in data")
            return pd.DataFrame()
        
        # Ensure all required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Missing required columns. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Calculate VWAP - reset at the start of each session
        try:
            # Handle zero volume case
            if df['volume'].sum() == 0:
                df['cum_vol'] = 1  # Avoid division by zero
                df['cum_vol_price'] = df['close']
                df['vwap'] = df['close']  # Set VWAP equal to close price when no volume
            else:
                df['cum_vol'] = df['volume'].cumsum()
                df['cum_vol_price'] = (df['close'] * df['volume']).cumsum()
                df['vwap'] = df['cum_vol_price'] / df['cum_vol']
        except Exception as e:
            st.error(f"Error calculating VWAP: {str(e)}")
            # Provide fallback values
            df['cum_vol'] = 1
            df['cum_vol_price'] = df['close']
            df['vwap'] = df['close']  # Use close price as fallback
        
        # Calculate VWMA
        try:
            df = calculate_vwma(df, CONFIG['analysis']['vwma_period'])
        except Exception as e:
            st.error(f"Error calculating VWMA: {str(e)}")
            df['vwma'] = df['close']  # Use close price as fallback
        
        # Calculate order book metrics using CCXT
        try:
            order_book = fetch_order_book(ccxt_client, symbol, orderbook_depth)
            
            # Debug order book structure
            debug_order_book(order_book, symbol, ccxt_client.id)
            
            if order_book and 'bids' in order_book and 'asks' in order_book:
                # Safely extract bid and ask volumes
                bids_volume = 0
                asks_volume = 0
                
                for bid in order_book['bids'][:orderbook_depth]:
                    if isinstance(bid, list) and len(bid) > 1:
                        bids_volume += bid[1]
                
                for ask in order_book['asks'][:orderbook_depth]:
                    if isinstance(ask, list) and len(ask) > 1:
                        asks_volume += ask[1]
                
                if bids_volume + asks_volume > 0:
                    df['orderbook_imbalance'] = bids_volume / (bids_volume + asks_volume)
                else:
                    df['orderbook_imbalance'] = 0.5  # Default to neutral if no volume
            else:
                df['orderbook_imbalance'] = 0.5  # Default to neutral if no bids/asks
                
            # Calculate bubble sizes
            df['bubble_size'] = df['volume'] * df['orderbook_imbalance']
            if df['bubble_size'].max() > 0:
                df['bubble_size'] = df['bubble_size'] / df['bubble_size'].max()
            else:
                df['bubble_size'] = 1.0  # Default if all values are 0
        except Exception as e:
            st.error(f"Error in calculate_metrics: {str(e)}")
            df['orderbook_imbalance'] = 0.5  # Default to neutral
            df['bubble_size'] = 1.0  # Default size
        
        # Create AmplitudeBasedLabeler instance with parameters from CONFIG
        try:
            labeler = AmplitudeBasedLabeler(
                minamp=CONFIG['analysis']['amplitude_threshold'],
                Tinactive=CONFIG['analysis']['inactive_period']
            )
            
            # Label the data
            labels_df = labeler.label(df)
            
            # Extract momentum values
            if 'label' in labels_df.columns:
                momentum_labels = labels_df['label']
            else:
                momentum_labels = labels_df.iloc[:, 0]
            
            # Add labels to DataFrame and set index back
            df['momentum_label'] = momentum_labels
        except Exception as e:
            st.error(f"Error in momentum labeling: {str(e)}")
            df['momentum_label'] = 0  # Default to neutral
        
        # Ensure timestamp is a proper datetime object but don't set it as index
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        st.error(f"Error in calculate_metrics: {str(e)}")
        # Return a minimal DataFrame with the required columns to prevent downstream errors
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            # Try to salvage what we can from the original DataFrame
            for col in ['vwap', 'vwma', 'orderbook_imbalance', 'bubble_size', 'momentum_label']:
                if col not in df.columns:
                    df[col] = 0 if col != 'momentum_label' else 0
            return df
        else:
            # Create a minimal DataFrame with default values
            return pd.DataFrame({
                'timestamp': [pd.Timestamp.now()],
                'open': [0], 'high': [0], 'low': [0], 'close': [0], 'volume': [0],
                'vwap': [0], 'vwma': [0], 'orderbook_imbalance': [0.5], 
                'bubble_size': [1.0], 'momentum_label': [0]
            })

# Visualization functions
def plot_price_action(ax, df, x_values):
    """Plot price action including price line, VWAP, VWMA, and bubbles."""
    # Plot main price line
    ax.plot(x_values, df['close'].values,
            color=CONFIG['visualization']['price_color'],
            linestyle='-', alpha=1.0, linewidth=1.0,  # Increased visibility
            zorder=2, marker='', markersize=0)  # Decreased zorder to be behind bubbles
    
    # Plot VWAP with dynamic coloring
    above_vwap = df['close'] >= df['vwap']
    
    # Plot VWAP segments
    for i in range(1, len(df)):
        if above_vwap.iloc[i]:
            color = CONFIG['visualization']['vwap_above_color']
        else:
            color = CONFIG['visualization']['vwap_below_color']
        
        ax.plot(x_values[i-1:i+1], df['vwap'].iloc[i-1:i+1],
                color=color, linestyle='-', alpha=0.5, linewidth=0.8,
                zorder=1)
    
    # Plot VWMA
    ax.plot(x_values, df['vwma'].values,
            color=CONFIG['visualization']['vwma_color'],
            linestyle='--', alpha=0.5, linewidth=0.8,
            zorder=1)
    
    # Plot bubbles for all points with a minimum size
    min_bubble_size = 20  # Minimum bubble size to ensure visibility
    
    # Create a default neutral color for points without momentum labels
    neutral_color = 'gray'
    
    # Plot all points with a small neutral bubble to ensure all data is visible
    ax.scatter(x_values, df['close'].values, 
               s=min_bubble_size, 
               color=neutral_color,
               alpha=0.3, 
               zorder=3)
    
    # Plot bubbles for points with momentum labels
    for i in range(len(df)):
        # Skip points without momentum labels
        if 'momentum_label' not in df.columns or pd.isna(df['momentum_label'].iloc[i]):
            continue
            
        momentum = df['momentum_label'].iloc[i]
        
        # Skip neutral points (they're already plotted with the neutral color)
        if momentum == 0:
            continue
            
        # Determine bubble size based on volume and orderbook imbalance
        if 'bubble_size' in df.columns:
            size = df['bubble_size'].iloc[i] * 100 + min_bubble_size
        else:
            size = df['volume'].iloc[i] / df['volume'].max() * 100 + min_bubble_size
        
        # Determine color based on momentum
        if momentum > 0:
            color = CONFIG['visualization']['bullish_color']
        elif momentum < 0:
            color = CONFIG['visualization']['bearish_color']
        else:
            color = neutral_color
        
        # Plot the bubble
        ax.scatter(x_values[i], df['close'].iloc[i], 
                   s=size, 
                   color=color,
                   alpha=0.7, 
                   zorder=4)
    
    # Remove grid
    ax.grid(False)
    
    # Format y-axis with dollar signs
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

def plot_volume_profile(ax, df):
    """Plot the volume profile with momentum-based coloring."""
    price_bins = np.linspace(df['close'].min(), df['close'].max(),
                           CONFIG['visualization']['volume_bins'])
    
    # Create separate histograms for different momentum labels
    volume_hist_up = np.histogram(df.loc[df['momentum_label'] > 0, 'close'].values,
                                bins=price_bins,
                                weights=df.loc[df['momentum_label'] > 0, 'volume'].values)[0]
    volume_hist_down = np.histogram(df.loc[df['momentum_label'] < 0, 'close'].values,
                                  bins=price_bins,
                                  weights=df.loc[df['momentum_label'] < 0, 'volume'].values)[0]
    volume_hist_neutral = np.histogram(df.loc[df['momentum_label'] == 0, 'close'].values,
                                     bins=price_bins,
                                     weights=df.loc[df['momentum_label'] == 0, 'volume'].values)[0]
    
    # Calculate total volume for normalization
    total_volume = volume_hist_up + volume_hist_down + volume_hist_neutral
    max_total = total_volume.max() if total_volume.max() > 0 else 1
    
    # Calculate bar width
    max_width = (price_bins[1] - price_bins[0]) * 20
    
    # Plot horizontal volume bars
    for i in range(len(price_bins) - 1):
        price = (price_bins[i] + price_bins[i+1]) / 2
        # Calculate bar height (slightly smaller than bin size for visual separation)
        height = (price_bins[i+1] - price_bins[i]) * 0.95
        
        # Find dominant momentum with priority (up > down > neutral)
        volumes = [
            (volume_hist_up[i], CONFIG['visualization']['volume_colors']['high']),
            (volume_hist_down[i], CONFIG['visualization']['volume_colors']['low']),
            (volume_hist_neutral[i], CONFIG['visualization']['volume_colors']['medium'])
        ]
        # Sort by volume and give priority to up/down over neutral
        sorted_volumes = sorted(volumes, key=lambda x: (x[0], x[1] != CONFIG['visualization']['volume_colors']['medium']), reverse=True)
        max_vol, color = sorted_volumes[0]
        
        # Only plot if there's volume
        if max_vol > 0:
            normalized_vol = -(max_vol / max_total * max_width)
            ax.barh(price, normalized_vol, height=height,
                   color=color, alpha=0.8,  # Slightly reduced alpha for better visibility
                   edgecolor='black', linewidth=0.5)
    
    # Style the volume profile box
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
        spine.set_visible(True)
    
    # Remove all ticks and labels from volume profile
    ax.set_xticks([])
    ax.set_yticks([])
    
    return max_width

def style_plot(fig, ax_price, ax_vol, df, x_values, max_width):
    """Apply styling to the plot."""
    # Calculate padding
    x_padding = (x_values.max() - x_values.min()) * 0.02
    y_padding = (df['close'].max() - df['close'].min()) * 0.05
    
    # Configure time axis with smaller font
    ax_price.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax_price.xaxis.set_major_locator(AutoDateLocator())
    ax_price.tick_params(axis='x', labelsize=6)
    
    # Remove all spines from price plot
    for spine in ax_price.spines.values():
        spine.set_visible(False)
    
    # Style volume profile spines
    for spine in ax_vol.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
        spine.set_visible(True)
    
    # Customize ticks
    ax_price.tick_params(axis='both', colors='#666666', length=3, width=0.5)
    ax_vol.tick_params(axis='x', colors='#666666', length=2, width=0.5)
    ax_vol.tick_params(axis='y', colors='#666666', length=0, width=0)
    
    # Set plot limits with padding
    ax_price.set_xlim(x_values.min() - x_padding, x_values.max() + x_padding)
    ax_vol.set_xlim(-max_width * 1.1, 0)  # Added 10% padding
    
    # Final adjustments
    plt.tight_layout()

def create_price_table(df):
    """Create a formatted price data table."""
    # Take the 5 most recent bars
    sampled_df = df.sort_index().tail(5)
    
    # Format the data
    table_data = {
        'Time/Price': [f"{t.strftime('%H:%M') if hasattr(t, 'strftime') else pd.Timestamp(t).strftime('%H:%M')} - ${p:,.2f}"  
                      for t, p in zip(sampled_df.index, sampled_df['close'])],
        'Momentum': sampled_df['momentum_label'].map({1: '↑', 0: '•', -1: '↓'}),  # Changed neutral symbol to dot
        'OB Imbalance': [(f"↑ {imb*100:.1f}%" if imb > 0 else f"↓ {imb*100:.1f}%") for imb in sampled_df['orderbook_imbalance']],
        'Volume': sampled_df['volume'].round(2).map('{:,.2f}'.format),  # Added thousands separator
        'VWAP Distance': ((sampled_df['close'] / sampled_df['vwap'] - 1) * 100).round(2).astype(str) + '%'
    }
    
    # Create DataFrame and reverse the order so most recent is at top
    return pd.DataFrame(table_data).iloc[::-1]

def fetch_order_book(exchange: ccxt.Exchange, symbol: str, limit: int = 1000) -> Dict:
    """
    Fetch the order book for a given symbol.
    
    Args:
        exchange: The exchange to fetch data from
        symbol: The trading pair symbol
        limit: The maximum number of price levels to fetch
        
    Returns:
        A dictionary containing the order book data
    """
    # Maximum number of retries
    max_retries = 3
    retry_count = 0
    
    # Fetch data with retries
    while retry_count < max_retries:
        try:
            # Special handling for Coinbase
            if exchange.id == 'coinbase':
                try:
                    # Coinbase's order book structure is different
                    order_book = exchange.fetch_order_book(symbol, limit=limit)
                    return order_book
                except Exception as e:
                    print(f"Error fetching Coinbase order book: {str(e)}")
                    retry_count += 1
                    time.sleep(2)
            else:
                # For other exchanges
                order_book = exchange.fetch_order_book(symbol, limit=limit)
                return order_book
        except Exception as e:
            print(f"Error in fetch_order_book: {str(e)}")
            retry_count += 1
            time.sleep(2)  # Wait before retrying
    
    # If no data was fetched after all attempts, return an empty order book
    return {'bids': [], 'asks': []}

def limit_price_levels(order_book: Dict, max_levels: int = 100) -> Dict:
    """Limit the number of price levels in the order book to prevent excessive image sizes."""
    if not order_book or 'bids' not in order_book or 'asks' not in order_book:
        return order_book
    
    # Create a copy to avoid modifying the original
    limited_book = {
        'bids': order_book['bids'][:max_levels] if len(order_book['bids']) > max_levels else order_book['bids'],
        'asks': order_book['asks'][:max_levels] if len(order_book['asks']) > max_levels else order_book['asks']
    }
    
    return limited_book

def plot_order_book_heatmap(order_book1: Dict, order_book2: Dict, symbol: str, min_amplitude: float = 0.01, figsize=(12, 8)):
    """Create an enhanced order book visualization with heatmap, volume delta, and OI delta.
    
    Args:
        order_book1: First order book snapshot
        order_book2: Second order book snapshot
        symbol: Trading symbol
        min_amplitude: Minimum change percentage to be considered significant (default: 0.01)
        figsize: Figure size as a tuple (width, height)
        
    Returns:
        The matplotlib figure object
    """
    if not order_book1 or not order_book2 or 'bids' not in order_book1 or 'asks' not in order_book1:
        # Return an empty figure if order books are invalid
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No valid order book data available", 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        return fig
    
    # Create figure and subplots
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    axes = [ax1, ax2, ax3]
    
    # Convert min_amplitude to threshold
    threshold = min_amplitude / 100.0 if min_amplitude > 1 else min_amplitude
    show_all_changes = (min_amplitude == 0)
    
    # Call the original implementation with the correct parameters
    plot_order_book_heatmap_impl(fig, axes, order_book1, order_book2, symbol, show_all_changes, threshold)
    
    return fig

def plot_order_book_heatmap_impl(fig, axes, order_book1: Dict, order_book2: Dict, symbol: str, show_all_changes: bool = True, threshold: float = 0.01):
    """Create an enhanced order book visualization with heatmap, volume delta, and OI delta.
    
    Args:
        fig: The matplotlib figure
        axes: List of three axes for the subplots
        order_book1: First order book snapshot
        order_book2: Second order book snapshot
        symbol: Trading symbol
        show_all_changes: Whether to show all changes or only significant ones (default: True)
        threshold: Minimum change to be considered significant (default: 0.01)
    """
    if not order_book1 or not order_book2 or 'bids' not in order_book1 or 'asks' not in order_book1:
        return
    
    # Extract the base and quote currency from the symbol
    if '-' in symbol:
        base_currency, quote_currency = symbol.split('-')
    elif '/' in symbol:
        base_currency, quote_currency = symbol.split('/')
    else:
        # Default fallback if symbol format is different
        base_currency = symbol[:-3] if len(symbol) > 3 else "BTC"
        quote_currency = symbol[-3:] if len(symbol) > 3 else "USD"
    
    # Process order book data
    # First snapshot - ensure we only take the first two columns (price and volume)
    bids1 = np.array([item[:2] for item in order_book1['bids']], dtype=float) if len(order_book1['bids']) > 0 else np.array([[0, 0]])
    asks1 = np.array([item[:2] for item in order_book1['asks']], dtype=float) if len(order_book1['asks']) > 0 else np.array([[0, 0]])
    
    # Second snapshot - ensure we only take the first two columns (price and volume)
    bids2 = np.array([item[:2] for item in order_book2['bids']], dtype=float) if len(order_book2['bids']) > 0 else np.array([[0, 0]])
    asks2 = np.array([item[:2] for item in order_book2['asks']], dtype=float) if len(order_book2['asks']) > 0 else np.array([[0, 0]])
    
    # Convert to DataFrames for easier manipulation
    bids1_df = pd.DataFrame(bids1, columns=['Price', 'Volume'])
    asks1_df = pd.DataFrame(asks1, columns=['Price', 'Volume'])
    bids2_df = pd.DataFrame(bids2, columns=['Price', 'Volume'])
    asks2_df = pd.DataFrame(asks2, columns=['Price', 'Volume'])
    
    # Convert to float
    bids1_df['Price'] = bids1_df['Price'].astype(float)
    bids1_df['Volume'] = bids1_df['Volume'].astype(float)
    asks1_df['Price'] = asks1_df['Price'].astype(float)
    asks1_df['Volume'] = asks1_df['Volume'].astype(float)
    bids2_df['Price'] = bids2_df['Price'].astype(float)
    bids2_df['Volume'] = bids2_df['Volume'].astype(float)
    asks2_df['Price'] = asks2_df['Price'].astype(float)
    asks2_df['Volume'] = asks2_df['Volume'].astype(float)
    
    # Sort DataFrames - IMPORTANT: bids should be sorted in descending order for proper visualization
    bids1_df = bids1_df.sort_values(by='Price', ascending=False)
    asks1_df = asks1_df.sort_values(by='Price')
    bids2_df = bids2_df.sort_values(by='Price', ascending=False)
    asks2_df = asks2_df.sort_values(by='Price')
    
    # Calculate cumulative volumes
    bids1_df['Cumulative Volume'] = bids1_df['Volume'].cumsum()
    asks1_df['Cumulative Volume'] = asks1_df['Volume'].cumsum()
    bids2_df['Cumulative Volume'] = bids2_df['Volume'].cumsum()
    asks2_df['Cumulative Volume'] = asks2_df['Volume'].cumsum()
    
    # Calculate volume deltas
    bids_delta = pd.merge(bids1_df[['Price', 'Volume']], 
                         bids2_df[['Price', 'Volume']], 
                         on='Price', 
                         how='outer', 
                         suffixes=('_1', '_2'))
    bids_delta.fillna(0, inplace=True)
    bids_delta['Volume_Delta'] = bids_delta['Volume_2'] - bids_delta['Volume_1']
    
    asks_delta = pd.merge(asks1_df[['Price', 'Volume']], 
                         asks2_df[['Price', 'Volume']], 
                         on='Price', 
                         how='outer', 
                         suffixes=('_1', '_2'))
    asks_delta.fillna(0, inplace=True)
    asks_delta['Volume_Delta'] = asks_delta['Volume_2'] - asks_delta['Volume_1']
    
    # Calculate OI delta (change in cumulative volume)
    bids_oi_delta = pd.merge(bids1_df[['Price', 'Cumulative Volume']], 
                            bids2_df[['Price', 'Cumulative Volume']], 
                            on='Price', 
                            how='outer', 
                            suffixes=('_1', '_2'))
    bids_oi_delta = bids_oi_delta.ffill()
    bids_oi_delta.fillna(0, inplace=True)
    bids_oi_delta['OI_Delta'] = bids_oi_delta['Cumulative Volume_2'] - bids_oi_delta['Cumulative Volume_1']
    
    asks_oi_delta = pd.merge(asks1_df[['Price', 'Cumulative Volume']], 
                            asks2_df[['Price', 'Cumulative Volume']], 
                            on='Price', 
                            how='outer', 
                            suffixes=('_1', '_2'))
    asks_oi_delta = asks_oi_delta.ffill()
    asks_oi_delta.fillna(0, inplace=True)
    asks_oi_delta['OI_Delta'] = asks_oi_delta['Cumulative Volume_2'] - asks_oi_delta['Cumulative Volume_1']
    
    # Extract axes
    ax1, ax2, ax3 = axes
    
    # Set background color for all plots
    for ax in axes:
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Add borders to the subplots
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)
    
    # Calculate mid price
    if len(bids1_df) > 0 and len(asks1_df) > 0:
        mid_price = (bids1_df['Price'].iloc[0] + asks1_df['Price'].iloc[0]) / 2
    else:
        mid_price = None
    
    # Plot 1: Order Book Depth (Top Pane) with filled areas - MODIFIED to match XBT/USD image
    # For bids: plot from highest price (left) to mid price (right)
    ax1.plot(bids1_df['Price'], bids1_df['Cumulative Volume'], 
            label='Bids (Buy)', color='green', linewidth=1.5)
    ax1.fill_between(bids1_df['Price'], bids1_df['Cumulative Volume'], 0, color='green', alpha=0.2)
    
    # For asks: plot from lowest price (right) to higher prices
    ax1.plot(asks1_df['Price'], asks1_df['Cumulative Volume'], 
            label='Asks (Sell)', color='red', linewidth=1.5)
    ax1.fill_between(asks1_df['Price'], asks1_df['Cumulative Volume'], 0, color='red', alpha=0.2)
    
    # Add mid price line
    if mid_price:
        ax1.axvline(x=mid_price, color='grey', linestyle='--', alpha=0.7, linewidth=1.0, label='Mid Price')
    
    # Set title for the top pane
    ax1.set_title(f"{symbol} Order Book Analysis", fontsize=12, pad=10)
    
    # Add legend to the top pane
    ax1.legend(loc='upper left', fontsize=9)
    
    # Calculate appropriate bar width based on price range
    all_prices = np.concatenate([
        bids_delta['Price'].values, 
        asks_delta['Price'].values
    ])
    
    if len(all_prices) > 0:
        price_range = np.max(all_prices) - np.min(all_prices) if len(all_prices) > 1 else 1
        bar_width = price_range * 0.0005  # 0.05% of the price range
    else:
        bar_width = 0.01  # Default fallback
    
    # Filter data based on threshold if show_all_changes is False
    if not show_all_changes:
        bid_to_plot = bids_delta[abs(bids_delta['Volume_Delta']) >= threshold]
        ask_to_plot = asks_delta[abs(asks_delta['Volume_Delta']) >= threshold]
    else:
        bid_to_plot = bids_delta
        ask_to_plot = asks_delta
    
    # Plot 2: Volume Delta (Middle Pane) with defined bars
    # Plot bid volume changes (green for positive, red for negative)
    ax2.bar(bid_to_plot[bid_to_plot['Volume_Delta'] > 0]['Price'], 
           bid_to_plot[bid_to_plot['Volume_Delta'] > 0]['Volume_Delta'], 
           width=bar_width, 
           color='green',
           edgecolor='darkgreen', 
           linewidth=0.3)
    
    ax2.bar(bid_to_plot[bid_to_plot['Volume_Delta'] < 0]['Price'], 
           bid_to_plot[bid_to_plot['Volume_Delta'] < 0]['Volume_Delta'], 
           width=bar_width, 
           color='red',
           edgecolor='darkred', 
           linewidth=0.3)
    
    # Plot ask volume changes (green for positive, red for negative)
    ax2.bar(ask_to_plot[ask_to_plot['Volume_Delta'] > 0]['Price'], 
           ask_to_plot[ask_to_plot['Volume_Delta'] > 0]['Volume_Delta'], 
           width=bar_width, 
           color='green',
           edgecolor='darkgreen', 
           linewidth=0.3)
    
    ax2.bar(ask_to_plot[ask_to_plot['Volume_Delta'] < 0]['Price'], 
           ask_to_plot[ask_to_plot['Volume_Delta'] < 0]['Volume_Delta'], 
           width=bar_width, 
           color='red',
           edgecolor='darkred', 
           linewidth=0.3)
    
    ax2.set_ylabel(f'Volume Change ({base_currency})', fontsize=9)
    ax2.set_title(f'Order Book Volume Changes', fontsize=10, pad=5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    
    # Filter data based on threshold if show_all_changes is False
    if not show_all_changes:
        bid_oi_to_plot = bids_oi_delta[abs(bids_oi_delta['OI_Delta']) >= threshold]
        ask_oi_to_plot = asks_oi_delta[abs(asks_oi_delta['OI_Delta']) >= threshold]
    else:
        bid_oi_to_plot = bids_oi_delta
        ask_oi_to_plot = asks_oi_delta
    
    # Plot 3: OI Delta (Bottom Pane) with blue and red colors
    # Plot bid OI changes (blue for positive, darker blue for negative)
    ax3.bar(bid_oi_to_plot[bid_oi_to_plot['OI_Delta'] > 0]['Price'], 
           bid_oi_to_plot[bid_oi_to_plot['OI_Delta'] > 0]['OI_Delta'], 
           width=bar_width, 
           color='blue',
           edgecolor='darkblue', 
           linewidth=0.3)
    
    ax3.bar(bid_oi_to_plot[bid_oi_to_plot['OI_Delta'] < 0]['Price'], 
           bid_oi_to_plot[bid_oi_to_plot['OI_Delta'] < 0]['OI_Delta'], 
           width=bar_width, 
           color='darkblue',
           edgecolor='navy', 
           linewidth=0.3)
    
    # Plot ask OI changes (red for positive, darker red for negative)
    ax3.bar(ask_oi_to_plot[ask_oi_to_plot['OI_Delta'] > 0]['Price'], 
           ask_oi_to_plot[ask_oi_to_plot['OI_Delta'] > 0]['OI_Delta'], 
           width=bar_width, 
           color='red',
           edgecolor='darkred', 
           linewidth=0.3)
    
    ax3.bar(ask_oi_to_plot[ask_oi_to_plot['OI_Delta'] < 0]['Price'], 
           ask_oi_to_plot[ask_oi_to_plot['OI_Delta'] < 0]['OI_Delta'], 
           width=bar_width, 
           color='darkred',
           edgecolor='maroon', 
           linewidth=0.3)
    
    ax3.set_xlabel(f'Price ({quote_currency})', fontsize=9)
    ax3.set_ylabel(f'OI Delta ({base_currency})', fontsize=9)
    ax3.set_title(f'Open Interest Delta', fontsize=10, pad=5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    
    # Set x-axis limits to focus on the mid-price area
    if mid_price:
        price_range = np.max(all_prices) - np.min(all_prices) if len(all_prices) > 1 else 100
        x_min = mid_price - price_range * 0.6
        x_max = mid_price + price_range * 0.6
        
        for ax in axes:
            ax.set_xlim(x_min, x_max)
    
    # Format x-axis to show currency
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.2f}'))
        # Remove grid lines
        ax.grid(False)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)
    
    # Remove x-axis label from bottom plot
    axes[2].set_xlabel("", fontsize=9)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

def convert_symbol_format(symbol, exchange_id):
    """Convert symbol to the appropriate format for the specified exchange.
    
    Args:
        symbol: The symbol to convert
        exchange_id: The exchange ID (coinbase, binance, kraken, etc.)
        
    Returns:
        The symbol in the appropriate format for the exchange
    """
    # Special case for Coinbase with BTC-USD format
    if exchange_id.lower() == 'coinbase':
        if symbol == 'BTC-USD':
            return 'BTC/USD'
        elif symbol == 'ETH-USD':
            return 'ETH/USD'
    
    # Handle common typos first
    # Check if the symbol looks like it might be a typo of a common pair
    if symbol in ['BTCU-SD', 'BTC-UDS', 'BTC-US', 'BTCUSD', 'BTC-USD']:
        if exchange_id.lower() == 'coinbase':
            return 'BTC/USD'
        elif exchange_id.lower() == 'binance':
            return 'BTCUSDT'
        elif exchange_id.lower() == 'kraken':
            return 'BTC/USD'
    
    if symbol in ['ETHU-SD', 'ETH-UDS', 'ETH-US', 'ETHUSD', 'ETH-USD']:
        if exchange_id.lower() == 'coinbase':
            return 'ETH/USD'
        elif exchange_id.lower() == 'binance':
            return 'ETHUSDT'
        elif exchange_id.lower() == 'kraken':
            return 'ETH/USD'
    
    # Handle Coinbase's specific format with dash
    if '-' in symbol and exchange_id.lower() == 'coinbase':
        parts = symbol.split('-')
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1]}"
    
    # Remove any existing separators
    clean_symbol = symbol.replace('-', '').replace('/', '').replace('_', '')
    
    # Common base/quote pairs
    common_pairs = {
        'BTCUSD': {'base': 'BTC', 'quote': 'USD'},
        'ETHUSD': {'base': 'ETH', 'quote': 'USD'},
        'SOLUSD': {'base': 'SOL', 'quote': 'USD'},
        'BTCUSDT': {'base': 'BTC', 'quote': 'USDT'},
        'ETHUSDT': {'base': 'ETH', 'quote': 'USDT'},
        'SOLUSDT': {'base': 'SOL', 'quote': 'USDT'},
    }
    
    # Try to match with common pairs
    base = None
    quote = None
    
    # Check if the symbol matches any common pair
    for pair, components in common_pairs.items():
        if clean_symbol.upper() == pair:
            base = components['base']
            quote = components['quote']
            break
    
    # If not found in common pairs, try to split based on common quote currencies
    if base is None:
        common_quotes = ['USD', 'USDT', 'USDC', 'EUR', 'BTC', 'ETH']
        for quote_currency in common_quotes:
            if clean_symbol.upper().endswith(quote_currency):
                quote = quote_currency
                base = clean_symbol[:-len(quote_currency)]
                break
    
    # If still not identified, use default approach (assume last 3-4 chars are quote)
    if base is None:
        if len(clean_symbol) > 4:
            quote = clean_symbol[-4:] if clean_symbol[-4:] in ['USDT', 'USDC'] else clean_symbol[-3:]
            base = clean_symbol[:-len(quote)]
        else:
            # Can't determine format, return original
            return symbol
    
    # Format according to exchange requirements
    if exchange_id.lower() == 'coinbase':
        return f"{base}/{quote}"
    elif exchange_id.lower() == 'binance':
        return f"{base}{quote}"
    elif exchange_id.lower() == 'kraken':
        return f"{base}/{quote}"
    elif exchange_id.lower() == 'kucoin':
        return f"{base}-{quote}"
    elif exchange_id.lower() == 'bitfinex':
        return f"{base}{quote}"
    else:
        # Default format
        return f"{base}/{quote}"

def debug_order_book(order_book: Dict, symbol: str, exchange_id: str):
    """Debug function to print order book structure and help diagnose issues."""
    print(f"\n=== DEBUG: Order Book for {symbol} on {exchange_id} ===")
    
    if not order_book or 'bids' not in order_book or 'asks' not in order_book:
        print(f"Invalid order book structure: {order_book}")
        return
    
    print(f"Number of bids: {len(order_book['bids'])}")
    print(f"Number of asks: {len(order_book['asks'])}")
    
    if len(order_book['bids']) > 0:
        print(f"First bid structure: {order_book['bids'][0]}")
        print(f"Bid data type: {type(order_book['bids'][0])}")
        print(f"Bid length: {len(order_book['bids'][0])}")
    
    if len(order_book['asks']) > 0:
        print(f"First ask structure: {order_book['asks'][0]}")
        print(f"Ask data type: {type(order_book['asks'][0])}")
        print(f"Ask length: {len(order_book['asks'][0])}")
    
    print("=== End DEBUG ===\n")

# Streamlit app
def main():
    # Add custom CSS for better spacing and section titles
    st.markdown("""
        <style>
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1 {
            margin-bottom: 2rem;
        }
        h2 {
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #FAFAFA;
            font-size: 1.5em;
        }
        .stTable {
            margin-top: 1rem;
        }
        .element-container {
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Quantavius Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        # Exchange selection - Removed and set to Kraken only
        st.markdown("<h3 style='margin-bottom: 0px;'>Exchange</h3>", unsafe_allow_html=True)
        st.info("Using Kraken exchange for all data")
        
        # Set fixed exchange values
        price_exchange = "kraken"
        orderbook_exchange = "kraken"
        
        st.markdown("<h3 style='margin-bottom: 0px; margin-top: 10px;'>Symbol</h3>", unsafe_allow_html=True)
        symbol_input = st.text_input("", value="BTC/USD", key="symbol_input", placeholder="Enter symbol (e.g. BTC/USD)").strip()
        
        # For Kraken, use the normal conversion
        symbol = convert_symbol_format(symbol_input, price_exchange)
        
        # Show the converted symbol if it's different from the input
        if symbol != symbol_input:
            st.info(f"Symbol converted to {symbol} for Kraken")
        
        st.markdown("<h3 style='margin-bottom: 0px; margin-top: 10px;'>Lookback Period</h3>", unsafe_allow_html=True)
        lookback_options = {
            "1 Day (1440 1m bars)": 1440,
            "4 Hours (240 1m bars)": 240
        }
        lookback = st.selectbox("", options=list(lookback_options.keys()), key="lookback_select", index=1)
        lookback_value = lookback_options[lookback]
        
        # Use the same symbol for order book
        ob_symbol = symbol
        
        # Order Book Settings
        st.markdown("<h3 style='margin-bottom: 0px; margin-top: 10px;'>Order Book Settings</h3>", unsafe_allow_html=True)
        orderbook_depth = st.slider("Depth", 
                                   min_value=1, 
                                   max_value=100, 
                                   value=50, 
                                   key="depth_slider")
        
        # Add minimum amplitude threshold for filtering small changes
        min_amplitude = st.slider("Min Change Threshold (%)", 
                                    min_value=0.0, 
                                    max_value=5.0, 
                                    value=1.0, 
                                    step=0.1,
                                    key="min_amplitude_slider")
        
        # Set update interval
        reset_interval_options = {
            "30 seconds": 30,
            "1 minute": 60,
            "5 minutes": 300
        }
        # Default to 30 seconds for more frequent updates
        reset_interval = 30
        
        # Convert percentage to decimal
        threshold = min_amplitude / 100.0
        show_all_changes = (min_amplitude == 0.0)
        
        st.markdown("<h3 style='margin-bottom: 0px; margin-top: 10px;'>Label Settings</h3>", unsafe_allow_html=True)
        amplitude_threshold = st.slider("Amplitude Threshold (BPS)", 
                                     min_value=5, 
                                     max_value=50, 
                                     value=20, 
                                     key="amplitude_slider")
        
        inactive_period = st.slider("Inactive Period (minutes)", 
                                  min_value=1, 
                                  max_value=30, 
                                  value=10, 
                                  key="inactive_slider")
        
        # Add a simple divider
        st.markdown("---")
        
        # Add a collapsible section for advanced info
        with st.expander("Advanced Information"):
            st.markdown("""
            **AmplitudeBasedLabeler Parameters:**
            - **Amplitude Threshold**: Minimum price movement (in basis points) to be considered significant
            - **Inactive Period**: Maximum time where no new high/low is achieved before resetting the trend
            
            Higher amplitude threshold = fewer but stronger signals
            Longer inactive period = trends can persist through more noise
            """)
            
            st.markdown("""
            **Common Kraken Pairs:**
            - BTC/USD
            - ETH/USD
            - XBT/USD
            - SOL/USD
            """)

    # Add auto-refresh functionality if reset_interval > 0
    if reset_interval > 0:
        st.markdown(f"""
        <div style="text-align: center; padding: 5px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 10px;">
            <p style="margin: 0; color: #31333F;">Auto-refresh enabled: Dashboard will update every {reset_interval} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add JavaScript for auto-refresh
        st.markdown(f"""
        <script>
            setTimeout(function(){{
                window.location.reload();
            }}, {reset_interval * 1000});
        </script>
        """, unsafe_allow_html=True)
    
    # Main content
    try:
        momentum_placeholder = st.empty()
        depth_placeholder = st.empty()
        heatmap_placeholder = st.empty()
        last_minute = None
        
        # Store previous order book for delta calculations
        previous_order_book = None
        
        while True:
            current_time = pd.Timestamp.now(tz='US/Eastern')
            current_minute = current_time.floor('1min')
            
            # Initialize exchanges with user-selected options
            exchanges = initialize_exchange(price_exchange, orderbook_exchange)
            price_exchange_client = exchanges['price']
            orderbook_exchange_client = exchanges['orderbook']
            
            with momentum_placeholder.container():
                # Momentum Imbalances Section
                st.markdown("<h2>Momentum Imbalances</h2>", unsafe_allow_html=True)
                st.markdown(f"Fetching data for {symbol} with {lookback_value} 1-minute data points from Kraken.", unsafe_allow_html=True)
                
                # Display current AmplitudeBasedLabeler settings
                st.markdown(f"""
                <div style='background-color: #262730; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <p style='margin: 0; color: #FAFAFA;'>
                        <strong>Current Settings:</strong> Amplitude Threshold = {amplitude_threshold} BPS, 
                        Inactive Period = {inactive_period} minutes
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    with st.spinner("Fetching momentum data..."):
                        try:
                            # Fetch market data
                            df = fetch_market_data(price_exchange_client, symbol, '1m', limit=lookback_value, lookback_hours=24)
                            
                            if last_minute is None or current_minute > last_minute:
                                # Update CONFIG with the user-selected values
                                CONFIG['analysis']['amplitude_threshold'] = amplitude_threshold
                                CONFIG['analysis']['inactive_period'] = inactive_period
                                df = calculate_metrics(df, price_exchange_client, symbol, orderbook_depth)
                                last_minute = current_minute
                            
                            # Create momentum plot
                            fig_momentum = plt.figure(figsize=CONFIG['visualization']['figure_size'])
                            gs_momentum = gridspec.GridSpec(1, 2, width_ratios=[8, 1])
                            gs_momentum.update(wspace=0.02)
                            
                            ax_price = fig_momentum.add_subplot(gs_momentum[0])
                            ax_vol = fig_momentum.add_subplot(gs_momentum[1], sharey=ax_price)
                            
                            # Safely convert to pydatetime
                            try:
                                # Fix for FutureWarning about DatetimeProperties.to_pydatetime
                                x_values = date2num(np.array(df['timestamp'].to_numpy()))
                            except (AttributeError, KeyError) as e:
                                st.warning(f"Error converting timestamps: {str(e)}. Using fallback.")
                                # Try alternative conversion methods
                                try:
                                    # Try converting with numpy array directly
                                    x_values = date2num(df['timestamp'].to_numpy())
                                except:
                                    try:
                                        # Try manual conversion
                                        x_values = date2num([pd.Timestamp(x) for x in df['timestamp']])
                                    except:
                                        # Last resort: use numeric range
                                        x_values = np.arange(len(df))
                            
                            plot_price_action(ax_price, df, x_values)
                            max_width = plot_volume_profile(ax_vol, df)
                            style_plot(fig_momentum, ax_price, ax_vol, df, x_values, max_width)
                            
                            st.pyplot(fig_momentum)
                            plt.close(fig_momentum)
                            
                            # Display price data table
                            price_table = create_price_table(df)
                            st.table(price_table)
                            
                            # Display last data timestamp
                            st.markdown(f"<p style='color: #666666; font-size: 0.8em;'>Last data timestamp (EST): {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}</p>", unsafe_allow_html=True)
                        except Exception as e:
                            error_msg = str(e)
                            if "Empty OHLCV data received" in error_msg:
                                st.error(f"No data available for {symbol} on {price_exchange.capitalize()}.")
                                
                                # Provide specific guidance based on the exchange
                                if price_exchange == 'coinbase':
                                    st.warning(f"Coinbase requires symbols in the format 'BTC-USD'. You entered: {symbol}")
                                    st.info("Common Coinbase pairs: BTC-USD, ETH-USD, SOL-USD")
                                elif price_exchange == 'binance':
                                    st.warning(f"Binance typically uses symbols like 'BTC/USDT' or 'BTCUSDT'. You entered: {symbol}")
                                    st.info("Common Binance pairs: BTC/USDT, ETH/USDT, SOL/USDT")
                                elif price_exchange == 'kraken':
                                    st.warning(f"Kraken typically uses symbols like 'BTC/USD'. You entered: {symbol}")
                                    st.info("Common Kraken pairs: BTC/USD, ETH/USD, XBT/USD")
                                else:
                                    st.info("Different exchanges use different symbol formats. For example:\n- Coinbase: BTC-USD\n- Binance: BTCUSDT\n- Kraken: BTC/USD")
                            elif "symbol" in error_msg.lower() and "not found" in error_msg.lower():
                                st.error(f"Symbol '{symbol}' not found on {price_exchange.capitalize()}.")
                                
                                # Provide specific guidance based on the exchange
                                if price_exchange == 'coinbase':
                                    st.warning("Coinbase requires symbols in the format 'BTC-USD'")
                                    st.info("Common Coinbase pairs: BTC-USD, ETH-USD, SOL-USD")
                                elif price_exchange == 'binance':
                                    st.warning("Binance typically uses symbols like 'BTC/USDT' or 'BTCUSDT'")
                                    st.info("Common Binance pairs: BTC/USDT, ETH/USDT, SOL/USDT")
                                elif price_exchange == 'kraken':
                                    st.warning("Kraken typically uses symbols like 'BTC/USD'")
                                    st.info("Common Kraken pairs: BTC/USD, ETH/USD, XBT/USD")
                                else:
                                    st.info("Different exchanges use different symbol formats. For example:\n- Coinbase: BTC-USD\n- Binance: BTCUSDT\n- Kraken: BTC/USD")
                            else:
                                st.error(f"Error fetching market data: {error_msg}")
                            
                            # Continue with other sections even if price data fails
                            time.sleep(5)  # Wait before retrying
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    time.sleep(5)  # Wait before retrying
            
            with depth_placeholder.container():
                # Order Book Depth Section
                st.markdown(f"<h2>📊 Order Book Depth (Live) - Kraken</h2>", unsafe_allow_html=True)
                
                try:
                    with st.spinner("Fetching order book data..."):
                        # Create market depth plot with more square aspect ratio
                        fig_depth = plt.figure(figsize=(10, 8))  # Square dimensions
                        ax_depth = fig_depth.add_subplot(111)
                        
                        # Fetch current order book from Kraken
                        current_order_book = fetch_order_book(orderbook_exchange_client, ob_symbol, orderbook_depth)
                        
                        # Plot market depth
                        plot_market_depth(fig_depth, ax_depth, current_order_book, ob_symbol)
                        
                        # Display the plot
                        st.pyplot(fig_depth)
                        plt.close(fig_depth)
                        
                        # Display information about Kraken's symbol format
                        st.info(f"Kraken typically uses symbols like 'BTC/USD'. Common pairs: BTC/USD, ETH/USD, XBT/USD")
                except Exception as e:
                    st.error(f"Error fetching order book data: {str(e)}")
                    st.info("Please check that you're using the correct symbol format for Kraken.")
            
            with heatmap_placeholder.container():
                # Order Book Heatmap Section
                st.markdown(f"<h2>📈 Order Book Heatmap (Live) - Kraken</h2>", unsafe_allow_html=True)
                
                try:
                    with st.spinner("Fetching order book heatmap data..."):
                        # Fetch current order book from Kraken
                        current_order_book = fetch_order_book(orderbook_exchange_client, ob_symbol, orderbook_depth)
                        
                        # Debug the order book structure
                        debug_order_book(current_order_book, ob_symbol, orderbook_exchange)
                        
                        # Create heatmap plot
                        if previous_order_book is None:
                            # First run, just store the order book
                            previous_order_book = current_order_book
                            st.info("Initializing order book data. Heatmap will appear after the next update.")
                        else:
                            # Create heatmap comparing previous and current order books
                            fig_heatmap = plt.figure(figsize=(12, 8))
                            gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
                            ax1 = fig_heatmap.add_subplot(gs[0])
                            ax2 = fig_heatmap.add_subplot(gs[1])
                            ax3 = fig_heatmap.add_subplot(gs[2])
                            plot_order_book_heatmap_impl(fig_heatmap, [ax1, ax2, ax3], 
                                                       previous_order_book, current_order_book, 
                                                       ob_symbol, show_all_changes, threshold)
                            
                            # Display the plot
                            st.pyplot(fig_heatmap)
                            plt.close(fig_heatmap)
                            
                            # Update previous order book for next comparison
                            previous_order_book = current_order_book
                except Exception as e:
                    st.error(f"Error fetching order book data: {str(e)}")
                    st.info("Please check that you're using the correct symbol format for Kraken.")
            
            # Wait before refreshing
            time.sleep(reset_interval)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Keep the original plot_market_depth function
def plot_market_depth(fig, ax, order_book: Dict, symbol: str):
    """Create a market depth visualization with white theme matching the reference design."""
    if not order_book or 'bids' not in order_book or 'asks' not in order_book:
        return
    
    # Process bids and asks
    bids = np.array(order_book['bids'], dtype=float)
    asks = np.array(order_book['asks'], dtype=float)
    
    if len(bids) == 0 or len(asks) == 0:
        return
    
    # Process bids (keep in descending order for proper visualization)
    bid_prices = bids[:, 0]  # First column is price
    bid_volumes = bids[:, 1]  # Second column is volume
    
    # Process asks (already in ascending order)
    ask_prices = asks[:, 0]
    ask_volumes = asks[:, 1]
    
    # Calculate mid price
    mid_price = (bid_prices[0] + ask_prices[0]) / 2  # Using highest bid and lowest ask
    
    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Clear previous plot
    ax.clear()
    
    # Create DataFrames for easier manipulation
    bids_df = pd.DataFrame({'Price': bid_prices, 'Volume': bid_volumes})
    asks_df = pd.DataFrame({'Price': ask_prices, 'Volume': ask_volumes})
    
    # Sort bids in descending order (high to low price)
    bids_df = bids_df.sort_values(by='Price', ascending=False)
    
    # Sort asks in ascending order (low to high price)
    asks_df = asks_df.sort_values(by='Price')
    
    # Calculate cumulative volumes
    bids_df['Cumulative Volume'] = bids_df['Volume'].cumsum()
    asks_df['Cumulative Volume'] = asks_df['Volume'].cumsum()
    
    # Plot bids and asks with step and fill - make them face each other like in the XBT/USD plot
    # For bids: plot from right to left (descending prices)
    ax.plot(bids_df['Price'], bids_df['Cumulative Volume'], '-', color='green', linewidth=2, label='Bids (Buy)')
    ax.fill_between(bids_df['Price'], bids_df['Cumulative Volume'], 0, color='green', alpha=0.2)
    
    # For asks: plot from left to right (ascending prices)
    ax.plot(asks_df['Price'], asks_df['Cumulative Volume'], '-', color='red', linewidth=2, label='Asks (Sell)')
    ax.fill_between(asks_df['Price'], asks_df['Cumulative Volume'], 0, color='red', alpha=0.2)
    
    # Add mid price line
    ax.axvline(x=mid_price, color='grey', linestyle='--', alpha=0.7, linewidth=1.0, label='Mid Price')
    
    # Add legend with better positioning
    ax.legend(loc='upper left', fontsize=9)
    
    # Set title and labels
    if '/' in symbol:
        base_currency, quote_currency = symbol.split('/')
    else:
        base_currency = symbol[:3]
        quote_currency = symbol[3:]
    
    ax.set_title(f"{symbol} Order Book Depth", fontsize=12)
    # Remove price label from x-axis
    ax.set_xlabel("", fontsize=10)
    ax.set_ylabel(f"Cumulative Volume ({base_currency})", fontsize=10)
    
    # Format x-axis with currency
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
    
    # Remove grid
    ax.grid(False)
    
    # Set x-axis limits to focus on the mid-price area - zoom out more to push data to edges
    price_range = max(np.max(bid_prices) - np.min(bid_prices), np.max(ask_prices) - np.min(ask_prices))
    x_min = mid_price - price_range * 1.2  # Increased from 0.6 to 1.2 to zoom out more
    x_max = mid_price + price_range * 1.2  # Increased from 0.6 to 1.2 to zoom out more
    ax.set_xlim(x_min, x_max)
    
    # Close the figure to prevent memory leaks
    plt.close('all')

if __name__ == "__main__":
    main() 