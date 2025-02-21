import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter, date2num, AutoDateLocator
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tseries_patterns import AmplitudeBasedLabeler
import time
import requests
import math
from typing import Dict, List, Optional, Tuple, Union
import hmac
import base64
import hashlib
import urllib.parse

class BinanceClient:
    """Binance Futures REST API client with rate limiting and efficient data fetching."""
    
    def __init__(self):
        self.base_url = "https://fapi.binance.com"  # Changed to futures API endpoint
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms between requests (Binance has higher rate limits)
        
    def _wait_for_rate_limit(self):
        """Implement simple rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def get_klines_data(self, symbol: str, interval: str = "1m", limit: int = 1000, start_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch klines (OHLCV) data from Binance Futures with efficient pagination.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of records to fetch (max 1500 for futures)
            start_time: Start time in milliseconds
        
        Returns:
            DataFrame with OHLCV data
        """
        # Convert symbol format and add perpetual suffix if needed
        symbol = self._convert_symbol_format(symbol)
        
        endpoint = "/fapi/v1/klines"  # Changed to futures klines endpoint
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500)  # Futures API allows up to 1500 candles
        }
        if start_time is not None:
            params["startTime"] = start_time
            
        self._wait_for_rate_limit()
        response = self.session.get(f"{self.base_url}{endpoint}", params=params)
        
        if response.status_code != 200:
            raise Exception(f"Error fetching klines data: {response.text}")
            
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])
        
        # Convert types and set index
        df["time"] = pd.to_datetime(df["time"].astype(float), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df.set_index("time", inplace=True)
        
        # Keep only necessary columns
        df = df[["open", "high", "low", "close", "volume"]]
        
        return df
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format to Binance Futures format."""
        if "/" in symbol:
            base, quote = symbol.split("/")
            # For futures, we only need to handle USDT pairs
            if quote == "USDT":
                return f"{base}{quote}"
            else:
                raise ValueError(f"Unsupported quote currency for futures: {quote}. Use USDT pairs.")
        return symbol

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
        'exchange': 'binanceusdm',  # Changed to Binance USD-M futures
    },
    'visualization': {
        'figure_size': (16, 8),  # Increased figure size for better visibility
        'price_color': '#c0c0c0',  # Updated to silver color
        'vwap_above_color': '#3399ff',  # Blue for VWAP when price is above
        'vwap_below_color': '#ff4d4d',  # Red for VWAP when price is below
        'vwma_color': '#90EE90',  # Light green for VWMA
        'up_color': '#3399ff',
        'down_color': '#ff4d4d',
        'volume_colors': {
            'high': '#3399ff',  # Match up_color for bullish volume
            'medium': '#cccccc',  # Keep neutral color
            'low': '#ff4d4d'  # Match down_color for bearish volume
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

def initialize_exchange():
    """Initialize the cryptocurrency exchange connection."""
    # Initialize both Binance Futures client and CCXT (for orderbook)
    binance_client = BinanceClient()
    exchange_class = getattr(ccxt, CONFIG['trading']['exchange'])
    ccxt_client = exchange_class({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # Use futures markets
            'fetchOHLCVWarning': False,
            'watchOrderBook': {
                'limit': 5000  # Configure maximum order book depth
            },
            'watchOrderBookLimit': 5000
        }
    })
    return binance_client, ccxt_client

def fetch_market_data(clients, symbol, lookback):
    """Fetch market data from Binance's REST API with proper pagination."""
    try:
        binance_client, _ = clients
        current_time = pd.Timestamp.now(tz='UTC')
        
        # Calculate the start time in milliseconds
        start_time = int((current_time - pd.Timedelta(minutes=lookback+10)).timestamp() * 1000)
        
        # Initialize an empty list to store all dataframes
        all_data = []
        remaining_bars = lookback + 10  # Add buffer
        current_start = start_time
        
        # Fetch data with retry mechanism and proper pagination
        max_retries = 3
        retry_delay = 2  # seconds
        
        while remaining_bars > 0:
            for attempt in range(max_retries):
                try:
                    # Calculate how many bars to fetch in this iteration (max 1000)
                    batch_size = min(1000, remaining_bars)
                    
                    # Fetch batch of data
                    df = binance_client.get_klines_data(
                        symbol=symbol,
                        interval="1m",
                        limit=batch_size,
                        start_time=current_start
                    )
                    
                    if df.empty:
                        raise Exception("Empty OHLCV data received")
                    
                    all_data.append(df)
                    
                    # Update remaining bars and start time for next batch
                    remaining_bars -= len(df)
                    if remaining_bars > 0:
                        current_start = int(df.index[-1].timestamp() * 1000) + 60000  # Add 1 minute in milliseconds
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise Exception(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        
        # Combine all data
        if not all_data:
            raise Exception("No data was fetched")
            
        df = pd.concat(all_data)
        
        # Convert timezone and sort
        df.index = df.index.tz_localize('UTC').tz_convert('America/Toronto')
        df = df.sort_index()
        
        # Take the required number of bars
        df = df.tail(lookback)
        
        if len(df) < lookback * 0.9:  # If we got less than 90% of requested data
            raise Exception(f"Insufficient data: got {len(df)} bars, expected {lookback}")
        
        return df
        
    except Exception as e:
        print(f"Error in fetch_market_data: {str(e)}")
        raise e

def calculate_vwma(df, period):
    """Calculate Volume Weighted Moving Average"""
    df['vwma'] = (df['close'] * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return df

def calculate_metrics(df, ccxt_client, symbol, orderbook_depth):
    """Calculate various market metrics including VWAP and order book imbalance."""
    try:
        # Calculate VWAP - reset at the start of each session
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_vol_price'] = (df['close'] * df['volume']).cumsum()
        df['vwap'] = df['cum_vol_price'] / df['cum_vol']
        
        # Calculate VWMA
        df = calculate_vwma(df, CONFIG['analysis']['vwma_period'])
        
        # Calculate order book metrics using CCXT
        order_book = ccxt_client.fetch_order_book(symbol, limit=orderbook_depth)
        bids_volume = sum(bid[1] for bid in order_book['bids'][:orderbook_depth])
        asks_volume = sum(ask[1] for ask in order_book['asks'][:orderbook_depth])
        
        df['orderbook_imbalance'] = bids_volume / (bids_volume + asks_volume)
        
        # Calculate bubble sizes
        df['bubble_size'] = df['volume'] * df['orderbook_imbalance']
        df['bubble_size'] = df['bubble_size'] / df['bubble_size'].max()
        
        # Reset index to make 'time' a column for AmplitudeBasedLabeler
        df = df.reset_index()
        
        # Create AmplitudeBasedLabeler instance
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
        df.set_index('time', inplace=True)
        
        # Clean up intermediate columns
        df = df.drop(['cum_vol', 'cum_vol_price'], axis=1)
        
        return df
    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        raise e

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
    
    # Plot bubbles
    up_mask = df['momentum_label'] > 0
    down_mask = df['momentum_label'] <= 0
    base_size = CONFIG['visualization']['base_bubble_size']
    
    ax.scatter(x_values[up_mask], df.loc[up_mask, 'close'],
              c=CONFIG['visualization']['up_color'],
              s=df.loc[up_mask, 'bubble_size'] * base_size,
              alpha=0.8, zorder=3, edgecolors='none')  # Increased alpha and zorder
    ax.scatter(x_values[down_mask], df.loc[down_mask, 'close'],
              c=CONFIG['visualization']['down_color'],
              s=df.loc[down_mask, 'bubble_size'] * base_size,
              alpha=0.8, zorder=3, edgecolors='none')  # Increased alpha and zorder

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
    ax_price.set_ylim(df['close'].min() - y_padding, df['close'].max() + y_padding)
    ax_vol.set_xlim(-max_width * 1.1, 0)  # Added 10% padding
    
    # Final adjustments
    plt.tight_layout()

def create_price_table(df):
    """Create a formatted price data table."""
    # Take the 5 most recent bars
    sampled_df = df.sort_index().tail(5)
    
    # Format the data
    table_data = {
        'Time/Price': [f"{t.strftime('%H:%M')} - ${p:,.2f}"  # Added thousands separator and removed seconds
                      for t, p in zip(sampled_df.index, sampled_df['close'])],
        'Momentum': sampled_df['momentum_label'].map({1: '↑', 0: '•', -1: '↓'}),  # Changed neutral symbol to dot
        'OB Imbalance': (sampled_df['orderbook_imbalance'] * 100).round(1).astype(str) + '%',
        'Volume': sampled_df['volume'].round(2).map('{:,.2f}'.format),  # Added thousands separator
        'VWAP Distance': ((sampled_df['close'] / sampled_df['vwap'] - 1) * 100).round(2).astype(str) + '%'
    }
    
    # Create DataFrame and reverse the order so most recent is at top
    return pd.DataFrame(table_data).iloc[::-1]

def fetch_order_book(binance_client: BinanceClient, symbol: str, limit: int = 1000) -> Dict:
    """Fetch order book data from Binance Futures API."""
    symbol = binance_client._convert_symbol_format(symbol)
    endpoint = "/fapi/v1/depth"
    params = {
        "symbol": symbol,
        "limit": min(limit, 5000)  # Ensure we don't exceed Binance's maximum limit
    }
    binance_client._wait_for_rate_limit()
    response = binance_client.session.get(f"{binance_client.base_url}{endpoint}", params=params)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching order book: {response.text}")
    
    return response.json()

def plot_market_depth(fig, ax, order_book: Dict, symbol: str):
    """Create a market depth visualization with white theme matching the reference design."""
    if not order_book or 'bids' not in order_book or 'asks' not in order_book:
        return
    
    # Process bids and asks
    bids = np.array(order_book['bids'], dtype=float)
    asks = np.array(order_book['asks'], dtype=float)
    
    if len(bids) == 0 or len(asks) == 0:
        return
    
    # Process bids (reverse order to get ascending)
    bid_prices = bids[:, 0][::-1]  # Reverse to get ascending order
    bid_volumes = bids[:, 1][::-1]
    cumulative_bid_volumes = np.cumsum(bid_volumes[::-1])[::-1]  # Compute properly ordered cumulative volume
    
    # Process asks (already in ascending order)
    ask_prices = asks[:, 0]
    ask_volumes = asks[:, 1]
    cumulative_ask_volumes = np.cumsum(ask_volumes)
    
    # Calculate mid price
    mid_price = (bid_prices[-1] + ask_prices[0]) / 2  # Using highest bid and lowest ask
    
    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Plot bids and asks with step and fill
    ax.step(bid_prices, cumulative_bid_volumes, where='post', label='Bids', color='green', linewidth=1)
    ax.fill_between(bid_prices, cumulative_bid_volumes, step='post', alpha=0.3, color='green')
    
    ax.step(ask_prices, cumulative_ask_volumes, where='post', label='Asks', color='red', linewidth=1)
    ax.fill_between(ask_prices, cumulative_ask_volumes, step='post', alpha=0.3, color='red')
    
    # Add mid price line
    ax.axvline(x=mid_price, color='grey', linestyle=':', alpha=0.3, label='Mid Price')
    
    # Set title and labels
    ax.set_title(f'Order Book Depth for {symbol}', pad=20, fontsize=12, color='black')
    ax.set_xlabel('Price', color='black', labelpad=10)
    ax.set_ylabel('Cumulative Volume', color='black', labelpad=10)
    
    # Remove grid
    ax.grid(False)
    
    # Adjust figure size to accommodate legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    
    # Place legend outside the plot on the right
    ax.legend(frameon=True, 
             facecolor='white', 
             edgecolor='none', 
             fontsize=8,
             loc='center left', 
             bbox_to_anchor=(1.01, 0.5))
    
    # Set x-axis limits and format ticks
    price_range = max(ask_prices[-1] - bid_prices[0], 1e-8)
    x_min = bid_prices[0] - price_range * 0.1
    x_max = ask_prices[-1] + price_range * 0.1
    ax.set_xlim(x_min, x_max)
    
    # Format x-axis ticks with dollar signs and proper spacing
    tick_positions = np.linspace(x_min, x_max, 10)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'${x:,.2f}' for x in tick_positions], rotation=0)
    
    # Format y-axis with comma-separated numbers
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    # Adjust tick colors and sizes
    ax.tick_params(axis='both', colors='black', labelsize=8)
    
    # Tight layout with more right padding for legend
    fig.tight_layout(pad=1.1)

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

    # Title with custom styling
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Quantavius Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Enter Ticker Symbol for Momentum/Volume")
        symbol = st.text_input("Symbol (USDT Perpetual)", value="BTC/USDT", key="symbol_input").strip()
        
        st.subheader("Select Lookback Period")
        lookback_options = {
            "1 Day (1440 1m bars)": 1440,
            "4 Hours (240 1m bars)": 240
        }
        lookback = st.selectbox("Lookback", options=list(lookback_options.keys()), key="lookback_select")
        lookback_value = lookback_options[lookback]
        
        st.subheader("Enter Ticker Symbol for Order Book Depth")
        ob_symbol = st.text_input("Symbol (USDT Perpetual)", value=symbol, key="ob_symbol_input").strip()
        
        st.subheader("Select Order Book Depth")
        normalized_depth = st.slider("Depth", 
                                   min_value=1, 
                                   max_value=100, 
                                   value=50, 
                                   key="depth_slider", 
                                   help="Order book depth (1-100 scale, where 100 = 1000 orders, 50 = 500 orders, etc.)")
        
        # Convert normalized depth to actual depth (1-100 maps to 1-1000)
        orderbook_depth = int(normalized_depth * 10)
        
        st.subheader("Label Settings")
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
        
        # Add note about perpetual futures
        st.markdown("""
        ---
        **Note:** This dashboard uses Binance USDT-M Perpetual Futures markets.
        All pairs must be USDT-margined perpetual contracts.
        """)
        
        st.subheader("Update Frequency")
        st.text("Data updates every 30 seconds")

    # Main content
    try:
        momentum_placeholder = st.empty()
        depth_placeholder = st.empty()
        last_minute = None
        
        while True:
            current_time = pd.Timestamp.now(tz='US/Eastern')
            current_minute = current_time.floor('1min')
            
            with momentum_placeholder.container():
                # Momentum Imbalances Section
                st.markdown("<h2>Momentum Imbalances</h2>", unsafe_allow_html=True)
                st.markdown(f"Fetching data for {symbol} with {lookback_value} 1-minute data points.", unsafe_allow_html=True)
                
                with st.spinner("Fetching momentum data..."):
                    clients = initialize_exchange()
                    df = fetch_market_data(clients, symbol, lookback_value)
                    
                    if last_minute is None or current_minute > last_minute:
                        CONFIG['analysis']['amplitude_threshold'] = amplitude_threshold
                        CONFIG['analysis']['inactive_period'] = inactive_period
                        df = calculate_metrics(df, clients[1], symbol, orderbook_depth)
                        last_minute = current_minute
                    
                    # Create momentum plot
                    fig_momentum = plt.figure(figsize=CONFIG['visualization']['figure_size'])
                    gs_momentum = gridspec.GridSpec(1, 2, width_ratios=[8, 1])
                    gs_momentum.update(wspace=0.02)
                    
                    ax_price = fig_momentum.add_subplot(gs_momentum[0])
                    ax_vol = fig_momentum.add_subplot(gs_momentum[1], sharey=ax_price)
                    
                    x_values = date2num(df.index.to_pydatetime())
                    plot_price_action(ax_price, df, x_values)
                    max_width = plot_volume_profile(ax_vol, df)
                    style_plot(fig_momentum, ax_price, ax_vol, df, x_values, max_width)
                    
                    st.pyplot(fig_momentum)
                    plt.close(fig_momentum)
                    
                    # Display price data table
                    price_table = create_price_table(df)
                    st.table(price_table)
                    
                    # Display last data timestamp
                    st.markdown(f"<p style='color: #666666; font-size: 0.8em;'>Last data timestamp (EST): {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}</p>", unsafe_allow_html=True)
            
            with depth_placeholder.container():
                # Market Depth Section
                st.markdown("<h2>📊 Order Book Depth (Live)</h2>", unsafe_allow_html=True)
                
                with st.spinner("Fetching order book data..."):
                    # Create market depth plot with more square aspect ratio
                    fig_depth = plt.figure(figsize=(10, 8))  # Changed to more square dimensions
                    ax_depth = fig_depth.add_subplot(111)
                    
                    order_book = fetch_order_book(clients[0], ob_symbol, orderbook_depth)
                    plot_market_depth(fig_depth, ax_depth, order_book, ob_symbol)
                    
                    st.pyplot(fig_depth)
                    plt.close(fig_depth)
            
            # Wait for 30 seconds before the next update
            time.sleep(30)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 