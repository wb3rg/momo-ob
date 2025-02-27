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
from typing import Dict, List, Optional, Tuple, Union

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

def initialize_exchange():
    """Initialize the cryptocurrency exchange connection."""
    exchange = ccxt.coinbase({
        'enableRateLimit': True,
        'timeout': 30000,
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        },
        'options': {
            'fetchOHLCVWarning': False,
            'createMarketBuyOrderRequiresPrice': False
        }
    })
    
    # Configure testnet if needed
    # exchange.set_sandbox_mode(True)
    
    return exchange

def fetch_market_data(exchange, symbol, lookback):
    """Fetch market data using CCXT with proper pagination."""
    try:
        # Calculate the start time
        current_time = pd.Timestamp.now(tz='UTC')
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
                    # Calculate how many bars to fetch in this iteration (max 300 for Coinbase)
                    batch_size = min(300, remaining_bars)
                    
                    # Fetch batch of data
                    ohlcv = exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe='1m',
                        since=current_start,
                        limit=batch_size
                    )
                    
                    if not ohlcv:
                        raise Exception("Empty OHLCV data received")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    all_data.append(df)
                    
                    # Update remaining bars and start time for next batch
                    remaining_bars -= len(df)
                    if remaining_bars > 0:
                        current_start = int(df.index[-1].timestamp() * 1000) + 60000  # Add 1 minute in milliseconds
                        time.sleep(0.1)  # Add small delay between requests
                    
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
        
        # Reset index to make timestamp a column for AmplitudeBasedLabeler
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
        df.set_index('timestamp', inplace=True)
        
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

def fetch_order_book(exchange: ccxt.Exchange, symbol: str, limit: int = 1000) -> Dict:
    """Fetch order book data using CCXT with retries and proper error handling."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            order_book = exchange.fetch_order_book(
                symbol,
                limit=limit
            )
            return order_book
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise Exception(f"Error fetching order book: {str(e)}")
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            continue
    
    return {'bids': [], 'asks': []}

def plot_order_book_heatmap(fig, axes, order_book1: Dict, order_book2: Dict, symbol: str):
    """Create an enhanced order book visualization with heatmap, volume delta, and OI delta."""
    if not order_book1 or not order_book2 or 'bids' not in order_book1 or 'asks' not in order_book1:
        return
    
    # Extract the base and quote currency from the symbol
    if '-' in symbol:
        base_currency, quote_currency = symbol.split('-')
    else:
        # Default fallback if symbol format is different
        base_currency = symbol[:-3] if len(symbol) > 3 else "BTC"
        quote_currency = symbol[-3:] if len(symbol) > 3 else "USD"
    
    # Process order book data
    # First snapshot
    bids1 = np.array(order_book1['bids'], dtype=float) if len(order_book1['bids']) > 0 else np.array([[0, 0]])
    asks1 = np.array(order_book1['asks'], dtype=float) if len(order_book1['asks']) > 0 else np.array([[0, 0]])
    
    # Second snapshot
    bids2 = np.array(order_book2['bids'], dtype=float) if len(order_book2['bids']) > 0 else np.array([[0, 0]])
    asks2 = np.array(order_book2['asks'], dtype=float) if len(order_book2['asks']) > 0 else np.array([[0, 0]])
    
    # Convert to DataFrames for easier manipulation
    bids1_df = pd.DataFrame(bids1, columns=['Price', 'Volume'])
    asks1_df = pd.DataFrame(asks1, columns=['Price', 'Volume'])
    bids2_df = pd.DataFrame(bids2, columns=['Price', 'Volume'])
    asks2_df = pd.DataFrame(asks2, columns=['Price', 'Volume'])
    
    # Sort DataFrames
    bids1_df = bids1_df.sort_values(by='Price')
    asks1_df = asks1_df.sort_values(by='Price')
    bids2_df = bids2_df.sort_values(by='Price')
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
    bids_oi_delta.fillna(method='ffill', inplace=True)
    bids_oi_delta.fillna(0, inplace=True)
    bids_oi_delta['OI_Delta'] = bids_oi_delta['Cumulative Volume_2'] - bids_oi_delta['Cumulative Volume_1']
    
    asks_oi_delta = pd.merge(asks1_df[['Price', 'Cumulative Volume']], 
                            asks2_df[['Price', 'Cumulative Volume']], 
                            on='Price', 
                            how='outer', 
                            suffixes=('_1', '_2'))
    asks_oi_delta.fillna(method='ffill', inplace=True)
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
    
    # Plot 1: Order Book Depth (Top Pane)
    ax1.plot(bids1_df['Price'], bids1_df['Cumulative Volume'], 
            label='Bids (Buy)', color='green', linewidth=1.5)
    ax1.plot(asks1_df['Price'], asks1_df['Cumulative Volume'], 
            label='Asks (Sell)', color='red', linewidth=1.5)
    ax1.fill_between(bids1_df['Price'], bids1_df['Cumulative Volume'], color='green', alpha=0.2)
    ax1.fill_between(asks1_df['Price'], asks1_df['Cumulative Volume'], color='red', alpha=0.2)
    
    # Calculate mid price
    if len(bids1_df) > 0 and len(asks1_df) > 0:
        mid_price = (bids1_df['Price'].max() + asks1_df['Price'].min()) / 2
        ax1.axvline(x=mid_price, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
    
    ax1.set_ylabel(f'Cumulative Volume ({base_currency})', fontsize=9)
    ax1.set_title(f'{symbol} Order Book Depth', fontsize=10, pad=5)
    ax1.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False)
    
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
    
    # Plot 2: Volume Delta (Middle Pane)
    # Filter out very small changes to reduce noise
    threshold = 0.01
    bid_significant = bids_delta[abs(bids_delta['Volume_Delta']) > threshold]
    ask_significant = asks_delta[abs(asks_delta['Volume_Delta']) > threshold]
    
    # Plot bid volume changes (green for positive, red for negative)
    if len(bid_significant) > 0:
        positive_bids = bid_significant[bid_significant['Volume_Delta'] > 0]
        negative_bids = bid_significant[bid_significant['Volume_Delta'] < 0]
        
        if len(positive_bids) > 0:
            ax2.bar(positive_bids['Price'], 
                   positive_bids['Volume_Delta'], 
                   width=bar_width, 
                   color='green',
                   edgecolor='darkgreen', 
                   linewidth=0.3,
                   label='Volume Increase')
        
        if len(negative_bids) > 0:
            ax2.bar(negative_bids['Price'], 
                   negative_bids['Volume_Delta'], 
                   width=bar_width, 
                   color='red',
                   edgecolor='darkred', 
                   linewidth=0.3,
                   label='Volume Decrease')
    
    # Plot ask volume changes (green for positive, red for negative)
    if len(ask_significant) > 0:
        positive_asks = ask_significant[ask_significant['Volume_Delta'] > 0]
        negative_asks = ask_significant[ask_significant['Volume_Delta'] < 0]
        
        if len(positive_asks) > 0:
            ax2.bar(positive_asks['Price'], 
                   positive_asks['Volume_Delta'], 
                   width=bar_width, 
                   color='green',
                   edgecolor='darkgreen', 
                   linewidth=0.3)
        
        if len(negative_asks) > 0:
            ax2.bar(negative_asks['Price'], 
                   negative_asks['Volume_Delta'], 
                   width=bar_width, 
                   color='red',
                   edgecolor='darkred', 
                   linewidth=0.3)
    
    ax2.set_ylabel(f'Volume Change ({base_currency})', fontsize=9)
    ax2.set_title(f'{symbol} Order Book Volume Changes', fontsize=10, pad=5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    
    # Plot 3: OI Delta (Bottom Pane)
    # Filter out very small changes to reduce noise
    bid_oi_significant = bids_oi_delta[abs(bids_oi_delta['OI_Delta']) > threshold]
    ask_oi_significant = asks_oi_delta[abs(asks_oi_delta['OI_Delta']) > threshold]
    
    # Plot bid OI changes (blue for positive, darker blue for negative)
    if len(bid_oi_significant) > 0:
        positive_bid_oi = bid_oi_significant[bid_oi_significant['OI_Delta'] > 0]
        negative_bid_oi = bid_oi_significant[bid_oi_significant['OI_Delta'] < 0]
        
        if len(positive_bid_oi) > 0:
            ax3.bar(positive_bid_oi['Price'], 
                   positive_bid_oi['OI_Delta'], 
                   width=bar_width, 
                   color='royalblue',
                   edgecolor='navy', 
                   linewidth=0.3,
                   label='Bid OI Increase')
        
        if len(negative_bid_oi) > 0:
            ax3.bar(negative_bid_oi['Price'], 
                   negative_bid_oi['OI_Delta'], 
                   width=bar_width, 
                   color='darkblue',
                   edgecolor='navy', 
                   linewidth=0.3,
                   label='Bid OI Decrease')
    
    # Plot ask OI changes (orange for positive, darker orange for negative)
    if len(ask_oi_significant) > 0:
        positive_ask_oi = ask_oi_significant[ask_oi_significant['OI_Delta'] > 0]
        negative_ask_oi = ask_oi_significant[ask_oi_significant['OI_Delta'] < 0]
        
        if len(positive_ask_oi) > 0:
            ax3.bar(positive_ask_oi['Price'], 
                   positive_ask_oi['OI_Delta'], 
                   width=bar_width, 
                   color='orange',
                   edgecolor='darkorange', 
                   linewidth=0.3,
                   label='Ask OI Increase')
        
        if len(negative_ask_oi) > 0:
            ax3.bar(negative_ask_oi['Price'], 
                   negative_ask_oi['OI_Delta'], 
                   width=bar_width, 
                   color='darkorange',
                   edgecolor='saddlebrown', 
                   linewidth=0.3,
                   label='Ask OI Decrease')
    
    ax3.set_xlabel(f'Price ({quote_currency})', fontsize=9)
    ax3.set_ylabel(f'OI Delta ({base_currency})', fontsize=9)
    ax3.set_title(f'{symbol} Open Interest Delta', fontsize=10, pad=5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    
    # Set x-axis limits for all plots based on the price range
    if len(all_prices) > 1:
        x_min = np.min(all_prices) - price_range * 0.05
        x_max = np.max(all_prices) + price_range * 0.05
        
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            
            # Format x-axis ticks with dollar signs
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.2f}'))
            
            # Format y-axis with comma-separated numbers
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y):,}' if y == int(y) else f'{y:,.2f}'))
            
            # Adjust tick parameters
            ax.tick_params(axis='both', colors='black', labelsize=8)
    
    # Add legends to plots 2 and 3
    ax2.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, loc='upper right')
    ax3.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()

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
        st.subheader("Enter Ticker Symbol")
        symbol = st.text_input("Symbol (e.g. BTC-USD)", value="BTC-USD", key="symbol_input").strip()
        
        st.subheader("Select Lookback Period")
        lookback_options = {
            "1 Day (1440 1m bars)": 1440,
            "4 Hours (240 1m bars)": 240
        }
        lookback = st.selectbox("Lookback", options=list(lookback_options.keys()), key="lookback_select")
        lookback_value = lookback_options[lookback]
        
        st.subheader("Enter Ticker Symbol for Order Book")
        ob_symbol = st.text_input("Symbol (e.g. BTC-USD)", value=symbol, key="ob_symbol_input").strip()
        
        st.subheader("Select Order Book Depth")
        normalized_depth = st.slider("Depth", 
                                   min_value=1, 
                                   max_value=100, 
                                   value=50, 
                                   key="depth_slider", 
                                   help="Order book depth (1-100 scale)")
        
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
        
        # Add note about market type
        st.markdown("""
        ---
        **Note:** This dashboard uses Coinbase spot markets.
        Common pairs: BTC-USD, ETH-USD, SOL-USD, etc.
        """)
        
        st.subheader("Update Frequency")
        st.text("Data updates every 30 seconds")

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
            
            with momentum_placeholder.container():
                # Momentum Imbalances Section
                st.markdown("<h2>Momentum Imbalances</h2>", unsafe_allow_html=True)
                st.markdown(f"Fetching data for {symbol} with {lookback_value} 1-minute data points.", unsafe_allow_html=True)
                
                with st.spinner("Fetching momentum data..."):
                    exchange = initialize_exchange()
                    df = fetch_market_data(exchange, symbol, lookback_value)
                    
                    if last_minute is None or current_minute > last_minute:
                        CONFIG['analysis']['amplitude_threshold'] = amplitude_threshold
                        CONFIG['analysis']['inactive_period'] = inactive_period
                        df = calculate_metrics(df, exchange, symbol, orderbook_depth)
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
                # Original Market Depth Section
                st.markdown("<h2>📊 Order Book Depth (Live)</h2>", unsafe_allow_html=True)
                
                with st.spinner("Fetching order book data..."):
                    # Create market depth plot with more square aspect ratio
                    fig_depth = plt.figure(figsize=(10, 8))  # Square dimensions
                    ax_depth = fig_depth.add_subplot(111)
                    
                    # Fetch current order book if not already fetched
                    if 'exchange' not in locals():
                        exchange = initialize_exchange()
                    current_order_book = fetch_order_book(exchange, ob_symbol, orderbook_depth)
                    
                    # Plot the original market depth visualization
                    plot_market_depth(fig_depth, ax_depth, current_order_book, ob_symbol)
                    
                    # Display the plot
                    st.pyplot(fig_depth)
                    plt.close(fig_depth)
            
            with heatmap_placeholder.container():
                # New Order Book Heatmap Section
                st.markdown("<h2>📈 Order Book Heatmap (Live)</h2>", unsafe_allow_html=True)
                
                with st.spinner("Fetching order book heatmap data..."):
                    # If we don't have a previous order book, use the current one
                    if previous_order_book is None:
                        # Wait a short time to get a second snapshot
                        time.sleep(2)
                        previous_order_book = current_order_book
                        current_order_book = fetch_order_book(exchange, ob_symbol, orderbook_depth)
                    
                    # Create market depth plot with three subplots
                    fig_heatmap = plt.figure(figsize=(10, 12))
                    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.8, 0.8])
                    
                    ax1 = fig_heatmap.add_subplot(gs[0])  # Order Book Depth
                    ax2 = fig_heatmap.add_subplot(gs[1])  # Volume Delta
                    ax3 = fig_heatmap.add_subplot(gs[2])  # OI Delta
                    
                    # Plot the order book heatmap
                    plot_order_book_heatmap(fig_heatmap, [ax1, ax2, ax3], previous_order_book, current_order_book, ob_symbol)
                    
                    # Display the plot
                    st.pyplot(fig_heatmap)
                    plt.close(fig_heatmap)
                    
                    # Update previous order book for next iteration
                    previous_order_book = current_order_book
            
            # Wait for 30 seconds before the next update
            time.sleep(30)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Keep the original plot_market_depth function
def plot_market_depth(fig, ax, order_book: Dict, symbol: str):
    """Create a market depth visualization with white theme matching the reference design."""
    if not order_book or 'bids' not in order_book or 'asks' not in order_book:
        return
    
    # Process bids and asks
    bids = np.array(order_book['bids'], dtype=float) if len(order_book['bids']) > 0 else np.array([[0, 0]])
    asks = np.array(order_book['asks'], dtype=float) if len(order_book['asks']) > 0 else np.array([[0, 0]])
    
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

if __name__ == "__main__":
    main() 