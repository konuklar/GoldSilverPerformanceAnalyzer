import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Gold & Silver Futures Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize QuantStats
qs.extend_pandas()

# Constants
RF_RATE = 0.02  # 2% risk-free rate

def calculate_bollinger_bands(price_series, window=20, num_std=2):
    """
    Calculate Bollinger Bands for a price series
    
    Parameters:
    - price_series: pandas Series of prices
    - window: moving average window (default 20)
    - num_std: number of standard deviations for bands (default 2)
    
    Returns:
    - dict with middle_band, upper_band, lower_band, price, and signals
    """
    if len(price_series) < window:
        return None
    
    # Calculate middle band (simple moving average)
    middle_band = price_series.rolling(window=window).mean()
    
    # Calculate rolling standard deviation
    rolling_std = price_series.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    # Calculate Bollinger Band Width (for normalization)
    bb_width = (upper_band - lower_band) / middle_band
    
    # Calculate %B indicator (where price is within bands)
    percent_b = (price_series - lower_band) / (upper_band - lower_band)
    
    # Identify breaches
    upper_breach = price_series > upper_band
    lower_breach = price_series < lower_band
    
    # Calculate log returns for additional insights
    log_returns = np.log(price_series / price_series.shift(1))
    
    # Calculate volatility (for band adjustment suggestions)
    volatility = price_series.pct_change().rolling(window=window).std() * np.sqrt(252) * 100
    
    return {
        'price': price_series,
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'upper_breach': upper_breach,
        'lower_breach': lower_breach,
        'percent_b': percent_b,
        'bb_width': bb_width,
        'log_returns': log_returns,
        'volatility': volatility,
        'window': window,
        'num_std': num_std
    }

def create_bollinger_bands_chart(price_data, returns_data, tickers, bollinger_params):
    """
    Create comprehensive Bollinger Bands chart with multiple views
    """
    if price_data.empty or returns_data.empty:
        return go.Figure()
    
    # Determine layout based on number of tickers
    num_tickers = len(tickers)
    
    # Create subplot layout: 3 rows for each ticker (price with BB, %B, band breaches)
    fig = make_subplots(
        rows=3 * num_tickers, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[]
    )
    
    # Generate titles for each row
    for ticker in tickers:
        fig.add_annotation(
            x=0.5, y=1.05, 
            xref="paper", yref="paper",
            text=f"{ticker} - Bollinger Bands Analysis",
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="center",
            yanchor="bottom",
            row=3 * tickers.index(ticker) + 1,
            col=1
        )
    
    row_counter = 1
    
    for ticker in tickers:
        if ticker in price_data.columns:
            # Get Bollinger Bands
            bb_data = calculate_bollinger_bands(
                price_data[ticker].dropna(),
                window=bollinger_params['window'],
                num_std=bollinger_params['num_std']
            )
            
            if bb_data is None:
                continue
            
            # Row 1: Price with Bollinger Bands
            price_idx = bb_data['price'].index
            
            # Price line
            fig.add_trace(
                go.Scatter(
                    x=price_idx,
                    y=bb_data['price'].values,
                    mode='lines',
                    name=f'{ticker} Price',
                    line=dict(color='blue', width=2),
                    legendgroup=ticker,
                    showlegend=True if row_counter == 1 else False
                ),
                row=row_counter, col=1
            )
            
            # Middle band
            fig.add_trace(
                go.Scatter(
                    x=price_idx,
                    y=bb_data['middle_band'].values,
                    mode='lines',
                    name=f'MA({bb_data["window"]})',
                    line=dict(color='red', width=1.5, dash='dash'),
                    legendgroup=ticker,
                    showlegend=True if row_counter == 1 else False
                ),
                row=row_counter, col=1
            )
            
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=price_idx,
                    y=bb_data['upper_band'].values,
                    mode='lines',
                    name=f'Upper Band ({bb_data["num_std"]}σ)',
                    line=dict(color='green', width=1, dash='dot'),
                    fill=None,
                    legendgroup=ticker,
                    showlegend=True if row_counter == 1 else False
                ),
                row=row_counter, col=1
            )
            
            # Lower band
            fig.add_trace(
                go.Scatter(
                    x=price_idx,
                    y=bb_data['lower_band'].values,
                    mode='lines',
                    name=f'Lower Band ({bb_data["num_std"]}σ)',
                    line=dict(color='orange', width=1, dash='dot'),
                    fill='tonexty',  # Fill between upper and lower bands
                    fillcolor='rgba(128, 128, 128, 0.1)',
                    legendgroup=ticker,
                    showlegend=True if row_counter == 1 else False
                ),
                row=row_counter, col=1
            )
            
            # Highlight breaches - Upper band breaches
            upper_breach_dates = price_idx[bb_data['upper_breach']]
            upper_breach_prices = bb_data['price'][bb_data['upper_breach']]
            
            if len(upper_breach_dates) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=upper_breach_dates,
                        y=upper_breach_prices,
                        mode='markers',
                        name='Upper Band Breach',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='triangle-down',
                            line=dict(width=1, color='darkred')
                        ),
                        legendgroup=f"{ticker}_breaches",
                        showlegend=True if row_counter == 1 else False
                    ),
                    row=row_counter, col=1
                )
            
            # Highlight breaches - Lower band breaches
            lower_breach_dates = price_idx[bb_data['lower_breach']]
            lower_breach_prices = bb_data['price'][bb_data['lower_breach']]
            
            if len(lower_breach_dates) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=lower_breach_dates,
                        y=lower_breach_prices,
                        mode='markers',
                        name='Lower Band Breach',
                        marker=dict(
                            color='green',
                            size=8,
                            symbol='triangle-up',
                            line=dict(width=1, color='darkgreen')
                        ),
                        legendgroup=f"{ticker}_breaches",
                        showlegend=True if row_counter == 1 else False
                    ),
                    row=row_counter, col=1
                )
            
            fig.update_yaxes(title_text="Price", row=row_counter, col=1)
            
            # Row 2: %B Indicator
            row_counter += 1
            
            # %B line
            fig.add_trace(
                go.Scatter(
                    x=price_idx,
                    y=bb_data['percent_b'].values,
                    mode='lines',
                    name='%B Indicator',
                    line=dict(color='purple', width=1.5),
                    legendgroup=f"{ticker}_percent_b",
                    showlegend=True if row_counter == 2 else False
                ),
                row=row_counter, col=1
            )
            
            # Add horizontal lines for overbought/oversold levels
            fig.add_hline(y=1, line_dash="dash", line_color="red", 
                         annotation_text="Overbought (100%)", 
                         annotation_position="bottom right",
                         row=row_counter, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="green", 
                         annotation_text="Oversold (0%)", 
                         annotation_position="top right",
                         row=row_counter, col=1)
            fig.add_hline(y=0.5, line_dash="dot", line_color="gray", 
                         annotation_text="Middle", 
                         annotation_position="top right",
                         row=row_counter, col=1)
            
            # Highlight overbought/oversold areas
            fig.add_hrect(y0=0.8, y1=1, line_width=0, 
                         fillcolor="red", opacity=0.1,
                         row=row_counter, col=1)
            fig.add_hrect(y0=0, y1=0.2, line_width=0, 
                         fillcolor="green", opacity=0.1,
                         row=row_counter, col=1)
            
            fig.update_yaxes(title_text="%B Indicator", 
                           range=[-0.1, 1.1],
                           row=row_counter, col=1)
            
            # Row 3: Bollinger Band Width and Breach Frequency
            row_counter += 1
            
            # Band Width
            fig.add_trace(
                go.Scatter(
                    x=price_idx,
                    y=bb_data['bb_width'].values,
                    mode='lines',
                    name='Band Width',
                    line=dict(color='brown', width=1.5),
                    yaxis='y1',
                    legendgroup=f"{ticker}_width",
                    showlegend=True if row_counter == 3 else False
                ),
                row=row_counter, col=1
            )
            
            # Calculate breach frequency (rolling 20-day)
            breach_frequency = (bb_data['upper_breach'] | bb_data['lower_breach']).rolling(window=20).mean() * 100
            
            fig.add_trace(
                go.Scatter(
                    x=price_idx,
                    y=breach_frequency.values,
                    mode='lines',
                    name='Breach Frequency (%)',
                    line=dict(color='cyan', width=1.5, dash='dash'),
                    yaxis='y2',
                    legendgroup=f"{ticker}_frequency",
                    showlegend=True if row_counter == 3 else False
                ),
                row=row_counter, col=1
            )
            
            # Add secondary y-axis for breach frequency
            fig.update_layout(
                yaxis2=dict(
                    title="Breach Frequency (%)",
                    titlefont=dict(color="cyan"),
                    tickfont=dict(color="cyan"),
                    overlaying="y",
                    side="right",
                    range=[0, 50]
                )
            )
            
            fig.update_yaxes(title_text="Band Width", 
                           row=row_counter, col=1)
            
            row_counter += 1
    
    # Update layout
    fig.update_layout(
        height=350 * num_tickers,
        title_text="Bollinger Bands Analysis",
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axis labels only for bottom row
    for i in range(1, 3 * num_tickers):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    fig.update_xaxes(title_text="Date", row=3 * num_tickers, col=1)
    
    return fig

def create_log_returns_with_bb_chart(price_data, tickers, bollinger_params):
    """
    Create chart showing log returns with Bollinger Bands on returns
    """
    if price_data.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=len(tickers), 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f"{ticker} - Log Returns with Bollinger Bands" for ticker in tickers]
    )
    
    for idx, ticker in enumerate(tickers, 1):
        if ticker in price_data.columns:
            # Calculate log returns
            price_series = price_data[ticker].dropna()
            log_returns = np.log(price_series / price_series.shift(1)).dropna()
            
            if len(log_returns) < bollinger_params['window']:
                continue
            
            # Calculate Bollinger Bands on log returns
            middle_band = log_returns.rolling(window=bollinger_params['window']).mean()
            rolling_std = log_returns.rolling(window=bollinger_params['window']).std()
            upper_band = middle_band + (rolling_std * bollinger_params['num_std'])
            lower_band = middle_band - (rolling_std * bollinger_params['num_std'])
            
            # Identify breaches
            upper_breach = log_returns > upper_band
            lower_breach = log_returns < lower_band
            
            # Log returns line
            fig.add_trace(
                go.Scatter(
                    x=log_returns.index,
                    y=log_returns.values * 100,  # Convert to percentage
                    mode='lines',
                    name='Log Returns',
                    line=dict(color='blue', width=1.5),
                    legendgroup=ticker,
                    showlegend=True if idx == 1 else False
                ),
                row=idx, col=1
            )
            
            # Middle band
            fig.add_trace(
                go.Scatter(
                    x=log_returns.index,
                    y=middle_band.values * 100,
                    mode='lines',
                    name=f'MA({bollinger_params["window"]})',
                    line=dict(color='red', width=1, dash='dash'),
                    legendgroup=ticker,
                    showlegend=True if idx == 1 else False
                ),
                row=idx, col=1
            )
            
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=log_returns.index,
                    y=upper_band.values * 100,
                    mode='lines',
                    name=f'Upper Band ({bollinger_params["num_std"]}σ)',
                    line=dict(color='green', width=1, dash='dot'),
                    legendgroup=ticker,
                    showlegend=True if idx == 1 else False
                ),
                row=idx, col=1
            )
            
            # Lower band
            fig.add_trace(
                go.Scatter(
                    x=log_returns.index,
                    y=lower_band.values * 100,
                    mode='lines',
                    name=f'Lower Band ({bollinger_params["num_std"]}σ)',
                    line=dict(color='orange', width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)',
                    legendgroup=ticker,
                    showlegend=True if idx == 1 else False
                ),
                row=idx, col=1
            )
            
            # Highlight breaches
            # Upper breaches
            upper_breach_dates = log_returns.index[upper_breach]
            upper_breach_values = log_returns[upper_breach] * 100
            
            if len(upper_breach_dates) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=upper_breach_dates,
                        y=upper_breach_values,
                        mode='markers',
                        name='Upper Band Breach',
                        marker=dict(
                            color='red',
                            size=6,
                            symbol='triangle-down',
                            line=dict(width=1, color='darkred')
                        ),
                        legendgroup=f"{ticker}_breaches",
                        showlegend=True if idx == 1 else False
                    ),
                    row=idx, col=1
                )
            
            # Lower breaches
            lower_breach_dates = log_returns.index[lower_breach]
            lower_breach_values = log_returns[lower_breach] * 100
            
            if len(lower_breach_dates) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=lower_breach_dates,
                        y=lower_breach_values,
                        mode='markers',
                        name='Lower Band Breach',
                        marker=dict(
                            color='green',
                            size=6,
                            symbol='triangle-up',
                            line=dict(width=1, color='darkgreen')
                        ),
                        legendgroup=f"{ticker}_breaches",
                        showlegend=True if idx == 1 else False
                    ),
                    row=idx, col=1
                )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="solid", line_color="black", 
                         line_width=0.5, row=idx, col=1)
            
            # Update y-axis
            fig.update_yaxes(title_text="Log Returns (%)", row=idx, col=1)
    
    fig.update_layout(
        height=300 * len(tickers),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Only show x-axis label for bottom subplot
    for i in range(1, len(tickers)):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    fig.update_xaxes(title_text="Date", row=len(tickers), col=1)
    
    return fig

def create_bb_statistics_table(price_data, tickers, bollinger_params):
    """
    Create statistics table for Bollinger Bands analysis
    """
    stats_data = []
    
    for ticker in tickers:
        if ticker in price_data.columns:
            bb_data = calculate_bollinger_bands(
                price_data[ticker].dropna(),
                window=bollinger_params['window'],
                num_std=bollinger_params['num_std']
            )
            
            if bb_data is None:
                continue
            
            # Calculate statistics
            total_breaches = bb_data['upper_breach'].sum() + bb_data['lower_breach'].sum()
            breach_percentage = (total_breaches / len(bb_data['price'])) * 100
            
            upper_breach_percentage = (bb_data['upper_breach'].sum() / len(bb_data['price'])) * 100
            lower_breach_percentage = (bb_data['lower_breach'].sum() / len(bb_data['price'])) * 100
            
            avg_band_width = bb_data['bb_width'].mean() * 100  # as percentage
            max_band_width = bb_data['bb_width'].max() * 100
            min_band_width = bb_data['bb_width'].min() * 100
            
            avg_volatility = bb_data['volatility'].mean()
            
            stats_data.append({
                'Ticker': ticker,
                'Total Breaches': int(total_breaches),
                'Breach %': f"{breach_percentage:.2f}%",
                'Upper Breach %': f"{upper_breach_percentage:.2f}%",
                'Lower Breach %': f"{lower_breach_percentage:.2f}%",
                'Avg Band Width %': f"{avg_band_width:.2f}%",
                'Max Band Width %': f"{max_band_width:.2f}%",
                'Min Band Width %': f"{min_band_width:.2f}%",
                'Avg Volatility': f"{avg_volatility:.2f}%",
                'Window': bb_data['window'],
                'Std Dev': bb_data['num_std']
            })
    
    return pd.DataFrame(stats_data)

# Cache data download with better error handling
@st.cache_data(ttl=3600)
def download_futures_data(tickers, start_date='2010-01-01', end_date=None):
    """Download futures data from Yahoo Finance with error handling"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Download data
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date,
            progress=False,
            auto_adjust=True  # Use adjusted prices
        )
        
        if data.empty:
            return pd.DataFrame()
        
        # Check if we have a single ticker or multiple
        if len(tickers) == 1:
            # Single ticker returns a DataFrame with single-level columns
            if 'Adj Close' in data.columns:
                price_data = data[['Adj Close']].copy()
                price_data.columns = tickers
            else:
                # Try to find adjusted close column
                price_data = data[['Close']].copy()
                price_data.columns = tickers
        else:
            # Multiple tickers returns MultiIndex columns
            if ('Adj Close', tickers[0]) in data.columns:
                price_data = data['Adj Close'].copy()
            else:
                price_data = data.xs('Close', axis=1, level=0).copy()
        
        return price_data.dropna()
    
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return pd.DataFrame()

def validate_and_prepare_data(price_data, tickers):
    """Validate and prepare data for analysis"""
    if price_data.empty:
        return pd.DataFrame()
    
    # Check for each ticker
    valid_data = {}
    for ticker in tickers:
        if ticker in price_data.columns:
            series = price_data[ticker].dropna()
            
            # Remove leading/trailing zeros or NaN values
            series = series.replace(0, np.nan).dropna()
            
            if len(series) >= 2:  # Need at least 2 points for returns
                # Forward fill small gaps (up to 5 days)
                series = series.ffill(limit=5)
                
                # Ensure no negative prices (though possible, very rare)
                series = series[series > 0]
                
                if len(series) >= 2:
                    valid_data[ticker] = series
    
    if not valid_data:
        return pd.DataFrame()
    
    # Create DataFrame from valid data
    valid_df = pd.DataFrame(valid_data)
    
    # Align dates (outer join then forward fill)
    valid_df = valid_df.ffill(limit=5).dropna()
    
    # Calculate returns
    returns_df = valid_df.pct_change().dropna()
    
    # Remove extreme outliers (more than 50% daily move)
    returns_df = returns_df[(returns_df.abs() < 0.5).all(axis=1)]
    
    # Ensure we have enough data
    if len(returns_df) < 5:
        return pd.DataFrame()
    
    return returns_df

def get_data_with_validation(tickers, start_date, end_date, period):
    """Get and validate data with period filtering"""
    try:
        # Download data
        price_data = download_futures_data(
            tickers, 
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if price_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate returns
        returns_df = validate_and_prepare_data(price_data, tickers)
        
        if returns_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Apply period filter
        if period != "Full History":
            if period == "YTD":
                current_year = pd.Timestamp.now().year
                price_filtered = price_data[price_data.index.year == current_year]
                returns_filtered = returns_df[returns_df.index.year == current_year]
            elif period == "1Y":
                cutoff_date = returns_df.index.max() - pd.DateOffset(years=1)
                price_filtered = price_data[price_data.index >= cutoff_date]
                returns_filtered = returns_df[returns_df.index >= cutoff_date]
            elif period == "3Y":
                cutoff_date = returns_df.index.max() - pd.DateOffset(years=3)
                price_filtered = price_data[price_data.index >= cutoff_date]
                returns_filtered = returns_df[returns_df.index >= cutoff_date]
            elif period == "5Y":
                cutoff_date = returns_df.index.max() - pd.DateOffset(years=5)
                price_filtered = price_data[price_data.index >= cutoff_date]
                returns_filtered = returns_df[returns_df.index >= cutoff_date]
            else:
                price_filtered = price_data
                returns_filtered = returns_df
        else:
            price_filtered = price_data
            returns_filtered = returns_df
        
        # Final validation - ensure we have enough data
        if len(returns_filtered) < 5:
            st.warning(f"Insufficient data for {period} period. Using all available data.")
            price_filtered = price_data
            returns_filtered = returns_df
        
        return price_filtered, returns_filtered
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def safe_quantstats_calculation(func, returns, *args, **kwargs):
    """Safely calculate QuantStats metrics with error handling"""
    try:
        if len(returns) < 2:
            return np.nan
        
        # Ensure returns is a pandas Series
        if isinstance(returns, pd.DataFrame):
            if len(returns.columns) > 0:
                returns = returns.iloc[:, 0]
            else:
                return np.nan
        
        # Check for NaN or infinite values
        if returns.isna().any() or not np.isfinite(returns).all():
            return np.nan
        
        result = func(returns, *args, **kwargs)
        
        # Handle potential NaN results
        if pd.isna(result):
            return np.nan
        
        return result
        
    except Exception:
        return np.nan

# Performance metrics calculation with error handling
@st.cache_data
def calculate_metrics(returns_df):
    """Calculate comprehensive performance metrics with error handling"""
    metrics = {}
    
    for col in returns_df.columns:
        returns = returns_df[col].dropna()
        
        if len(returns) < 10:  # Minimum data points
            metrics[col] = {metric: np.nan for metric in [
                'Cumulative Return', 'Annual Return', 'Annual Volatility',
                'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown',
                'Calmar Ratio', 'Omega Ratio', 'VaR (95%)', 'CVaR (95%)',
                'Skewness', 'Kurtosis', 'Win Rate', 'Profit Factor',
                'Tail Ratio', 'Daily Value at Risk', 'Expected Shortfall'
            ]}
            continue
        
        # Calculate each metric with error handling
        metrics[col] = {
            'Cumulative Return': safe_quantstats_calculation(qs.stats.comp, returns) * 100,
            'Annual Return': safe_quantstats_calculation(qs.stats.cagr, returns) * 100,
            'Annual Volatility': safe_quantstats_calculation(qs.stats.volatility, returns) * 100,
            'Sharpe Ratio': safe_quantstats_calculation(qs.stats.sharpe, returns, rf=RF_RATE),
            'Sortino Ratio': safe_quantstats_calculation(qs.stats.sortino, returns, rf=RF_RATE),
            'Max Drawdown': safe_quantstats_calculation(qs.stats.max_drawdown, returns) * 100,
            'Calmar Ratio': safe_quantstats_calculation(qs.stats.calmar, returns),
            'Omega Ratio': safe_quantstats_calculation(qs.stats.omega, returns, rf=RF_RATE),
            'VaR (95%)': safe_quantstats_calculation(qs.stats.value_at_risk, returns) * 100,
            'CVaR (95%)': safe_quantstats_calculation(qs.stats.cvar, returns) * 100,
            'Skewness': safe_quantstats_calculation(qs.stats.skew, returns),
            'Kurtosis': safe_quantstats_calculation(qs.stats.kurtosis, returns),
            'Win Rate': safe_quantstats_calculation(qs.stats.win_rate, returns) * 100,
            'Profit Factor': safe_quantstats_calculation(qs.stats.profit_factor, returns),
            'Tail Ratio': safe_quantstats_calculation(qs.stats.tail_ratio, returns),
            'Daily Value at Risk': safe_quantstats_calculation(qs.stats.value_at_risk, returns) * 100,
            'Expected Shortfall': safe_quantstats_calculation(qs.stats.expected_shortfall, returns) * 100,
        }
    
    return metrics

# [Previous chart functions remain the same: create_returns_chart, create_drawdown_chart, 
# create_monthly_heatmap, create_distribution_chart, create_rolling_metrics_chart]
# (Keeping them as they were, not repeating for brevity)

def format_metric_value(value, metric_name):
    """Format metric values appropriately"""
    if pd.isna(value):
        return "N/A"
    
    if 'Ratio' in metric_name or 'Rate' in metric_name or metric_name in ['Skewness', 'Kurtosis', 'Profit Factor', 'Tail Ratio']:
        return f"{value:.3f}"
    elif 'Return' in metric_name or 'Volatility' in metric_name or 'Drawdown' in metric_name or 'VaR' in metric_name:
        return f"{value:.2f}%"
    else:
        return f"{value:.3f}"

# Main application
def main():
    st.title("📊 Gold & Silver Futures Performance Analyzer")
    st.markdown("""
    Analyze performance and risk metrics for Gold (GC=F) and Silver (SI=F) futures using QuantStats.
    Risk-free rate is set to 2%.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Ticker selection with full names
    ticker_info = {
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "HG=F": "Copper Futures"
    }
    
    ticker_options = list(ticker_info.keys())
    display_names = [f"{ticker} - {ticker_info[ticker]}" for ticker in ticker_options]
    
    selected_display = st.sidebar.multiselect(
        "Select Futures Contracts:",
        display_names,
        default=[display_names[0], display_names[1]]
    )
    
    # Extract ticker symbols
    tickers = [name.split(" - ")[0] for name in selected_display]
    
    # Date range with sensible defaults
    col1, col2 = st.sidebar.columns(2)
    with col1:
        # Default to 5 years ago for better data
        default_start = datetime.now() - timedelta(days=5*365)
        start_date = st.date_input("Start Date", default_start)
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Ensure start date is before end date
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date!")
        return
    
    # Analysis period
    period = st.sidebar.selectbox(
        "Analysis Period:",
        ["Full History", "YTD", "1Y", "3Y", "5Y"]
    )
    
    # Bollinger Bands parameters
    st.sidebar.header("Bollinger Bands Settings")
    
    bb_window = st.sidebar.slider(
        "Moving Average Window:",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of periods for moving average"
    )
    
    bb_std = st.sidebar.slider(
        "Standard Deviations:",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.5,
        help="Number of standard deviations for bands"
    )
    
    bollinger_params = {
        'window': bb_window,
        'num_std': bb_std
    }
    
    # Minimum data threshold
    min_days = st.sidebar.slider(
        "Minimum days of data required:",
        min_value=10,
        max_value=100,
        value=30,
        help="Skip analysis if we have fewer than this many data points"
    )
    
    # Download button
    if st.sidebar.button("🔄 Update Data"):
        st.cache_data.clear()
    
    if not tickers:
        st.warning("Please select at least one futures contract.")
        return
    
    # Download and validate data
    with st.spinner("Downloading and validating futures data..."):
        price_data, returns_df = get_data_with_validation(tickers, start_date, end_date, period)
        
        if returns_df.empty:
            st.error("""
            No valid data available for the selected criteria. This could be due to:
            1. No trading data for the selected date range
            2. All data points are NaN or zero
            3. Insufficient data points after cleaning
            
            Please try:
            - Selecting a different date range
            - Using the default futures (Gold and Silver)
            - Checking if markets were open during the selected period
            """)
            return
        
        if len(returns_df) < min_days:
            st.warning(f"Only {len(returns_df)} days of data available (minimum requested: {min_days}). Analysis may be limited.")
        
        # Display data info
        st.sidebar.success(f"Data loaded: {len(returns_df)} trading days")
        st.sidebar.info(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    
    # Main dashboard - ADDED BOLLINGER BANDS TAB
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", "📊 Performance Metrics", "📉 Risk Analysis", 
        "📊 Bollinger Bands", "🔍 Advanced Charts", "📋 Data & Diagnostics"
    ])
    
    # [Previous tabs 1-3 remain the same]
    # (Keeping their content as before, not repeating for brevity)
    
    with tab4:  # NEW BOLLINGER BANDS TAB
        st.header("📊 Bollinger Bands Analysis")
        st.markdown(f"""
        **Parameters:** {bollinger_params['window']}-period moving average with ±{bollinger_params['num_std']} standard deviation bands
        
        Bollinger Bands consist of:
        - **Middle Band**: {bollinger_params['window']}-period simple moving average
        - **Upper Band**: Middle band + ({bollinger_params['num_std']} × standard deviation)
        - **Lower Band**: Middle band - ({bollinger_params['num_std']} × standard deviation)
        """)
        
        # Bollinger Bands Statistics
        st.subheader("Bollinger Bands Statistics")
        bb_stats = create_bb_statistics_table(price_data, tickers, bollinger_params)
        
        if not bb_stats.empty:
            # Display statistics
            cols = st.columns(len(bb_stats.columns))
            for idx, col in enumerate(bb_stats.columns):
                with cols[idx % len(cols)]:
                    st.dataframe(bb_stats[[col]], use_container_width=True)
        
        # Comprehensive Bollinger Bands Chart
        st.subheader("Comprehensive Bollinger Bands Analysis")
        bb_chart = create_bollinger_bands_chart(price_data, returns_df, tickers, bollinger_params)
        st.plotly_chart(bb_chart, use_container_width=True)
        
        # Log Returns with Bollinger Bands
        st.subheader("Log Returns with Bollinger Bands")
        st.markdown("""
        This chart shows **log returns** (continuously compounded returns) with Bollinger Bands applied to the returns themselves.
        This helps identify periods of unusually high or low volatility in returns.
        """)
        
        log_returns_bb_chart = create_log_returns_with_bb_chart(price_data, tickers, bollinger_params)
        st.plotly_chart(log_returns_bb_chart, use_container_width=True)
        
        # Trading Signals Section
        st.subheader("Bollinger Bands Trading Signals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Traditional Signals:**
            
            **1. Squeeze (Low Volatility)**
            - Narrowing bands indicate low volatility
            - Often precedes a period of high volatility
            - Watch for band expansion
            
            **2. Breakout Signals**
            - Price moving outside bands suggests strong momentum
            - Upper band breach: potential overbought
            - Lower band breach: potential oversold
            """)
        
        with col2:
            st.markdown("""
            **Advanced Signals:**
            
            **1. %B Indicator**
            - Above 0.8: Overbought zone
            - Below 0.2: Oversold zone
            - Crosses above/below 0.5: Trend changes
            
            **2. Band Width**
            - Increasing width: Rising volatility
            - Decreasing width: Falling volatility
            - Extreme width: Potential mean reversion
            """)
        
        # Interactive Analysis
        st.subheader("Interactive Band Analysis")
        
        selected_ticker = st.selectbox("Select ticker for detailed band analysis:", tickers)
        
        if selected_ticker and selected_ticker in price_data.columns:
            bb_data = calculate_bollinger_bands(
                price_data[selected_ticker].dropna(),
                window=bollinger_params['window'],
                num_std=bollinger_params['num_std']
            )
            
            if bb_data:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_price = bb_data['price'].iloc[-1]
                    current_upper = bb_data['upper_band'].iloc[-1]
                    current_lower = bb_data['lower_band'].iloc[-1]
                    
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric("Distance to Upper Band", 
                             f"{(current_upper - current_price) / current_price * 100:.2f}%",
                             delta="Above" if current_price > current_upper else "Below")
                    st.metric("Distance to Lower Band", 
                             f"{(current_price - current_lower) / current_price * 100:.2f}%",
                             delta="Above" if current_price > current_lower else "Below")
                
                with col2:
                    current_percent_b = bb_data['percent_b'].iloc[-1]
                    current_width = bb_data['bb_width'].iloc[-1] * 100
                    
                    st.metric("%B Indicator", f"{current_percent_b:.2%}")
                    
                    if current_percent_b > 0.8:
                        st.error("⚠️ Overbought territory (>80%)")
                    elif current_percent_b < 0.2:
                        st.success("✅ Oversold territory (<20%)")
                    else:
                        st.info("📊 Neutral territory")
                    
                    st.metric("Band Width", f"{current_width:.2f}%")
                
                with col3:
                    # Recent breaches
                    recent_days = 20
                    recent_upper_breaches = bb_data['upper_breach'][-recent_days:].sum()
                    recent_lower_breaches = bb_data['lower_breach'][-recent_days:].sum()
                    
                    st.metric(f"Upper Breaches (Last {recent_days} days)", recent_upper_breaches)
                    st.metric(f"Lower Breaches (Last {recent_days} days)", recent_lower_breaches)
                    st.metric("Total Recent Breaches", recent_upper_breaches + recent_lower_breaches)
        
        # Download Bollinger Bands Data
        st.subheader("Download Bollinger Bands Data")
        
        if st.button("📥 Export Bollinger Bands Analysis"):
            all_bb_data = {}
            
            for ticker in tickers:
                if ticker in price_data.columns:
                    bb_data = calculate_bollinger_bands(
                        price_data[ticker].dropna(),
                        window=bollinger_params['window'],
                        num_std=bollinger_params['num_std']
                    )
                    
                    if bb_data:
                        df = pd.DataFrame({
                            f'{ticker}_Price': bb_data['price'],
                            f'{ticker}_MA_{bb_data["window"]}': bb_data['middle_band'],
                            f'{ticker}_Upper_Band': bb_data['upper_band'],
                            f'{ticker}_Lower_Band': bb_data['lower_band'],
                            f'{ticker}_Percent_B': bb_data['percent_b'],
                            f'{ticker}_Band_Width': bb_data['bb_width'],
                            f'{ticker}_Upper_Breach': bb_data['upper_breach'],
                            f'{ticker}_Lower_Breach': bb_data['lower_breach'],
                            f'{ticker}_Volatility_%': bb_data['volatility']
                        })
                        
                        all_bb_data[ticker] = df
            
            if all_bb_data:
                # Combine all data
                combined_df = pd.concat(all_bb_data.values(), axis=1)
                csv = combined_df.to_csv().encode('utf-8')
                
                st.download_button(
                    label="Download Bollinger Bands Data (CSV)",
                    data=csv,
                    file_name=f"bollinger_bands_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
    
    # [Previous tabs 5-6 remain as Advanced Charts and Data & Diagnostics]
    # (Keeping their content as before, not repeating for brevity)
    
    # Footer
    st.sidebar.divider()
    st.sidebar.info("""
    **Data Sources:** 
    - Futures data: Yahoo Finance
    - Risk-free rate: 2% (annualized)
    
    **Bollinger Bands:**
    - Developed by John Bollinger
    - Used to identify volatility and potential reversal points
    - Band breaches indicate extreme price movements
    """)

if __name__ == "__main__":
    main()
