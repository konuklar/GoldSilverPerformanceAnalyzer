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
    page_title="QuantEdge Futures Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize QuantEdge (QuantStats)
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
    Simplified version without complex subplot layout issues
    """
    if price_data.empty or returns_data.empty:
        return go.Figure()
    
    # We'll create separate figures for each chart type
    # This avoids the subplot complexity issues
    
    # Create tabs for each ticker
    st.subheader("Detailed Bollinger Bands Analysis by Ticker")
    
    tabs = st.tabs([f"📊 {ticker}" for ticker in tickers])
    
    for tab_idx, ticker in enumerate(tickers):
        with tabs[tab_idx]:
            if ticker in price_data.columns:
                # Get Bollinger Bands
                bb_data = calculate_bollinger_bands(
                    price_data[ticker].dropna(),
                    window=bollinger_params['window'],
                    num_std=bollinger_params['num_std']
                )
                
                if bb_data is None:
                    st.warning(f"Insufficient data for Bollinger Bands on {ticker}")
                    continue
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_price = bb_data['price'].iloc[-1]
                    current_ma = bb_data['middle_band'].iloc[-1]
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric(f"MA({bb_data['window']})", f"${current_ma:.2f}")
                
                with col2:
                    current_upper = bb_data['upper_band'].iloc[-1]
                    current_lower = bb_data['lower_band'].iloc[-1]
                    st.metric("Upper Band", f"${current_upper:.2f}")
                    st.metric("Lower Band", f"${current_lower:.2f}")
                
                with col3:
                    current_percent_b = bb_data['percent_b'].iloc[-1] * 100
                    current_width = bb_data['bb_width'].iloc[-1] * 100
                    st.metric("%B Indicator", f"{current_percent_b:.1f}%")
                    st.metric("Band Width", f"{current_width:.2f}%")
                
                # Create Price with Bollinger Bands chart
                st.subheader(f"Price with Bollinger Bands")
                
                fig1 = go.Figure()
                
                price_idx = bb_data['price'].index
                
                # Lower band first (for proper filling)
                fig1.add_trace(go.Scatter(
                    x=price_idx,
                    y=bb_data['lower_band'].values,
                    mode='lines',
                    name=f'Lower Band ({bb_data["num_std"]}σ)',
                    line=dict(color='orange', width=1, dash='dot'),
                    fillcolor='rgba(128, 128, 128, 0.1)',
                    fill='tonexty'
                ))
                
                # Upper band
                fig1.add_trace(go.Scatter(
                    x=price_idx,
                    y=bb_data['upper_band'].values,
                    mode='lines',
                    name=f'Upper Band ({bb_data["num_std"]}σ)',
                    line=dict(color='green', width=1, dash='dot'),
                    fill='tonexty'
                ))
                
                # Middle band
                fig1.add_trace(go.Scatter(
                    x=price_idx,
                    y=bb_data['middle_band'].values,
                    mode='lines',
                    name=f'MA({bb_data["window"]})',
                    line=dict(color='red', width=1.5, dash='dash')
                ))
                
                # Price
                fig1.add_trace(go.Scatter(
                    x=price_idx,
                    y=bb_data['price'].values,
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Highlight breaches
                upper_breach_mask = bb_data['upper_breach']
                if upper_breach_mask.any():
                    fig1.add_trace(go.Scatter(
                        x=price_idx[upper_breach_mask],
                        y=bb_data['price'][upper_breach_mask].values,
                        mode='markers',
                        name='Upper Band Breach',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='triangle-down',
                            line=dict(width=1, color='darkred')
                        )
                    ))
                
                lower_breach_mask = bb_data['lower_breach']
                if lower_breach_mask.any():
                    fig1.add_trace(go.Scatter(
                        x=price_idx[lower_breach_mask],
                        y=bb_data['price'][lower_breach_mask].values,
                        mode='markers',
                        name='Lower Band Breach',
                        marker=dict(
                            color='green',
                            size=8,
                            symbol='triangle-up',
                            line=dict(width=1, color='darkgreen')
                        )
                    ))
                
                fig1.update_layout(
                    height=500,
                    title=f"{ticker} - Price with Bollinger Bands",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode='x unified',
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # Create %B Indicator chart
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("%B Indicator")
                    
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Scatter(
                        x=price_idx,
                        y=bb_data['percent_b'].values * 100,  # Convert to percentage
                        mode='lines',
                        name='%B Indicator',
                        line=dict(color='purple', width=1.5)
                    ))
                    
                    # Add horizontal lines and zones
                    fig2.add_hrect(y0=80, y1=100, line_width=0, 
                                 fillcolor="red", opacity=0.1,
                                 annotation_text="Overbought", 
                                 annotation_position="top right")
                    fig2.add_hrect(y0=0, y1=20, line_width=0, 
                                 fillcolor="green", opacity=0.1,
                                 annotation_text="Oversold", 
                                 annotation_position="bottom right")
                    
                    fig2.add_hline(y=100, line_dash="dash", line_color="red")
                    fig2.add_hline(y=0, line_dash="dash", line_color="green")
                    fig2.add_hline(y=50, line_dash="dot", line_color="gray")
                    
                    fig2.update_layout(
                        height=300,
                        yaxis_range=[-10, 110],
                        xaxis_title="Date",
                        yaxis_title="%B Indicator (%)",
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    st.subheader("Band Statistics")
                    
                    # Calculate statistics
                    total_breaches = bb_data['upper_breach'].sum() + bb_data['lower_breach'].sum()
                    breach_percentage = (total_breaches / len(bb_data['price'])) * 100
                    
                    upper_breach_percentage = (bb_data['upper_breach'].sum() / len(bb_data['price'])) * 100
                    lower_breach_percentage = (bb_data['lower_breach'].sum() / len(bb_data['price'])) * 100
                    
                    avg_band_width = bb_data['bb_width'].mean() * 100
                    max_band_width = bb_data['bb_width'].max() * 100
                    min_band_width = bb_data['bb_width'].min() * 100
                    
                    # Display metrics
                    st.metric("Total Breaches", f"{int(total_breaches)}")
                    st.metric("Breach %", f"{breach_percentage:.1f}%")
                    st.metric("Upper Breach %", f"{upper_breach_percentage:.1f}%")
                    st.metric("Lower Breach %", f"{lower_breach_percentage:.1f}%")
                    st.metric("Avg Band Width", f"{avg_band_width:.2f}%")
                    st.metric("Max Band Width", f"{max_band_width:.2f}%")
                    st.metric("Min Band Width", f"{min_band_width:.2f}%")
                
                # Create Band Width and Breach Frequency chart
                st.subheader("Band Width & Breach Frequency")
                
                fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Band Width
                fig3.add_trace(
                    go.Scatter(
                        x=price_idx,
                        y=bb_data['bb_width'].values * 100,
                        mode='lines',
                        name='Band Width (%)',
                        line=dict(color='brown', width=1.5)
                    ),
                    secondary_y=False
                )
                
                # Breach Frequency (rolling 20-day)
                breach_frequency = (bb_data['upper_breach'] | bb_data['lower_breach']).rolling(window=20).mean() * 100
                
                fig3.add_trace(
                    go.Scatter(
                        x=price_idx,
                        y=breach_frequency.values,
                        mode='lines',
                        name='Breach Frequency (%)',
                        line=dict(color='cyan', width=1.5, dash='dash')
                    ),
                    secondary_y=True
                )
                
                # Set y-axes titles
                fig3.update_yaxes(title_text="Band Width (%)", secondary_y=False)
                fig3.update_yaxes(title_text="Breach Frequency (%)", secondary_y=True)
                
                fig3.update_layout(
                    height=300,
                    xaxis_title="Date",
                    hovermode='x unified',
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig3, use_container_width=True)
    
    # Return empty figure for compatibility
    return go.Figure()

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

def safe_quantedge_calculation(func, returns, *args, **kwargs):
    """Safely calculate QuantEdge metrics with error handling"""
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
            'Cumulative Return': safe_quantedge_calculation(qs.stats.comp, returns) * 100,
            'Annual Return': safe_quantedge_calculation(qs.stats.cagr, returns) * 100,
            'Annual Volatility': safe_quantedge_calculation(qs.stats.volatility, returns) * 100,
            'Sharpe Ratio': safe_quantedge_calculation(qs.stats.sharpe, returns, rf=RF_RATE),
            'Sortino Ratio': safe_quantedge_calculation(qs.stats.sortino, returns, rf=RF_RATE),
            'Max Drawdown': safe_quantedge_calculation(qs.stats.max_drawdown, returns) * 100,
            'Calmar Ratio': safe_quantedge_calculation(qs.stats.calmar, returns),
            'Omega Ratio': safe_quantedge_calculation(qs.stats.omega, returns, rf=RF_RATE),
            'VaR (95%)': safe_quantedge_calculation(qs.stats.value_at_risk, returns) * 100,
            'CVaR (95%)': safe_quantedge_calculation(qs.stats.cvar, returns) * 100,
            'Skewness': safe_quantedge_calculation(qs.stats.skew, returns),
            'Kurtosis': safe_quantedge_calculation(qs.stats.kurtosis, returns),
            'Win Rate': safe_quantedge_calculation(qs.stats.win_rate, returns) * 100,
            'Profit Factor': safe_quantedge_calculation(qs.stats.profit_factor, returns),
            'Tail Ratio': safe_quantedge_calculation(qs.stats.tail_ratio, returns),
            'Daily Value at Risk': safe_quantedge_calculation(qs.stats.value_at_risk, returns) * 100,
            'Expected Shortfall': safe_quantedge_calculation(qs.stats.expected_shortfall, returns) * 100,
        }
    
    return metrics

# Advanced plotting functions
def create_returns_chart(returns_df):
    """Create cumulative returns chart"""
    fig = go.Figure()
    
    for col in returns_df.columns:
        try:
            cum_returns = (1 + returns_df[col].dropna()).cumprod()
            if len(cum_returns) > 0:
                fig.add_trace(go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns.values * 100,
                    mode='lines',
                    name=col,
                    line=dict(width=2)
                ))
        except Exception:
            continue
    
    fig.update_layout(
        title='Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_drawdown_chart(returns_df):
    """Create drawdown chart"""
    fig = go.Figure()
    
    for col in returns_df.columns:
        try:
            returns = returns_df[col].dropna()
            if len(returns) > 0:
                drawdown = qs.stats.to_drawdown_series(returns) * 100
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name=col,
                    fill='tozeroy',
                    line=dict(width=1)
                ))
        except Exception:
            continue
    
    fig.update_layout(
        title='Drawdown (%)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_monthly_heatmap(returns_df):
    """Create monthly returns heatmap"""
    if returns_df.empty:
        return go.Figure()
    
    if len(returns_df.columns) > 1:
        returns = returns_df.mean(axis=1)
    else:
        returns = returns_df.iloc[:, 0]
    
    try:
        monthly_returns = qs.stats.monthly_returns(returns) * 100
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=monthly_returns.values,
            x=monthly_returns.columns,
            y=monthly_returns.index,
            colorscale='RdYlGn',
            zmid=0,
            text=monthly_returns.round(2).values,
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap (%)',
            xaxis_title='Month',
            yaxis_title='Year',
            height=400,
            template='plotly_white'
        )
        
        return fig
    except Exception:
        return go.Figure()

def create_distribution_chart(returns_df):
    """Create returns distribution chart"""
    if returns_df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=len(returns_df.columns),
        subplot_titles=returns_df.columns,
        horizontal_spacing=0.1
    )
    
    for idx, col in enumerate(returns_df.columns, 1):
        try:
            returns = returns_df[col].dropna() * 100
            
            if len(returns) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=returns,
                        nbinsx=50,
                        name=col,
                        marker_color='skyblue',
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=1, col=idx
                )
                
                # Add vertical line for mean
                mean_return = returns.mean()
                fig.add_vline(
                    x=mean_return, 
                    line_dash="dash", 
                    line_color="red",
                    row=1, col=idx
                )
                
                fig.update_xaxes(title_text="Daily Return (%)", row=1, col=idx)
                fig.update_yaxes(title_text="Frequency", row=1, col=idx)
        except Exception:
            continue
    
    fig.update_layout(
        title='Returns Distribution',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_rolling_metrics_chart(returns_df):
    """Create rolling Sharpe and volatility chart"""
    if returns_df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Rolling Sharpe Ratio (6-month)', 'Rolling Volatility (6-month)'),
        vertical_spacing=0.15
    )
    
    window = 126  # 6 months trading days
    
    for col in returns_df.columns:
        try:
            returns = returns_df[col].dropna()
            
            if len(returns) < window:
                continue
            
            # Rolling Sharpe
            rolling_sharpe = returns.rolling(window).apply(
                lambda x: safe_quantedge_calculation(qs.stats.sharpe, x, rf=RF_RATE),
                raw=False
            ).dropna()
            
            if len(rolling_sharpe) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe.values,
                        mode='lines',
                        name=f'{col} - Sharpe',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
            
            # Rolling Volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol.values,
                        mode='lines',
                        name=f'{col} - Volatility',
                        line=dict(width=1)
                    ),
                    row=2, col=1
                )
        except Exception:
            continue
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Annualized Volatility (%)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

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
    st.title("📊 QuantEdge Futures Performance Analyzer")
    st.markdown("""
    Analyze performance and risk metrics for Gold (GC=F) and Silver (SI=F) futures using QuantEdge.
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
    
    # Main dashboard
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", "📊 Performance Metrics", "📉 Risk Analysis", 
        "📊 Bollinger Bands", "🔍 Advanced Charts", "📋 Data & Diagnostics"
    ])
    
    with tab1:
        st.header("Performance Overview")
        
        # Display data preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Returns Data Preview")
            st.dataframe(returns_df.tail(10).style.format("{:.4%}"))
        
        with col2:
            st.subheader("Data Statistics")
            stats_df = pd.DataFrame({
                'Mean': returns_df.mean() * 100,
                'Std Dev': returns_df.std() * 100,
                'Min': returns_df.min() * 100,
                'Max': returns_df.max() * 100,
                'Count': returns_df.count()
            }).T
            st.dataframe(stats_df.style.format("{:.2f}%", subset=[col for col in stats_df.columns if col != 'Count']))
        
        # Summary statistics
        st.subheader("Quick Metrics")
        cols = st.columns(len(tickers))
        
        for idx, ticker in enumerate(tickers):
            if ticker in returns_df.columns:
                returns = returns_df[ticker].dropna()
                
                if len(returns) >= 10:
                    cagr = safe_quantedge_calculation(qs.stats.cagr, returns) * 100
                    sharpe = safe_quantedge_calculation(qs.stats.sharpe, returns, rf=RF_RATE)
                    volatility = safe_quantedge_calculation(qs.stats.volatility, returns) * 100
                    
                    with cols[idx]:
                        st.metric(
                            label=f"{ticker} ({ticker_info.get(ticker, ticker)})",
                            value=f"{cagr:.2f}%" if not pd.isna(cagr) else "N/A",
                            delta=f"Sharpe: {sharpe:.2f}" if not pd.isna(sharpe) else "N/A"
                        )
                        st.caption(f"Volatility: {volatility:.2f}%" if not pd.isna(volatility) else "Volatility: N/A")
        
        # Cumulative returns chart
        st.plotly_chart(create_returns_chart(returns_df), use_container_width=True)
        
        # Drawdown chart
        st.plotly_chart(create_drawdown_chart(returns_df), use_container_width=True)
    
    with tab2:
        st.header("Performance Metrics")
        
        # Calculate metrics
        metrics = calculate_metrics(returns_df)
        
        # Display metrics in columns
        for ticker in tickers:
            if ticker in metrics:
                st.subheader(f"{ticker} - {ticker_info.get(ticker, ticker)}")
                
                # Check if we have valid metrics
                if all(pd.isna(v) for v in metrics[ticker].values()):
                    st.warning("Insufficient data to calculate metrics for this instrument.")
                    continue
                
                # Create two columns for metrics
                col1, col2 = st.columns(2)
                
                metric_data = metrics[ticker]
                with col1:
                    for key in list(metric_data.keys())[:len(metric_data)//2]:
                        value = metric_data[key]
                        st.metric(key, format_metric_value(value, key))
                
                with col2:
                    for key in list(metric_data.keys())[len(metric_data)//2:]:
                        value = metric_data[key]
                        st.metric(key, format_metric_value(value, key))
                
                st.divider()
    
    with tab3:
        st.header("Risk Analysis")
        
        # Rolling metrics
        st.plotly_chart(create_rolling_metrics_chart(returns_df), use_container_width=True)
        
        # Risk metrics comparison
        st.subheader("Risk Metrics Comparison")
        
        risk_metrics = ['Annual Volatility', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)', 'Sharpe Ratio']
        
        cols = st.columns(len(tickers))
        for idx, ticker in enumerate(tickers):
            if ticker in metrics:
                with cols[idx]:
                    st.markdown(f"**{ticker}**")
                    for metric in risk_metrics:
                        if metric in metrics[ticker]:
                            value = metrics[ticker][metric]
                            if not pd.isna(value):
                                st.metric(
                                    label=metric,
                                    value=format_metric_value(value, metric)
                                )
    
    with tab4:
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
            st.dataframe(bb_stats, use_container_width=True)
        
        # Comprehensive Bollinger Bands Chart
        create_bollinger_bands_chart(price_data, returns_df, tickers, bollinger_params)
        
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
    
    with tab5:
        st.header("Advanced Charts")
        
        # Returns distribution
        st.plotly_chart(create_distribution_chart(returns_df), use_container_width=True)
        
        # Monthly heatmap
        st.plotly_chart(create_monthly_heatmap(returns_df), use_container_width=True)
        
        # Additional QuantEdge charts
        st.subheader("QuantEdge Detailed Analysis")
        
        if len(returns_df.columns) > 0:
            selected_ticker = st.selectbox("Select ticker for detailed analysis:", tickers)
            
            if selected_ticker:
                returns = returns_df[selected_ticker].dropna()
                
                if len(returns) >= 20:  # Minimum for monthly analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Monthly returns distribution
                        st.write("**Monthly Returns Distribution**")
                        try:
                            monthly_table = qs.stats.monthly_returns(returns) * 100
                            st.dataframe(monthly_table.style.format("{:.2f}%").background_gradient(
                                cmap='RdYlGn', axis=None, vmin=-10, vmax=10
                            ))
                        except Exception:
                            st.warning("Could not calculate monthly returns")
                    
                    with col2:
                        # Worst drawdown periods
                        st.write("**Worst Drawdown Periods**")
                        try:
                            worst_dd = qs.stats.top_drawdowns(returns)
                            
                            if len(worst_dd) > 0:
                                dd_data = []
                                for peak, recovery, dd in worst_dd:
                                    dd_data.append({
                                        'Peak': peak.date() if hasattr(peak, 'date') else peak,
                                        'Recovery': recovery.date() if hasattr(recovery, 'date') else recovery,
                                        'Drawdown': f"{dd * 100:.2f}%"
                                    })
                                st.dataframe(pd.DataFrame(dd_data))
                            else:
                                st.info("No significant drawdowns found")
                        except Exception:
                            st.warning("Could not calculate drawdown periods")
                else:
                    st.warning("Insufficient data for detailed analysis")
    
    with tab6:
        st.header("Data & Diagnostics")
        
        # Show raw data
        st.subheader("Raw Price Data")
        
        # Download raw prices for reference
        try:
            if not price_data.empty:
                st.dataframe(price_data.tail(20))
                
                # Download button for data
                csv = price_data.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Price Data (CSV)",
                    data=csv,
                    file_name="futures_price_data.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No raw price data available")
        except Exception as e:
            st.error(f"Could not load raw data: {str(e)}")
        
        # Data quality report
        st.subheader("Data Quality Report")
        
        quality_report = []
        for ticker in tickers:
            if ticker in returns_df.columns:
                returns = returns_df[ticker].dropna()
                
                quality_report.append({
                    'Ticker': ticker,
                    'Days Available': len(returns),
                    'Missing Values': returns.isna().sum(),
                    'Zero Returns': (returns == 0).sum(),
                    'Positive Days': (returns > 0).sum(),
                    'Negative Days': (returns < 0).sum(),
                    'Start Date': returns.index.min().date() if len(returns) > 0 else 'N/A',
                    'End Date': returns.index.max().date() if len(returns) > 0 else 'N/A'
                })
        
        if quality_report:
            st.dataframe(pd.DataFrame(quality_report))
        
        # System information
        st.subheader("System Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Python Version", f"{pd.__version__}")
        with col2:
            st.metric("Pandas Version", f"{pd.__version__}")
        with col3:
            st.metric("QuantEdge Version", f"{qs.__version__}")
    
    # Footer
    st.sidebar.divider()
    st.sidebar.info("""
    **Data Sources:** 
    - Futures data: Yahoo Finance
    - Risk-free rate: 2% (annualized)
    
    **Notes:**
    - All returns are daily returns
    - Missing data is forward-filled up to 5 days
    - Extreme returns (>50% daily) are filtered out
    - Analysis requires minimum 10 data points
    
    **Bollinger Bands:**
    - Developed by John Bollinger
    - Used to identify volatility and potential reversal points
    - Band breaches indicate extreme price movements
    """)

if __name__ == "__main__":
    main()
