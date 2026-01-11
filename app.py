import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import timedelta

st.set_page_config(layout="wide", page_title="US Stock Monitor v3.1")

# ==========================================
# 1. ロジック関数群
# ==========================================

def calculate_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Spike'] = df['Volume'] > (df['Vol_MA20'] * 2.0)
    return df

def find_horizontal_levels(df, window=10, touch_threshold=2):
    levels = []
    calc_df = df.tail(300)
    highs = calc_df['High'].rolling(window=window*2+1, center=True).max()
    lows = calc_df['Low'].rolling(window=window*2+1, center=True).min()
    pivots = pd.concat([calc_df[calc_df['High'] == highs]['High'], 
                        calc_df[calc_df['Low'] == lows]['Low']]).sort_values()
    if len(pivots) == 0: return []
    
    threshold_pct = 0.015
    current_level = [pivots.iloc[0]]
    for price in pivots.iloc[1:]:
        avg = sum(current_level) / len(current_level)
        if abs(price - avg) / avg <= threshold_pct:
            current_level.append(price)
        else:
            if len(current_level) >= touch_threshold:
                levels.append(sum(current_level) / len(current_level))
            current_level = [price]
    if len(current_level) >= touch_threshold:
        levels.append(sum(current_level) / len(current_level))
    return levels

def find_trendline(df_in, line_type='resistance', lookback=100, restrict_slope=True):
    df = df_in.tail(lookback).copy()
    if len(df) < 10: return None
    
    if line_type == 'resistance':
        target_col = 'High'
        check_func = lambda p, l, buf: p > (l + buf)
    else:
        target_col = 'Low'
        check_func = lambda p, l, buf: p < (l - buf)

    pivot_window = 3 if lookback < 150 else 5
    rolling_max = df[target_col].rolling(window=pivot_window*2+1, center=True).max()
    rolling_min = df[target_col].rolling(window=pivot_window*2+1, center=True).min()
    pivots = df[df[target_col] == rolling_max] if line_type == 'resistance' else df[df[target_col] == rolling_min]
    
    points = []
    for date, row in pivots.iterrows():
        idx = df_in.index.get_loc(date)
        val = row[target_col]
        points.append((idx, val))
        
    if len(points) < 2: return None

    best_line = None
    max_score = -1
    for i in range(len(points) - 1, 0, -1):
        idx_b, val_b = points[i]
        for j in range(i - 1, -1, -1):
            idx_a, val_a = points[j]
            duration = idx_b - idx_a
            if duration < 10: continue
            slope = (val_b - val_a) / duration
            
            if restrict_slope:
                if line_type == 'resistance' and slope > 0: continue
                if line_type == 'support' and slope < 0: continue

            intercept = val_a - slope * idx_a
            collision = False
            check_start = idx_a + 1
            check_end = idx_b - 1
            if check_start <= check_end:
                actual = df_in[target_col].iloc[check_start:check_end+1].values
                x_range = np.arange(check_start, check_end+1)
                line_vals = slope * x_range + intercept
                buffer = actual * 0.002
                if check_func(actual, line_vals, buffer).any():
                    collision = True
            
            if not collision:
                recency_bonus = idx_b * 2 
                score = duration + recency_bonus
                if score > max_score:
                    max_score = score
                    full_x = np.arange(len(df_in))
                    best_line = slope * full_x + intercept
    return best_line

# ==========================================
# 2. データ取得 & 分析
# ==========================================
@st.cache_data(ttl=3600)
def fetch_ultimate_data(tickers):
    results = {}
    for ticker in tickers:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = calculate_technical_indicators(df)
        
        horizontal = find_horizontal_levels(df, window=5)
        short_res = find_trendline(df, 'resistance', lookback=150, restrict_slope=True)
        short_sup = find_trendline(df, 'support',    lookback=150, restrict_slope=True)
        long_res = find_trendline(df, 'resistance', lookback=500, restrict_slope=False)
        long_sup = find_trendline(df, 'support',    lookback=500, restrict_slope=False)
        
        curr = df['Close'].iloc[-1]
        breakout_flags = []
        
        for lvl in horizontal:
            if lvl > curr * 0.90 and curr >= lvl and curr <= lvl * 1.03:
                breakout_flags.append(f"Horizontal")
        if short_res is not None and curr >= short_res[-1] and curr <= short_res[-1] * 1.03:
            breakout_flags.append(f"Trend")
            
        max_high = df['High'].max()
        is_ath = curr >= (max_high * 0.99)
        
        results[ticker] = {
            "df": df, "current_price": curr, "horizontal": horizontal,
            "short_res": short_res, "short_sup": short_sup,
            "long_res": long_res, "long_sup": long_sup,
            "breakout_flags": breakout_flags,
            "is_ath": is_ath
        }
    return results

# ==========================================
# 3. UI & 描画
# ==========================================
with st.sidebar:
    st.header("Settings")
    col_num = st.slider("列数 (Columns)", 1, 4, 3)
    chart_height = st.slider("チャートの高さ (px)", 300, 1000, 300)
    lookback_option = st.selectbox("表示期間 (Zoom)", ["3 Months", "6 Months", "1 Year", "2 Years"], index=2)
    
    default_tickers = "SPY, QQQ, NVDA, TSLA, MSFT, AMZN, GOOGL, META, NFLX, COIN, MSTR"
    ticker_input = st.text_area("監視銘柄リスト", default_tickers)
    tickers = [t.strip() for t in ticker_input.split(',')]
    
    st.divider()
    st.markdown("""
    - 🟦 **Blue Line**: SMA 20 (短期)
    - 🟠 **Orange Line**: SMA 50 (中期)
    - 🟣 **Purple Line**: SMA 200 (長期)
    - 🟩 **Green BG**: 買い集め (Volume Spike)
    - 🟥 **Red BG**: 売り抜け (Volume Spike)
    - 💥 **Annotation**: ブレイクポイント
    - 🏆 **ATH**: 過去最高値更新中
    """)
    st.divider()
    refresh = st.button("🔄 データを更新", type="primary")

st.title("US Stock Monitor v3.1")

if refresh or 'ultimate_data_v3' not in st.session_state:
    with st.spinner('Analyzing Markets & Checking ATH...'):
        st.session_state['ultimate_data_v3'] = fetch_ultimate_data(tickers)

if 'ultimate_data_v3' in st.session_state:
    data = st.session_state['ultimate_data_v3']
    
    alerts = [t for t, d in data.items() if d['breakout_flags']]
    if alerts:
        st.success(f"🔥 **BREAKOUT ALERT:** {', '.join(alerts)}")
    
    cols = st.columns(col_num)
    
    for i, ticker in enumerate(tickers):
        if ticker not in data: continue
        item = data[ticker]
        col_index = i % col_num
        
        with cols[col_index]:
            df = item['df']
            
            # 期間計算
            days_map = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730}
            visible_days = days_map[lookback_option]
            start_date = df.index[-1] - timedelta(days=visible_days)
            visible_df = df[df.index >= start_date]
            
            if not visible_df.empty:
                y_min = visible_df['Low'].min()
                y_max = visible_df['High'].max()
                vol_max = visible_df['Volume'].max()
            else:
                y_min, y_max, vol_max = 0, 100, 1000

            # ==================================================
            # ★ 修正ポイント: 休場日（土日祝）の特定
            # ==================================================
            # 1. 全期間の日付リストを作成
            all_dates = pd.date_range(start=visible_df.index[0], end=visible_df.index[-1])
            # 2. データが存在する日付リストを取得
            existing_dates = visible_df.index
            # 3. 差分（データがない日＝土日祝）をリスト化
            # 文字列型に変換して比較するのが確実
            dt_breaks = [d.strftime("%Y-%m-%d") for d in all_dates if d.strftime("%Y-%m-%d") not in existing_dates.strftime("%Y-%m-%d")]

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.05, row_heights=[0.75, 0.25]
            )
            
            # 1. ローソク足
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price',
                increasing_line_width=1, decreasing_line_width=1,
            ), row=1, col=1)
            
            # 2. SMA
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='#00E5FF', width=1), name='SMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='purple', width=1), name='SMA 200'), row=1, col=1)
            
            # 3. ライン
            for lvl in item['horizontal']:
                if abs(item['current_price'] - lvl) / lvl < 0.3:
                    color = '#00C853' if lvl < item['current_price'] else '#D50000'
                    fig.add_hline(y=lvl, line_width=1, line_dash="dot", line_color=color, opacity=0.6, row=1, col=1)
            
            if item['short_res'] is not None:
                fig.add_trace(go.Scatter(x=df.index, y=item['short_res'], mode='lines', 
                                         line=dict(color='rgba(235, 70, 70, 1)', width=1, dash='dash'), name='Short Res'), row=1, col=1)
            if item['short_sup'] is not None:
                fig.add_trace(go.Scatter(x=df.index, y=item['short_sup'], mode='lines', 
                                         line=dict(color='rgba(70, 70, 235, 1)', width=1, dash='dash'), name='Short Sup'), row=1, col=1)
            if item['long_res'] is not None:
                fig.add_trace(go.Scatter(x=df.index, y=item['long_res'], mode='lines', 
                                         line=dict(color='rgba(180, 0, 0, 0.3)', width=1.5), name='Major Res'), row=1, col=1)
            if item['long_sup'] is not None:
                fig.add_trace(go.Scatter(x=df.index, y=item['long_sup'], mode='lines', 
                                         line=dict(color='rgba(0, 0, 180, 0.3)', width=1.5), name='Major Sup'), row=1, col=1)

            # 4. 出来高
            colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(
                x=df.index, y=df['Volume'], marker_color=colors, name='Volume', marker_line_width=0
            ), row=2, col=1)
            
            # 5. 背景強調
            spike_dates = visible_df[visible_df['Vol_Spike']]
            for date, row in spike_dates.iterrows():
                bg_color = "rgba(0, 200, 83, 0.08)" if row['Close'] >= row['Open'] else "rgba(213, 0, 0, 0.08)"
                fig.add_vrect(x0=date - timedelta(hours=12), x1=date + timedelta(hours=12),
                              fillcolor=bg_color, layer="below", line_width=0, row=1, col=1)

            # 6. 注釈
            curr = item['current_price']
            last_date = df.index[-1]
            if item['short_res'] is not None:
                res_val = item['short_res'][-1]
                if curr >= res_val and curr <= res_val * 1.03:
                    fig.add_annotation(
                        x=last_date, y=curr, text="BREAK!", 
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, ax=-30, ay=-30,
                        bgcolor="#D50000", bordercolor="white", font=dict(color="white", size=10),
                        row=1, col=1
                    )
            for lvl in item['horizontal']:
                 if lvl > curr * 0.90 and curr >= lvl and curr <= lvl * 1.03:
                    fig.add_annotation(
                        x=last_date, y=curr, text="Lv Break", 
                        showarrow=True, arrowhead=1, ax=30, ay=-20,
                        bgcolor="#00C853", bordercolor="white", font=dict(color="white", size=9),
                        row=1, col=1
                    )

            title_text = f"{ticker} : ${curr:.2f}"
            if item['is_ath']: title_text += " 🏆 ATH"
            if item['breakout_flags']: title_text = "🔥 " + title_text
            
            fig.update_layout(
                title=dict(text=title_text, font=dict(size=16)),
                height=chart_height,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
                hovermode="x unified",
                xaxis=dict(rangeslider=dict(visible=False)),
                xaxis2=dict(rangeslider=dict(visible=False)),
            )
            
            # ★ 修正ポイント: 休場日をrangebreaksに渡して非表示にする
            fig.update_xaxes(
                range=[start_date, last_date], 
                rangebreaks=[dict(values=dt_breaks)], # これで完全に詰まる
                showgrid=False, row=1, col=1, visible=False
            )
            fig.update_xaxes(
                range=[start_date, last_date], 
                rangebreaks=[dict(values=dt_breaks)], # 下の軸も詰める
                showgrid=False, row=2, col=1
            )
            
            fig.update_yaxes(range=[y_min*0.95, y_max*1.05], showgrid=True, gridwidth=0.5, gridcolor='lightgray', row=1, col=1)
            fig.update_yaxes(range=[0, vol_max*1.2], showgrid=False, visible=False, row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)