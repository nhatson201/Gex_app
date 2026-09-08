import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timezone
import time
import os

# ===================== CẤU HÌNH =====================
BASE_URL = "https://eapi.binance.com"
UNDERLYING = "XAUUSDT"
RISK_FREE_RATE = 0.045
CONTRACT_MULTIPLIER = 1
# =====================================================

st.set_page_config(page_title="GEX Calculator - XAUUSDT", layout="wide")

# === Trang kiểm tra sức khỏe cho UptimeRobot ===
query_params = st.query_params
if "health" in query_params:
    st.write("OK")
    st.stop()

st.title("📊 Gamma Exposure (GEX) — XAUUSDT Options")
st.markdown("---")

# Cache dữ liệu 5 phút
@st.cache_data(ttl=300)
def get_data():
    results = {}

    # ✅ === LẤY GIÁ XAUUSDT — ĐÃ SỬA NGUỒN & TÊN API ===
    S = None
    try:
        # Ưu tiên lấy giá từ API E-Options (cùng nguồn dữ liệu quyền chọn)
        r = requests.get(f"{BASE_URL}/eapi/v1/ticker", params={"symbol": "XAUUSDT"}, timeout=10)
        data = r.json()
        if 'markPrice' in data and float(data['markPrice']) > 0:
            S = float(data['markPrice'])
        elif 'lastPrice' in data and float(data['lastPrice']) > 0:
            S = float(data['lastPrice'])
        if S:
            results['price'] = S
    except Exception as e:
        st.warning(f"⚠️ API E-Options: {e}")

    # Dự phòng: gọi API công khai nếu trên lỗi
    if not S:
        try:
            r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "XAUUSDT"}, timeout=10)
            data = r.json()
            if 'price' in data and float(data['price']) > 0:
                S = float(data['price'])
                results['price'] = S
        except Exception as e2:
            st.error(f"❌ Không lấy được giá: {e2}")
            return None

    if not S:
        st.error("❌ Không xác định được giá XAUUSDT!")
        return None

    # ✅ === LẤY DANH SÁCH HỢP ĐỒNG QUYỀN CHỌN ===
    try:
        r = requests.get(f"{BASE_URL}/eapi/v1/exchangeInfo", timeout=10)
        data = r.json()
        options = []
        for sym in data['optionSymbols']:
            if sym['underlying'] == UNDERLYING and sym['status'] == 'TRADING':
                options.append({
                    'symbol': sym['symbol'],
                    'strike': float(sym['strikePrice']),
                    'expiry': sym['expiryDate'] / 1000,
                    'type': sym['optionSide'],
                    'multiplier': float(sym['contractMultiplier'])
                })
        results['options_count'] = len(options)
        if len(options) == 0:
            st.warning("⚠️ Không tìm thấy hợp đồng quyền chọn nào đang giao dịch!")
    except Exception as e:
        st.error(f"❌ Lỗi lấy danh sách hợp đồng: {e}")
        return None

    # ✅ === HÀM TÍNH GAMMA THEO BLACK-SCHOLES ===
    def gamma(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0.0
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            return norm.pdf(d1) / (S * sigma * np.sqrt(T))
        except:
            return 0.0

    # ✅ === LẤY DỮ LIỆU TỪNG HỢP ĐỒNG & TÍNH GEX ===
    current_ts = time.time()
    gex_list = []
    total_gex = 0.0

    for opt in options:
        T = max(0, (opt['expiry'] - current_ts) / (365 * 24 * 3600))
        if T <= 0:
            continue

        # Lấy Mark Price, Delta, IV
        try:
            mp = requests.get(f"{BASE_URL}/eapi/v1/markPrice", params={"symbol": opt['symbol']}, timeout=10).json()
            iv = float(mp['impliedVolatility']) / 100.0
        except:
            continue

        # Lấy OI & chuyển đổi giá trị USDT → số hợp đồng
        try:
            ticker = requests.get(f"{BASE_URL}/eapi/v1/ticker", params={"symbol": opt['symbol']}, timeout=10).json()
            oi_value = float(ticker.get('openInterest', 0))
            mk_price = float(ticker.get('markPrice', 0))
            oi_contracts = oi_value / mk_price if mk_price > 0 else 0
        except:
            continue

        if oi_contracts <= 0:
            continue

        # Tính GEX
        g = gamma(S, opt['strike'], T, RISK_FREE_RATE, iv)
        mult = opt.get('multiplier', CONTRACT_MULTIPLIER)
        if opt['type'] == 'CALL':
            gex_val = g * oi_contracts * mult * (S ** 2) * 0.01
        else:  # PUT → đảo dấu
            gex_val = -g * oi_contracts * mult * (S ** 2) * 0.01

        total_gex += gex_val
        gex_list.append({
            'Strike': opt['strike'],
            'Type': opt['type'],
            'Expiry_Days': round(T * 365, 1),
            'IV(%)': round(iv * 100, 2),
            'OI_Contracts': round(oi_contracts, 1),
            'Gamma': round(g, 8),
            'GEX': round(gex_val, 2)
        })

    if not gex_list:
        st.warning("⚠️ Chưa có dữ liệu hợp đồng hợp lệ, vui lòng thử lại sau!")
        return None

    # ✅ === TÍNH TỔNG HỢP ===
    df = pd.DataFrame(gex_list)
    df_sorted = df.sort_values('Strike')
    df_sorted['Cumulative_GEX'] = df_sorted['GEX'].cumsum()

    # Tìm Gamma Flip
    gamma_flip = None
    for i in range(1, len(df_sorted)):
        prev = df_sorted.iloc[i-1]['Cumulative_GEX']
        curr = df_sorted.iloc[i]['Cumulative_GEX']
        if (prev < 0 and curr >= 0) or (prev > 0 and curr <= 0):
            gamma_flip = (df_sorted.iloc[i-1]['Strike'] + df_sorted.iloc[i]['Strike']) / 2
            break

    # Tìm Call Wall & Put Wall
    call_wall = df[df['Type'] == 'CALL'].nlargest(1, 'GEX').iloc[0] if len(df[df['Type'] == 'CALL']) else None
    put_wall = df[df['Type'] == 'PUT'].nsmallest(1, 'GEX').iloc[0] if len(df[df['Type'] == 'PUT']) else None

    results['total_gex'] = total_gex
    results['gamma_flip'] = gamma_flip
    results['call_wall'] = call_wall
    results['put_wall'] = put_wall
    results['df'] = df
    results['update_time'] = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')
    return results

# ✅ === HIỂN THỊ GIAO DIỆN ===
data = get_data()

if data:
    st.subheader(f"🕒 Cập nhật: {data['update_time']}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("💰 Giá XAUUSDT", f"{data['price']:.2f}")

    with col2:
        gex_status = "🟢 GEX DƯƠNG" if data['total_gex'] > 0 else "🔴 GEX ÂM"
        st.metric("📈 Tổng GEX", f"{data['total_gex']:,.0f}",
                  help=f"{gex_status} — {'Bình ổn giá' if data['total_gex'] > 0 else 'Khuếch đại biến động'}")

    with col3:
        if data['gamma_flip']:
            pos = "TRÊN → GEX+" if data['price'] > data['gamma_flip'] else "DƯỚI → GEX-"
            st.metric("🔄 Gamma Flip", f"{data['gamma_flip']:.2f}", delta=pos)
        else:
            st.metric("🔄 Gamma Flip", "Không xác định")

    c1, c2 = st.columns(2)
    with c1:
        if data['call_wall'] is not None:
            st.info(f"🧱 **Call Wall (Kháng cự):** {data['call_wall']['Strike']:.2f} | GEX: {data['call_wall']['GEX']:,.0f}")
    with c2:
        if data['put_wall'] is not None:
            st.info(f"🧱 **Put Wall (Hỗ trợ):** {data['put_wall']['Strike']:.2f} | GEX: {data['put_wall']['GEX']:,.0f}")

    st.markdown("---")
    st.subheader("📊 Chi tiết GEX theo mức giá (Top 15)")
    df_show = data['df'].sort_values('GEX', key=abs, ascending=False).head(15).reset_index(drop=True)
    st.dataframe(df_show, use_container_width=True)

    st.markdown("---")
    st.caption(f"Nguồn: Binance Options | Tự động cập nhật mỗi 5 phút | Hợp đồng đang giao dịch: {data['options_count']}")

# Nút làm mới dữ liệu
if st.button("🔄 Làm mới dữ liệu"):
    st.cache_data.clear()
    st.rerun()
