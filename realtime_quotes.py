#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股实时看盘 · 单文件应用（Streamlit + AkShare + Plotly）

功能：
- 自选股实时盘口快照（东财源，秒级延时）
- 单只股票当日1分钟K线 + VWAP 曲线
- 可设置自动刷新频率、切换展示列、导出当前表格

依赖：
  pip install -U streamlit akshare plotly pandas streamlit-autorefresh

运行：
  streamlit run realtime_quotes.py

注意：
- 数据来源为东财公开接口，经 AkShare 抓取，通常有 1-5 秒延迟；量化/毫秒级、逐笔或L2需券商/交易所授权接口。
- 仅用于学习研究，不构成任何投资建议。
"""
from __future__ import annotations
import io
from datetime import datetime
from typing import List

import akshare as ak
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --------------------------- 基础设置 ---------------------------
st.set_page_config(page_title="A股实时看盘 • AkShare", layout="wide")
st.title("A股实时看盘 · AkShare")
st.caption("数据源：东方财富（AkShare 封装） · 教学用途 · 可能存在短暂延迟")

# 默认自选（全部为沪深主板）：工业富联、中国船舶、浪潮信息、国机通用、中国中车
DEFAULT_WATCHLIST = "601138, 600150, 000977, 600444, 601766"

# --------------------------- 侧边栏 ---------------------------
with st.sidebar:
    st.header("设置")
    codes_input = st.text_area(
        "自选股（逗号/空格分隔，支持 6 位代码或含 sh/sz 前缀）",
        value=DEFAULT_WATCHLIST,
        height=80,
    )
    refresh_sec = st.slider("自动刷新频率（秒）", min_value=2, max_value=30, value=5, step=1)
    st.checkbox("自动刷新", value=True, key="auto_refresh")

    show_cols = st.multiselect(
        "表格列选择",
        [
            "代码",
            "名称",
            "最新价",
            "涨跌幅",
            "涨跌额",
            "今开",
            "最高",
            "最低",
            "昨收",
            "成交量",
            "成交额",
            "换手率",
            "量比",
            "市盈率-动态",
            "市净率",
            "总市值",
            "流通市值",
        ],
        default=["代码", "名称", "最新价", "涨跌幅", "成交额", "换手率", "总市值"],
    )

    st.divider()
    st.markdown("**图表**")
    chart_symbol = st.text_input(
        "单图查看（输入 6 位代码或带前缀：如 601138 / sh601138）",
        value="601138",
        help="用于展示当日 1 分钟K线与 VWAP",
    )
    chart_period = st.selectbox("K线周期", options=["1"], index=0, help="当前示例仅演示 1 分钟")

# 自动刷新
if st.session_state.get("auto_refresh", True):
    st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh_key")

# --------------------------- 工具函数 ---------------------------

def parse_codes(raw: str) -> List[str]:
    # 接受格式："601138, sh600000  000977" => ["601138","600000","000977"]
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace("
", " ").replace("，", ",").replace(" ", ",").split(",") if p.strip()]
    normalized = []
    for p in parts:
        p = p.lower()
        if p.startswith("sh") or p.startswith("sz"):
            p = p[-6:]
        # 保留最后6位数字
        digits = "".join([c for c in p if c.isdigit()])
        if len(digits) >= 6:
            normalized.append(digits[-6:])
    # 去重，保持顺序
    seen = set()
    dedup = []
    for c in normalized:
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    return dedup


def add_exchange_prefix(code6: str) -> str:
    # 6 开头走上交所 sh，其余常见主板代码走深交所 sz
    return ("sh" + code6) if code6.startswith("6") else ("sz" + code6)


@st.cache_data(ttl=2)
def load_spot_all() -> pd.DataFrame:
    """拉取全市场快照（东财）"""
    df = ak.stock_zh_a_spot_em()
    # 统一列名，便于选择
    return df


def get_watchlist_snapshot(codes6: List[str]) -> pd.DataFrame:
    if not codes6:
        return pd.DataFrame()
    spot = load_spot_all()
    # 兼容列名（AkShare可能随版本调整，这里做兜底）
    if "代码" not in spot.columns:
        # 兼容老版本 '代码' 可能叫 '股票代码' 等
        code_col = [c for c in spot.columns if "代码" in c][0]
        spot.rename(columns={code_col: "代码"}, inplace=True)

    sub = spot[spot["代码"].isin(codes6)].copy()
    # 排序按涨跌幅
    if "涨跌幅" in sub.columns:
        sub.sort_values("涨跌幅", ascending=False, inplace=True)
    return sub.reset_index(drop=True)


@st.cache_data(ttl=2)
def load_minute_df(symbol_with_prefix: str, period: str = "1") -> pd.DataFrame:
    """当日分钟线（东财）"""
    # Eastmoney 分钟函数
    try:
        df = ak.stock_zh_a_minute_em(symbol=symbol_with_prefix, period=period, adjust="")
    except Exception as e:
        # 兼容函数名可能变动
        df = ak.stock_zh_a_minute(symbol=symbol_with_prefix, period=period, adjust="")
    # 统一列名
    rename_map = {
        "时间": "时间",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "volume": "成交量",
        "amount": "成交额",
    }
    for k, v in list(rename_map.items()):
        if k in df.columns and v != k:
            df.rename(columns={k: v}, inplace=True)
    # 计算 VWAP（逐步累计）
    if set(["收盘", "成交量"]).issubset(df.columns):
        vol = df["成交量"].astype(float).clip(lower=0.0)
        price = df["收盘"].astype(float)
        cum_vol = vol.cumsum()
        df["VWAP"] = (price * vol).cumsum() / cum_vol.replace(0, pd.NA)
    return df


# --------------------------- 主视图：表格 ---------------------------
codes6 = parse_codes(codes_input)
try:
    snap_df = get_watchlist_snapshot(codes6)
except Exception as e:
    st.error(f"拉取快照失败：{e}")
    snap_df = pd.DataFrame()

left, right = st.columns([2, 1])
with left:
    st.subheader("自选股快照")
    if snap_df.empty:
        st.info("请在左侧输入 6 位代码或带前缀的代码，如 601138 / sh601138 / 000977 / sz000977")
    else:
        cols = [c for c in show_cols if c in snap_df.columns]
        view = snap_df[cols].copy() if cols else snap_df.copy()
        # 美化百分比列
        for col in [c for c in view.columns if "涨跌幅" in c or "换手率" in c]:
            with pd.option_context('mode.chained_assignment', None):
                view[col] = pd.to_numeric(view[col], errors='coerce')
        st.dataframe(view, use_container_width=True)

        # 导出按钮
        csv_bytes = view.to_csv(index=False).encode("utf-8-sig")
        st.download_button("导出当前表格 CSV", data=csv_bytes, file_name="watchlist_snapshot.csv", mime="text/csv")

with right:
    if not snap_df.empty:
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.metric("最后刷新", t)

# --------------------------- 图表：分钟K + VWAP ---------------------------
if chart_symbol:
    code6 = parse_codes(chart_symbol)
    if code6:
        sym = add_exchange_prefix(code6[0])
        st.subheader(f"{sym} 当日 {chart_period} 分钟K + VWAP")
        try:
            mdf = load_minute_df(sym, period=chart_period)
            if mdf.empty:
                st.warning("暂无分钟数据（可能未开盘或接口暂不可用）")
            else:
                # 仅展示当日
                # 有的接口会返回带日期的字符串，我们直接展示全量
                fig = go.Figure()
                fig.add_trace(
                    go.Candlestick(
                        x=mdf["时间"],
                        open=mdf["开盘"],
                        high=mdf["最高"],
                        low=mdf["最低"],
                        close=mdf["收盘"],
                        name="K线",
                        showlegend=True,
                    )
                )
                if "VWAP" in mdf.columns:
                    fig.add_trace(
                        go.Scatter(x=mdf["时间"], y=mdf["VWAP"], mode="lines", name="VWAP")
                    )
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # 成交量柱状
                if "成交量" in mdf.columns:
                    vol_fig = go.Figure()
                    vol_fig.add_trace(go.Bar(x=mdf["时间"], y=mdf["成交量"], name="成交量"))
                    vol_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(vol_fig, use_container_width=True)
        except Exception as e:
            st.error(f"分钟数据获取失败：{e}")
    else:
        st.info("请输入有效的 6 位代码或带前缀代码")

st.caption(
    "提示：如果你需要撮合队列、逐笔成交、委托档位等更细的实时数据，请使用券商开放接口（WebSocket/HTTP）或 Tushare Pro/JQData 等付费数据源。"
)
