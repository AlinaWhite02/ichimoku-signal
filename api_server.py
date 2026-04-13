"""
╔══════════════════════════════════════════════════════════════════╗
║  일목균형표 시그널 API 서버                                       ║
║                                                                  ║
║  사용법:                                                        ║
║    pip install fastapi uvicorn                                  ║
║    python api_server.py                                         ║
║                                                                  ║
║  API:                                                           ║
║    GET /api/signals              → 최신 시그널 조회              ║
║    GET /api/signals?date=20260413 → 특정일 시그널 조회           ║
║    POST /api/scan                → 수동 스캔 실행               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── 스캐너 로직 (ichimoku_scanner.py에서 가져옴) ─────────────────

import pandas as pd
from pykrx import stock

try:
    import FinanceDataReader as fdr
    HAS_FDR = True
except ImportError:
    HAS_FDR = False


def ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
    high, low = df["고가"], df["저가"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df = df.copy()
    df["span_a"] = span_a
    df["span_b"] = span_b
    df["cloud_top"] = df[["span_a", "span_b"]].max(axis=1)
    df["cloud_bot"] = df[["span_a", "span_b"]].min(axis=1)
    return df


def detect_signal(df: pd.DataFrame) -> str | None:
    if len(df) < 2:
        return None
    today, yesterday = df.iloc[-1], df.iloc[-2]
    if pd.isna(today["cloud_top"]) or pd.isna(yesterday["cloud_top"]):
        return None
    if yesterday["종가"] <= yesterday["cloud_top"] and today["종가"] > today["cloud_top"]:
        return "BUY"
    if yesterday["종가"] >= yesterday["cloud_bot"] and today["종가"] < today["cloud_bot"]:
        return "SELL"
    return None


def get_all_tickers() -> list[tuple[str, str, str]]:
    tickers = []
    if HAS_FDR:
        for market in ["KOSPI", "KOSDAQ"]:
            try:
                df = fdr.StockListing(market)
                code_col = "Code" if "Code" in df.columns else df.columns[0]
                name_col = "Name" if "Name" in df.columns else df.columns[1]
                for _, row in df.iterrows():
                    code = str(row[code_col]).strip()
                    name = str(row[name_col]).strip()
                    if len(code) == 6 and code.isdigit():
                        tickers.append((code, name, market))
            except Exception:
                pass
    return tickers


def run_scan(target_date: str) -> dict:
    """전종목 스캔 실행 → JSON dict 반환."""
    tickers = get_all_tickers()
    start_dt = datetime.strptime(target_date, "%Y%m%d") - timedelta(days=320)
    start_str = start_dt.strftime("%Y%m%d")

    buy_signals = []
    sell_signals = []

    for ticker, name, market in tickers:
        try:
            df = stock.get_market_ohlcv(start_str, target_date, ticker)
            if len(df) < 80:
                continue
            df = df[~df.index.duplicated(keep="last")]
            df = ichimoku_cloud(df)
            signal = detect_signal(df)

            if signal:
                today = df.iloc[-1]
                yesterday = df.iloc[-2]
                change_pct = round((today["종가"] / yesterday["종가"] - 1) * 100, 2)

                entry = {
                    "code": ticker,
                    "name": name,
                    "market": market,
                    "close": int(today["종가"]),
                    "change": change_pct,
                    "cloud_top": int(today["cloud_top"]),
                    "cloud_bot": int(today["cloud_bot"]),
                    "volume": int(today["거래량"]),
                }

                if signal == "BUY":
                    buy_signals.append(entry)
                else:
                    sell_signals.append(entry)

            time.sleep(0.05)
        except Exception:
            continue

    # 거래량 내림차순 정렬
    buy_signals.sort(key=lambda x: x["volume"], reverse=True)
    sell_signals.sort(key=lambda x: x["volume"], reverse=True)

    result = {
        "date": f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}",
        "scan_time": datetime.now().strftime("%H:%M"),
        "total_scanned": len(tickers),
        "buy": buy_signals,
        "sell": sell_signals,
    }

    return result


# ── 결과 저장/로드 ───────────────────────────────────────────────

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


def save_result(result: dict):
    date_str = result["date"].replace("-", "")
    path = DATA_DIR / f"signals_{date_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # 최신 결과 심볼릭 저장
    latest = DATA_DIR / "latest.json"
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def load_result(date_str: str = None) -> dict | None:
    if date_str:
        path = DATA_DIR / f"signals_{date_str}.json"
    else:
        path = DATA_DIR / "latest.json"

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ── FastAPI 앱 ────────────────────────────────────────────────────

app = FastAPI(title="일목균형표 시그널 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scan_status = {"running": False, "last_scan": None}


@app.get("/api/signals")
async def get_signals(date: str = None):
    """시그널 조회. date 미지정 시 최신 결과 반환."""
    result = load_result(date)
    if result:
        return result
    return {"error": "해당 날짜의 스캔 결과가 없습니다.", "available": [
        f.stem.replace("signals_", "") for f in DATA_DIR.glob("signals_*.json")
    ]}


@app.post("/api/scan")
async def trigger_scan(background_tasks: BackgroundTasks, date: str = None):
    """수동 스캔 실행 (백그라운드)."""
    if scan_status["running"]:
        return {"status": "already_running", "message": "스캔이 이미 진행 중입니다."}

    if not date:
        today = datetime.now()
        for delta in range(7):
            candidate = (today - timedelta(days=delta)).strftime("%Y%m%d")
            try:
                df = stock.get_market_ohlcv(candidate, candidate, "005930")
                if not df.empty:
                    date = candidate
                    break
            except Exception:
                continue

    if not date:
        return {"status": "error", "message": "거래일을 찾을 수 없습니다."}

    def do_scan(d):
        scan_status["running"] = True
        try:
            result = run_scan(d)
            save_result(result)
            scan_status["last_scan"] = d
        finally:
            scan_status["running"] = False

    background_tasks.add_task(do_scan, date)
    return {"status": "started", "date": date}


@app.get("/api/status")
async def get_status():
    """스캔 상태 확인."""
    return {
        "running": scan_status["running"],
        "last_scan": scan_status["last_scan"],
        "available_dates": sorted([
            f.stem.replace("signals_", "")
            for f in DATA_DIR.glob("signals_*.json")
        ], reverse=True),
    }


# ── 프론트엔드 정적 파일 서빙 ────────────────────────────────────

@app.get("/")
async def serve_index():
    """메인 대시보드 페이지."""
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "index.html 파일을 같은 폴더에 넣어주세요."}


# ── 실행 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║  일목균형표 시그널 API 서버                    ║")
    print("  ║  http://localhost:8000                       ║")
    print("  ║  API: http://localhost:8000/api/signals      ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
