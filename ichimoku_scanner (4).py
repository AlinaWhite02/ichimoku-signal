"""
╔══════════════════════════════════════════════════════════════════╗
║  일목균형표 구름대 시그널 스캐너 (KOSPI + KOSDAQ 전종목)        ║
║  — 종가 vs 구름대 상단 돌파(매수) / 하단 이탈(매도) 판별       ║
║                                                                  ║
║  사용법:                                                        ║
║    pip install pykrx pandas tabulate                            ║
║    python ichimoku_scanner.py                                    ║
║    python ichimoku_scanner.py --date 20260413                   ║
║    python ichimoku_scanner.py --output signals.csv              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pykrx import stock


# ═══════════════════════════════════════════════════════════════════
# 1. 일목균형표 계산
# ═══════════════════════════════════════════════════════════════════

def ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame에 일목균형표 구름대(선행스팬1, 선행스팬2)를 추가.
    - 전환선(9일), 기준선(26일)은 선행스팬 계산용으로만 사용.
    - 선행스팬은 26일 앞으로 시프트하지 않음 (당일 판정 기준).
      → KB증권 HTS와 동일하게, "오늘의 구름"은 26일 전에 계산된 값.
      → pykrx 데이터 기준으로는 shift(26)을 적용.
    """
    high = df["고가"]
    low = df["저가"]

    # 전환선 (9일 중간값)
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2

    # 기준선 (26일 중간값)
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2

    # 선행스팬A: (전환선 + 기준선) / 2 → 26일 선행
    span_a = ((tenkan + kijun) / 2).shift(26)

    # 선행스팬B: 52일 중간값 → 26일 선행
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    df = df.copy()
    df["span_a"] = span_a
    df["span_b"] = span_b
    df["cloud_top"] = df[["span_a", "span_b"]].max(axis=1)
    df["cloud_bot"] = df[["span_a", "span_b"]].min(axis=1)

    return df


# ═══════════════════════════════════════════════════════════════════
# 2. 시그널 판별
# ═══════════════════════════════════════════════════════════════════

def detect_signal(df: pd.DataFrame) -> str | None:
    """
    마지막 2일(전일 vs 당일) 기준으로 구름대 돌파/이탈 판별.
    Returns: "BUY", "SELL", or None
    """
    if len(df) < 2:
        return None

    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    # 구름대 값이 없으면 판별 불가
    if pd.isna(today["cloud_top"]) or pd.isna(yesterday["cloud_top"]):
        return None

    close_today = today["종가"]
    close_yesterday = yesterday["종가"]
    cloud_top_today = today["cloud_top"]
    cloud_bot_today = today["cloud_bot"]
    cloud_top_yesterday = yesterday["cloud_top"]
    cloud_bot_yesterday = yesterday["cloud_bot"]

    # 매수: 전일 종가 ≤ 구름 상단 → 당일 종가 > 구름 상단
    if close_yesterday <= cloud_top_yesterday and close_today > cloud_top_today:
        return "BUY"

    # 매도: 전일 종가 ≥ 구름 하단 → 당일 종가 < 구름 하단
    if close_yesterday >= cloud_bot_yesterday and close_today < cloud_bot_today:
        return "SELL"

    return None


# ═══════════════════════════════════════════════════════════════════
# 3. 전종목 스캔
# ═══════════════════════════════════════════════════════════════════

def get_all_tickers(date_str: str) -> list[tuple[str, str, str]]:
    """
    KOSPI + KOSDAQ 전종목 티커 + 종목명 리스트.
    FinanceDataReader로 종목 목록 조회 (pykrx 벌크 함수 호환 이슈 우회).
    """
    try:
        import FinanceDataReader as fdr
    except ImportError:
        print("  ⚠ FinanceDataReader가 설치되지 않았습니다.")
        print("    설치: py -3.12 -m pip install finance-datareader")
        return []

    tickers = []

    for market in ["KOSPI", "KOSDAQ"]:
        try:
            df = fdr.StockListing(market)
            if df.empty:
                continue

            # FinanceDataReader 버전에 따라 컬럼명이 다를 수 있음
            code_col = "Code" if "Code" in df.columns else "Symbol" if "Symbol" in df.columns else None
            name_col = "Name" if "Name" in df.columns else None

            if code_col is None or name_col is None:
                # 컬럼명을 못 찾으면 첫 번째, 두 번째 컬럼 사용
                code_col = df.columns[0]
                name_col = df.columns[1]

            for _, row in df.iterrows():
                code = str(row[code_col]).strip()
                name = str(row[name_col]).strip()
                # 6자리 숫자 종목코드만 (ETF, ETN 등 제외)
                if len(code) == 6 and code.isdigit():
                    # 시가총액 (억원 단위)
                    mcap = None
                    for mcap_col in ["Marcap", "MarCap", "Market Cap"]:
                        if mcap_col in df.columns:
                            try:
                                mcap = int(row[mcap_col] / 100000000)
                            except Exception:
                                pass
                            break
                    tickers.append((code, name, market, mcap))

        except Exception as e:
            print(f"  ⚠ {market} 종목 목록 조회 실패: {e}")

    return tickers


def fetch_ohlcv(ticker: str, end_date: str, lookback_days: int = 200) -> pd.DataFrame:
    """
    pykrx로 일봉 OHLCV 가져오기.
    일목균형표 계산에 최소 52 + 26 = 78 거래일 필요 → 여유있게 200일.
    """
    start_dt = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=lookback_days * 1.6)
    start_str = start_dt.strftime("%Y%m%d")

    df = stock.get_market_ohlcv(start_str, end_date, ticker)

    if df.empty:
        return df

    # 일자 중복 제거 (간혹 발생)
    df = df[~df.index.duplicated(keep="last")]

    return df


def scan_all(target_date: str, progress: bool = True) -> pd.DataFrame:
    """전종목 스캔 후 시그널 DataFrame 반환."""
    tickers = get_all_tickers(target_date)
    total = len(tickers)

    if total == 0:
        print("  ⚠ 종목 목록을 가져올 수 없습니다. 날짜를 확인해주세요.")
        return pd.DataFrame()

    if progress:
        print(f"  총 {total}개 종목 스캔 시작...\n")

    results = []
    errors = []

    for idx, (ticker, name, market, mcap) in enumerate(tickers, 1):
        if progress and idx % 50 == 0:
            pct = idx / total * 100
            print(f"  [{idx:>4}/{total}] {pct:5.1f}% 처리 중... (현재: {name})")

        try:
            df = fetch_ohlcv(ticker, target_date)

            if len(df) < 80:
                continue  # 데이터 부족 (신규 상장 등)

            df = ichimoku_cloud(df)
            signal = detect_signal(df)

            if signal:
                today = df.iloc[-1]
                yesterday = df.iloc[-2]
                change_pct = (today["종가"] / yesterday["종가"] - 1) * 100

                results.append({
                    "시그널": signal,
                    "종목코드": ticker,
                    "종목명": name,
                    "시장": market,
                    "종가": int(today["종가"]),
                    "전일대비(%)": round(change_pct, 2),
                    "구름상단": int(today["cloud_top"]),
                    "구름하단": int(today["cloud_bot"]),
                    "거래량": int(today["거래량"]),
                    "시가총액": mcap,
                })

            # pykrx rate limit 방지
            time.sleep(0.05)

        except Exception as e:
            errors.append((ticker, name, str(e)))
            time.sleep(0.1)

    if errors and progress:
        print(f"\n  ⚠ {len(errors)}개 종목 데이터 수집 실패 (신규상장/거래정지 등)")

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        # 매수는 거래량 내림차순, 매도도 거래량 내림차순
        result_df = result_df.sort_values(
            ["시그널", "거래량"], ascending=[True, False]
        ).reset_index(drop=True)

    return result_df


# ═══════════════════════════════════════════════════════════════════
# 4. 출력
# ═══════════════════════════════════════════════════════════════════

def print_results(df: pd.DataFrame, target_date: str):
    """결과를 터미널에 보기 좋게 출력."""
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    date_display = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"

    print()
    print("=" * 70)
    print(f"  일목균형표 구름대 시그널 스캔 결과 — {date_display}")
    print("=" * 70)

    if df.empty:
        print("\n  오늘 구름대 돌파/이탈 시그널이 발생한 종목이 없습니다.\n")
        return

    # 매수 시그널
    buys = df[df["시그널"] == "BUY"].copy()
    sells = df[df["시그널"] == "SELL"].copy()

    for label, sub_df, color in [("🟢 매수 시그널 (구름 상단 돌파)", buys, "\033[92m"),
                                   ("🔴 매도 시그널 (구름 하단 이탈)", sells, "\033[91m")]:
        print(f"\n  {label} — {len(sub_df)}건")
        print("  " + "─" * 66)

        if sub_df.empty:
            print("  (해당 없음)")
            continue

        display_df = sub_df.drop(columns=["시그널"]).copy()
        display_df["종가"] = display_df["종가"].apply(lambda x: f"{x:,}")
        display_df["구름상단"] = display_df["구름상단"].apply(lambda x: f"{x:,}")
        display_df["구름하단"] = display_df["구름하단"].apply(lambda x: f"{x:,}")
        display_df["거래량"] = display_df["거래량"].apply(lambda x: f"{x:,}")
        display_df["전일대비(%)"] = display_df["전일대비(%)"].apply(
            lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
        )

        if use_tabulate:
            print(tabulate(
                display_df,
                headers="keys",
                tablefmt="simple",
                showindex=False,
                stralign="right",
            ))
        else:
            print(display_df.to_string(index=False))

    print()
    print(f"  합계: 매수 {len(buys)}건 / 매도 {len(sells)}건")
    print("=" * 70)
    print("  ※ 본 자료는 기술적 지표 참고용이며, 투자 판단은 본인 책임입니다.")
    print("=" * 70)
    print()


# ═══════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="일목균형표 구름대 시그널 전종목 스캐너"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="분석 기준일 (YYYYMMDD). 미입력 시 최근 거래일 자동 탐색.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="결과를 CSV로 저장할 경로 (예: signals.csv)",
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=["ALL", "KOSPI", "KOSDAQ"],
        default="ALL",
        help="스캔 대상 시장 (기본: ALL)",
    )
    parser.add_argument(
        "--json", "-j",
        type=str,
        default=None,
        help="결과를 JSON으로 저장할 경로 (예: data/latest.json)",
    )

    args = parser.parse_args()

    # 기준일 결정
    if args.date:
        target_date = args.date
    else:
        # 삼성전자(005930) 데이터로 최근 거래일 탐색
        today = datetime.now()
        target_date = None
        for delta in range(7):
            candidate = (today - timedelta(days=delta)).strftime("%Y%m%d")
            try:
                df = stock.get_market_ohlcv(candidate, candidate, "005930")
                if not df.empty:
                    target_date = candidate
                    break
            except Exception:
                continue

        if not target_date:
            print("최근 거래일을 찾을 수 없습니다. --date 옵션으로 직접 지정해주세요.")
            sys.exit(1)

    date_display = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"
    print(f"\n  📊 일목균형표 구름대 스캐너 시작")
    print(f"  기준일: {date_display}")
    print(f"  대상: KOSPI + KOSDAQ 전종목\n")

    # 스캔 실행
    start_time = time.time()
    result_df = scan_all(target_date)
    elapsed = time.time() - start_time

    # 시장 필터
    if args.market != "ALL" and not result_df.empty:
        result_df = result_df[result_df["시장"] == args.market].reset_index(drop=True)

    # 결과 출력
    print_results(result_df, target_date)
    print(f"  소요시간: {elapsed:.0f}초")

    # CSV 저장
    if args.output and not result_df.empty:
        output_path = Path(args.output)
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  💾 CSV 저장: {output_path.absolute()}")

    # JSON 저장 (API 서버 연동용)
    if args.json and not result_df.empty:
        json_path = Path(args.json)
        buys = result_df[result_df["시그널"] == "BUY"].to_dict("records")
        sells = result_df[result_df["시그널"] == "SELL"].to_dict("records")

        def remap(row):
            return {
                "code": row["종목코드"], "name": row["종목명"], "market": row["시장"],
                "close": row["종가"], "change": row["전일대비(%)"],
                "cloud_top": row["구름상단"], "cloud_bot": row["구름하단"],
                "volume": row["거래량"], "market_cap": row.get("시가총액"),
            }

        import json
        json_data = {
            "date": date_display,
            "scan_time": time.strftime("%H:%M"),
            "total_scanned": len(get_all_tickers(target_date)),
            "buy": [remap(r) for r in buys],
            "sell": [remap(r) for r in sells],
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"  💾 JSON 저장: {json_path.absolute()}")

    print()


if __name__ == "__main__":
    main()
