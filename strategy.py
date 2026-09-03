from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Signal:
    side: Literal["LONG", "SHORT"]
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    poc: float
    vah: float
    val: float
    vwap: float
    volume_ratio: float
    reason: str


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP hesablayır.

    Typical Price = (High + Low + Close) / 3

    VWAP = Sum(Typical Price * Volume) / Sum(Volume)
    """
    typical_price = (
        df["high"] + df["low"] + df["close"]
    ) / 3.0

    volume = df["volume"].astype(float)

    cumulative_volume = volume.cumsum().replace(0, np.nan)

    return (
        typical_price * volume
    ).cumsum() / cumulative_volume


def calculate_volume_profile(
    df: pd.DataFrame,
    bins: int = 40,
    value_area_pct: float = 0.70,
) -> tuple[float, float, float]:
    """
    Volume Profile hesablayır.

    Qaytarır:
        POC = Point of Control
        VAH = Value Area High
        VAL = Value Area Low

    Volume barların typical price səviyyəsinə
    yerləşdirilməsi ilə təxmini Volume Profile qurulur.
    """

    if df.empty:
        raise ValueError(
            "Volume Profile üçün ən azı 1 candle lazımdır."
        )

    if bins < 5:
        raise ValueError(
            "bins ən azı 5 olmalıdır."
        )

    if not 0 < value_area_pct <= 1:
        raise ValueError(
            "value_area_pct 0 ilə 1 arasında olmalıdır."
        )

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    typical_price = typical_price.to_numpy(float)

    volume = df["volume"].to_numpy(float)

    price_low = float(df["low"].min())
    price_high = float(df["high"].max())

    # Əgər bütün qiymətlər eynidirsə
    if np.isclose(price_low, price_high):
        return (
            price_low,
            price_high,
            price_low,
        )

    # Price bins
    edges = np.linspace(
        price_low,
        price_high,
        bins + 1,
    )

    # Hər price bin üçün volume
    volume_by_price, _ = np.histogram(
        typical_price,
        bins=edges,
        weights=volume,
    )

    # Bin mərkəzləri
    price_levels = (
        edges[:-1] + edges[1:]
    ) / 2.0

    # POC = ən çox volume olan level
    poc_index = int(
        np.argmax(volume_by_price)
    )

    poc = float(
        price_levels[poc_index]
    )

    total_volume = float(
        volume_by_price.sum()
    )

    if total_volume <= 0:
        return poc, poc, poc

    target_volume = (
        total_volume * value_area_pct
    )

    covered_volume = float(
        volume_by_price[poc_index]
    )

    left_index = poc_index
    right_index = poc_index

    # POC-dan başlayaraq Value Area genişləndirilir
    while (
        covered_volume < target_volume
        and (
            left_index > 0
            or right_index < len(volume_by_price) - 1
        )
    ):

        left_volume = (
            volume_by_price[left_index - 1]
            if left_index > 0
            else -1
        )

        right_volume = (
            volume_by_price[right_index + 1]
            if right_index < len(volume_by_price) - 1
            else -1
        )

        # Daha çox volume olan tərəfə genişlən
        if right_volume >= left_volume:
            right_index += 1

            covered_volume += float(
                volume_by_price[right_index]
            )

        else:
            left_index -= 1

            covered_volume += float(
                volume_by_price[left_index]
            )

    val = float(
        price_levels[left_index]
    )

    vah = float(
        price_levels[right_index]
    )

    return poc, vah, val


def generate_signal(
    df: pd.DataFrame,
    profile_window: int = 100,
    volume_window: int = 20,
    volume_multiplier: float = 1.5,
    level_tolerance: float = 0.0015,
    risk_reward_1: float = 1.0,
    risk_reward_2: float = 2.0,
    swing_window: int = 10,
    sl_buffer: float = 0.0005,
) -> Signal | None:
    """
    XAU/USD 5M signal generator.

    LONG şərtləri:

    1. Qiymət VWAP üzərindədir
    2. Candle bullish-dir
    3. Volume orta volume-dan yüksəkdir
    4. Qiymət VAL və ya POC-a yaxındır

    SHORT şərtləri:

    1. Qiymət VWAP altındadır
    2. Candle bearish-dir
    3. Volume orta volume-dan yüksəkdir
    4. Qiymət VAH və ya POC-a yaxındır

    SL:
        Son swing low/high + buffer

    TP1:
        1R

    TP2:
        2R

    Bu funksiya yalnız signal yaradır.
    Real order açmır.
    """

    required_columns = {
        "high",
        "low",
        "close",
        "volume",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Çatışmayan sütunlar: "
            f"{sorted(missing_columns)}"
        )

    minimum_rows = max(
        profile_window,
        volume_window + 2,
        swing_window + 2,
    )

    if len(df) < minimum_rows:
        return None

    # DataFrame-i kopyalayırıq
    work = df.copy()

    # Index tarixdirsə sıralayırıq
    work = work.sort_index()

    # VWAP
    work["vwap"] = calculate_vwap(work)

    # Son profile_window candle
    profile_data = work.iloc[
        -profile_window:
    ]

    # Volume Profile
    poc, vah, val = calculate_volume_profile(
        profile_data,
        bins=40,
        value_area_pct=0.70,
    )

    # Son candle
    latest = work.iloc[-1]

    # Ondan əvvəlki candle
    previous = work.iloc[-2]

    # Son candle-dan əvvəlki 20 candle
    previous_volumes = work[
        "volume"
    ].iloc[
        -volume_window - 1:-1
    ]

    average_volume = (
        previous_volumes.mean()
    )

    if (
        not np.isfinite(average_volume)
        or average_volume <= 0
    ):
        return None

    current_volume = float(
        latest["volume"]
    )

    volume_ratio = (
        current_volume
        / float(average_volume)
    )

    close = float(
        latest["close"]
    )

    # Əgər open yoxdursa əvvəlki close istifadə olunur
    if "open" in work.columns:
        open_price = float(
            latest["open"]
        )
    else:
        open_price = float(
            previous["close"]
        )

    previous_close = float(
        previous["close"]
    )

    vwap = float(
        latest["vwap"]
    )

    # Price level tolerance
    tolerance = max(
        close * level_tolerance,
        1e-9,
    )

    # Qiymət VAL-a yaxındır?
    near_val = (
        abs(close - val)
        <= tolerance
    )

    # Qiymət POC-a yaxındır?
    near_poc = (
        abs(close - poc)
        <= tolerance
    )

    # Qiymət VAH-a yaxındır?
    near_vah = (
        abs(close - vah)
        <= tolerance
    )

    # Volume confirmation
    high_volume = (
        volume_ratio
        >= volume_multiplier
    )

    # Bullish candle
    bullish_candle = (
        close > open_price
        and close > previous_close
    )

    # Bearish candle
    bearish_candle = (
        close < open_price
        and close < previous_close
    )

    # Son swing üçün məlumat
    recent_data = work.iloc[
        -swing_window - 1:-1
    ]

    swing_low = float(
        recent_data["low"].min()
    )

    swing_high = float(
        recent_data["high"].max()
    )

    # ==========================================================
    # LONG SIGNAL
    # ==========================================================

    long_condition = (
        close > vwap
        and high_volume
        and bullish_candle
        and (
            near_val
            or near_poc
        )
    )

    if long_condition:

        entry = close

        # SL həm swing low,
        # həm də VAL əsasında qoruyucu səviyyə
        stop_reference = min(
            swing_low,
            val,
        )

        stop_loss = (
            stop_reference
            * (1.0 - sl_buffer)
        )

        risk = (
            entry
            - stop_loss
        )

        if risk <= 0:
            return None

        tp1 = (
            entry
            + risk * risk_reward_1
        )

        tp2 = (
            entry
            + risk * risk_reward_2
        )

        reason = (
            "LONG: qiymət VWAP üzərindədir, "
            "volume yüksəkdir və bullish confirmation "
            "VAL/POC yaxınlığında gəlib."
        )

        return Signal(
            side="LONG",
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            poc=poc,
            vah=vah,
            val=val,
            vwap=vwap,
            volume_ratio=volume_ratio,
            reason=reason,
        )

    # ==========================================================
    # SHORT SIGNAL
    # ==========================================================

    short_condition = (
        close < vwap
        and high_volume
        and bearish_candle
        and (
            near_vah
            or near_poc
        )
    )

    if short_condition:

        entry = close

        # SL swing high və VAH əsasında
        stop_reference = max(
            swing_high,
            vah,
        )

        stop_loss = (
            stop_reference
            * (1.0 + sl_buffer)
        )

        risk = (
            stop_loss
            - entry
        )

        if risk <= 0:
            return None

        tp1 = (
            entry
            - risk * risk_reward_1
        )

        tp2 = (
            entry
            - risk * risk_reward_2
        )

        reason = (
            "SHORT: qiymət VWAP altındadır, "
            "volume yüksəkdir və bearish confirmation "
            "VAH/POC yaxınlığında gəlib."
        )

        return Signal(
            side="SHORT",
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            poc=poc,
            vah=vah,
            val=val,
            vwap=vwap,
            volume_ratio=volume_ratio,
            reason=reason,
        )

    # Heç bir signal yoxdursa
    return None


def signal_to_dict(
    signal: Signal | None,
) -> dict | None:
    """
    Signal obyektini Telegram/API üçün
    dictionary formatına çevirir.
    """

    if signal is None:
        return None

    return {
        "side": signal.side,
        "entry": round(signal.entry, 2),
        "stop_loss": round(
            signal.stop_loss,
            2,
        ),
        "tp1": round(
            signal.tp1,
            2,
        ),
        "tp2": round(
            signal.tp2,
            2,
        ),
        "poc": round(
            signal.poc,
            2,
        ),
        "vah": round(
            signal.vah,
            2,
        ),
        "val": round(
            signal.val,
            2,
        ),
        "vwap": round(
            signal.vwap,
            2,
        ),
        "volume_ratio": round(
            signal.volume_ratio,
            2,
        ),
        "reason": signal.reason,
    }


def format_signal(
    signal: Signal,
) -> str:
    """
    Telegram mesajını hazırlayır.
    """

    emoji = (
        "🟢"
        if signal.side == "LONG"
        else "🔴"
    )

    return f"""
{emoji} XAU/USD {signal.side}

⏱ Timeframe: 5M

🎯 Entry: {signal.entry:.2f}

🛑 SL: {signal.stop_loss:.2f}

✅ TP1: {signal.tp1:.2f}

🚀 TP2: {signal.tp2:.2f}

━━━━━━━━━━━━━━

📊 Volume: {signal.volume_ratio:.2f}x

📍 POC: {signal.poc:.2f}

🔺 VAH: {signal.vah:.2f}

🔻 VAL: {signal.val:.2f}

〰️ VWAP: {signal.vwap:.2f}

━━━━━━━━━━━━━━

🧠 Strategy:
{signal.reason}

⚠️ Signal only — no real order.
""".strip()
