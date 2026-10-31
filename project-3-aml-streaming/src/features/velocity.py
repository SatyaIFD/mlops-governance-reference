from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List

CANONICAL_AML_FEATURES: List[str] = [
    "Amount",
    "pass_through_ratio_1h",
    "is_structuring",
    "velocity_acceleration",
    "fan_out_count_24h",
    "receiver_inflow_count_1h",
    "receiver_inflow_amount_1h",
]


class StreamingStateManager:
    """
    In-memory rolling lookback window manager for stateful streaming features.
    Uses event-time watermarking (rather than wall-clock datetime.now()) and
    event_id deduplication so historical replays and DLQ recoveries preserve
    exact rolling 1h/24h graph features without double-counting.
    """

    def __init__(self):
        self.state: Dict[str, deque] = {}
        self.watermark: datetime | None = None
        self._seen_events: deque[tuple[datetime, str]] = deque()
        self._seen_event_map: Dict[str, Dict[str, Any]] = {}

    def _advance_watermark(self, event_ts: datetime) -> datetime:
        if self.watermark is None or event_ts > self.watermark:
            self.watermark = event_ts
        return self.watermark

    def _prune_seen_events(self, cutoff_24h: datetime) -> None:
        while self._seen_events and self._seen_events[0][0] < cutoff_24h:
            _, old_id = self._seen_events.popleft()
            self._seen_event_map.pop(old_id, None)

    def update_and_enrich(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Updates internal lookback state and appends the 7 canonical graph/velocity features."""
        sender = str(tx["Sender_account"])
        receiver = str(tx["Receiver_account"])
        amount = float(tx["Amount"])
        ts = tx["Timestamp"]
        if isinstance(ts, str):
            ts = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")

        wm = self._advance_watermark(ts)
        cutoff_1h = wm - timedelta(hours=1)
        cutoff_24h = wm - timedelta(hours=24)

        event_id = tx.get("event_id")
        self._prune_seen_events(cutoff_24h)
        if event_id and event_id in self._seen_event_map:
            return self._seen_event_map[event_id].copy()

        if sender not in self.state:
            self.state[sender] = deque()
        if receiver not in self.state:
            self.state[receiver] = deque()

        self.state[sender].append(
            {"ts": ts, "type": "out", "amount": amount, "counterparty": receiver}
        )
        self.state[receiver].append(
            {"ts": ts, "type": "in", "amount": amount, "counterparty": sender}
        )

        # 1. Prune sender state to 24h watermark window and compute sender features
        self._prune_history(sender, cutoff_24h)
        sender_records = self.state.get(sender, ())

        sender_in_1h = sum(
            r["amount"]
            for r in sender_records
            if r["type"] == "in" and r["ts"] >= cutoff_1h
        )
        sender_out_records_1h = [
            r for r in sender_records if r["type"] == "out" and r["ts"] >= cutoff_1h
        ]
        sender_out_records_24h = [
            r for r in sender_records if r["type"] == "out" and r["ts"] >= cutoff_24h
        ]

        sender_out_1h = sum(r["amount"] for r in sender_out_records_1h)
        epsilon = 1e-6
        pass_through_1h = (
            min(1.0, sender_out_1h / (sender_in_1h + epsilon))
            if sender_in_1h > 0.0
            else 0.0
        )

        is_structuring = 1 if 8000.0 <= amount < 10000.0 else 0

        out_count_1h = len(sender_out_records_1h)
        out_count_24h = len(sender_out_records_24h)
        baseline_hourly_rate = out_count_24h / 24.0
        velocity_acceleration = (
            out_count_1h / (baseline_hourly_rate + epsilon)
            if out_count_24h > 0
            else 0.0
        )

        fan_out_count_24h = len(
            {r["counterparty"] for r in sender_out_records_24h if "counterparty" in r}
        )

        # 2. Prune receiver state to 24h watermark window and compute receiver 1h inflow
        self._prune_history(receiver, cutoff_24h)
        receiver_records = self.state.get(receiver, ())
        receiver_in_events_1h = [
            r for r in receiver_records if r["type"] == "in" and r["ts"] >= cutoff_1h
        ]
        rx_inflow_count_1h = len(receiver_in_events_1h)
        rx_inflow_amount_1h = sum(r["amount"] for r in receiver_in_events_1h)

        enriched_tx = tx.copy()
        enriched_tx["Timestamp"] = ts
        enriched_tx["Amount"] = amount
        enriched_tx["pass_through_ratio_1h"] = round(pass_through_1h, 4)
        enriched_tx["is_structuring"] = int(is_structuring)
        enriched_tx["velocity_acceleration"] = round(velocity_acceleration, 4)
        enriched_tx["fan_out_count_24h"] = int(fan_out_count_24h)
        enriched_tx["receiver_inflow_count_1h"] = int(rx_inflow_count_1h)
        enriched_tx["receiver_inflow_amount_1h"] = round(rx_inflow_amount_1h, 2)

        if event_id:
            self._seen_events.append((ts, event_id))
            self._seen_event_map[event_id] = enriched_tx.copy()

        return enriched_tx

    def _prune_history(self, account: str, cutoff_time: datetime) -> None:
        """Removes records older than cutoff_time from an account deque."""
        if account not in self.state:
            return

        while self.state[account] and self.state[account][0]["ts"] < cutoff_time:
            self.state[account].popleft()

        if not self.state[account]:
            del self.state[account]

    def _evict_dormant_accounts(self, cutoff_time: datetime | None = None) -> None:
        """
        Sweeps global state to remove dormant accounts outside the 24h event-time watermark window.
        If cutoff_time is omitted, uses (self.watermark - 24h).
        """
        if cutoff_time is None:
            if self.watermark is None:
                return
            cutoff_time = self.watermark - timedelta(hours=24)

        for account in list(self.state.keys()):
            self._prune_history(account, cutoff_time)
        self._prune_seen_events(cutoff_time)
