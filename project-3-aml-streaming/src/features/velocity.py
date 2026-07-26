from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List


class StreamingStateManager:
    """In-memory rolling lookback window manager with TTL key eviction.

    Maintains rolling state queues per account to track transaction velocity,
    pass-through ratios, structuring patterns, and fan-out dispersion over 1h and 24h windows.
    Includes automated TTL eviction for inactive account keys to prevent memory leaks.

    Attributes:
        state (Dict[str, deque]): Mapping of account ID to a queue of transaction records.
        last_eviction_ts (datetime): Timestamp of the last full account sweep.
    """

    def __init__(self):
        """Initializes state store and sets the eviction tracker."""
        self.state: Dict[str, deque] = {}
        self.last_eviction_ts: datetime | None = None

    def update_and_enrich(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Updates account queues, evicts dormant keys, and calculates graph features.

        Args:
            tx (Dict[str, Any]): Validated transaction record containing 'Timestamp',
                'Sender_account', 'Receiver_account', and 'Amount'.

        Returns:
            Dict[str, Any]: Transaction enriched with graph and velocity features.
        """
        sender = tx['Sender_account']
        receiver = tx['Receiver_account']
        amount = float(tx['Amount'])
        ts_raw = tx['Timestamp']

        # Defensive timestamp parsing: Ensure ts is a datetime instance
        if isinstance(ts_raw, str):
            ts = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
                try:
                    ts = datetime.strptime(ts_raw, fmt)
                    break
                except ValueError:
                    continue
            if ts is None:
                ts = datetime.now()
        else:
            ts = ts_raw

        # Periodic Sweep: Evict dormant account keys every 6 hours of stream time
        if self.last_eviction_ts is None or (ts - self.last_eviction_ts) > timedelta(hours=6):
            self._evict_dormant_accounts(cutoff_time=ts - timedelta(hours=24))
            self.last_eviction_ts = ts

        # Ensure state queues exist
        if sender not in self.state:
            self.state[sender] = deque()
        if receiver not in self.state:
            self.state[receiver] = deque()

        # Append new events
        self.state[sender].append({'ts': ts, 'type': 'out', 'amount': amount, 'target': receiver})
        self.state[receiver].append({'ts': ts, 'type': 'in', 'amount': amount, 'target': sender})

        # Define time boundaries
        cutoff_24h = ts - timedelta(hours=24)
        cutoff_1h = ts - timedelta(hours=1)

        # 1. Prune and compute sender metrics
        self._prune_history(sender, cutoff_24h)
        sender_events_24h = list(self.state[sender])
        sender_events_1h = [r for r in sender_events_24h if r['ts'] >= cutoff_1h]

        sender_in_1h = sum(r['amount'] for r in sender_events_1h if r['type'] == 'in')
        sender_out_1h = sum(r['amount'] for r in sender_events_1h if r['type'] == 'out')
        sender_out_count_1h = sum(1 for r in sender_events_1h if r['type'] == 'out')
        sender_out_count_24h = sum(1 for r in sender_events_24h if r['type'] == 'out')

        # Pass-Through Ratio (1h)
        epsilon = 1e-6
        pass_through_1h = min(1.0, sender_out_1h / (sender_in_1h + epsilon)) if sender_in_1h > 0 else 0.0

        # Structuring Flag ($8,000 <= Amount < $10,000 CTR threshold)
        is_structuring = 1 if 8000.0 <= amount < 10000.0 else 0

        # Velocity Acceleration (1h rate vs 24h average hourly rate)
        hourly_24h_avg = (sender_out_count_24h / 24.0) + epsilon
        velocity_acceleration = round(sender_out_count_1h / hourly_24h_avg, 4)

        # Fan-Out Count (Unique receivers in 24h)
        unique_receivers_24h = len({r['target'] for r in sender_events_24h if r['type'] == 'out'})

        # 2. Prune and compute receiver metrics
        self._prune_history(receiver, cutoff_24h)
        receiver_events_1h = [r for r in self.state[receiver] if r['type'] == 'in' and r['ts'] >= cutoff_1h]
        rx_inflow_count_1h = len(receiver_events_1h)
        rx_inflow_amount_1h = sum(r['amount'] for r in receiver_events_1h)

        # Enriched output
        enriched_tx = tx.copy()
        enriched_tx['Timestamp'] = ts
        enriched_tx['pass_through_ratio_1h'] = round(pass_through_1h, 4)
        enriched_tx['is_structuring'] = is_structuring
        enriched_tx['velocity_acceleration'] = velocity_acceleration
        enriched_tx['fan_out_count_24h'] = unique_receivers_24h
        enriched_tx['receiver_inflow_count_1h'] = rx_inflow_count_1h
        enriched_tx['receiver_inflow_amount_1h'] = round(rx_inflow_amount_1h, 2)

        return enriched_tx

    def _prune_history(self, account: str, cutoff_time: datetime) -> None:
        """Drops records older than cutoff_time from an account deque."""
        while self.state[account] and self.state[account][0]['ts'] < cutoff_time:
            self.state[account].popleft()

    def _evict_dormant_accounts(self, cutoff_time: datetime) -> None:
        """Evicts account keys with no activity in the last 24 hours to prevent memory leaks."""
        dormant_keys: List[str] = []
        for account, history in self.state.items():
            while history and history[0]['ts'] < cutoff_time:
                history.popleft()
            if not history:
                dormant_keys.append(account)

        for key in dormant_keys:
            del self.state[key]