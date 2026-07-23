from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List


class StreamingStateManager:
    """In-memory rolling lookback window manager for stateful streaming features.

    Maintains rolling state queues per account to track transaction velocity,
    pass-through ratios, and beneficiary inflow patterns over time-based windows (1h, 24h).

    Attributes:
        state (Dict[str, deque]): Mapping of account ID to a double-ended queue of transaction records.
    """

    def __init__(self):
        """Initializes an empty account state store."""
        self.state: Dict[str, deque] = {}

    def update_and_enrich(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Updates internal lookback state and appends engineered graph features to the event.

        Args:
            tx (Dict[str, Any]): Validated transaction record containing 'Timestamp',
                'Sender_account', 'Receiver_account', and 'Amount'.

        Returns:
            Dict[str, Any]: Copy of the transaction enriched with 'pass_through_ratio_1h',
                'receiver_inflow_count_1h', and 'receiver_inflow_amount_1h'.
        """
        sender = tx['Sender_account']
        receiver = tx['Receiver_account']
        amount = float(tx['Amount'])
        ts = tx['Timestamp']

        # Ensure deque state exists for sender and receiver
        if sender not in self.state:
            self.state[sender] = deque()
        if receiver not in self.state:
            self.state[receiver] = deque()

        # Append new transaction event to sender's outgoing and receiver's incoming history
        self.state[sender].append({'ts': ts, 'type': 'out', 'amount': amount})
        self.state[receiver].append({'ts': ts, 'type': 'in', 'amount': amount})

        # Calculate time-windowed features
        cutoff_1h = ts - timedelta(hours=1)
        
        # 1. Prune sender state and compute 1h pass-through
        self._prune_history(sender, ts - timedelta(hours=24))
        sender_in_1h = sum(r['amount'] for r in self.state[sender] if r['type'] == 'in' and r['ts'] >= cutoff_1h)
        sender_out_1h = sum(r['amount'] for r in self.state[sender] if r['type'] == 'out' and r['ts'] >= cutoff_1h)
        
        epsilon = 1e-6
        pass_through_1h = min(1.0, sender_out_1h / (sender_in_1h + epsilon)) if sender_in_1h > 0 else 0.0

        # 2. Prune receiver state and compute 1h inflow counts/volume
        self._prune_history(receiver, ts - timedelta(hours=24))
        receiver_in_events_1h = [r for r in self.state[receiver] if r['type'] == 'in' and r['ts'] >= cutoff_1h]
        rx_inflow_count_1h = len(receiver_in_events_1h)
        rx_inflow_amount_1h = sum(r['amount'] for r in receiver_in_events_1h)

        # Build enriched payload
        enriched_tx = tx.copy()
        enriched_tx['pass_through_ratio_1h'] = round(pass_through_1h, 4)
        enriched_tx['receiver_inflow_count_1h'] = rx_inflow_count_1h
        enriched_tx['receiver_inflow_amount_1h'] = round(rx_inflow_amount_1h, 2)

        return enriched_tx

    def _prune_history(self, account: str, cutoff_time: datetime) -> None:
        """Removes records older than the maximum lookback threshold (24h) from an account deque.

        Args:
            account (str): Account identifier to prune.
            cutoff_time (datetime): Timestamp threshold; events before this time are dropped.
        """
        while self.state[account] and self.state[account][0]['ts'] < cutoff_time:
            self.state[account].popleft()