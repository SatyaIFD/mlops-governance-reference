import pandas as pd
from collections import deque
from datetime import timedelta

class StreamingStateManager:
    """
    In-memory stateful lookback engine for real-time transaction feature engineering.
    Tracks windowed behavioral velocities and relational pass-through dynamics.
    """
    def __init__(self):
        self.state = {}

    def _prune_expired_records(self, account_id, current_time):
        if account_id not in self.state:
            return
        cutoff_time = current_time - timedelta(hours=24)
        while self.state[account_id] and self.state[account_id][0][0] < cutoff_time:
            self.state[account_id].popleft()

    def update_and_enrich(self, transaction: dict) -> dict:
        ts = pd.to_datetime(transaction['Timestamp'])
        sender = str(transaction['Sender_account'])
        receiver = str(transaction['Receiver_account'])
        amount = float(transaction['Amount'])

        if sender not in self.state:
            self.state[sender] = deque()
        if receiver not in self.state:
            self.state[receiver] = deque()

        self._prune_expired_records(sender, ts)
        self._prune_expired_records(receiver, ts)

        h1_cutoff = ts - timedelta(hours=1)
        
        # --- SENDER OUTFLOW VELOCITIES ---
        tx_count_1h = 0
        tx_amount_sum_1h = 0.0
        tx_count_24h = 0
        unique_beneficiaries_24h = set()

        for history_ts, history_amt, role, counterparty in self.state[sender]:
            if role == 'sender':
                tx_count_24h += 1
                unique_beneficiaries_24h.add(counterparty)
                if history_ts >= h1_cutoff:
                    tx_count_1h += 1
                    tx_amount_sum_1h += history_amt

        tx_count_1h += 1
        tx_amount_sum_1h += amount
        tx_count_24h += 1
        unique_beneficiaries_24h.add(receiver)

        beneficiary_diversity_24h = unique_beneficiaries_24h.add(receiver)
        beneficiary_diversity_24h = len(unique_beneficiaries_24h) / tx_count_24h

        inflow_sum_1h = 0.0
        outflow_sum_1h = tx_amount_sum_1h

        for history_ts, history_amt, role, _ in self.state[sender]:
            if role == 'receiver' and history_ts >= h1_cutoff:
                inflow_sum_1h += history_amt

        pass_through_ratio_1h = (2.0 * min(inflow_sum_1h, outflow_sum_1h)) / (inflow_sum_1h + outflow_sum_1h + 1e-5)

        # --- RECEIVER INFLOW VELOCITIES (NEW CRITICAL GRAPH FEATURES) ---
        receiver_inflow_count_1h = 0
        receiver_inflow_sum_1h = 0.0

        for history_ts, history_amt, role, _ in self.state[receiver]:
            if role == 'receiver' and history_ts >= h1_cutoff:
                receiver_inflow_count_1h += 1
                receiver_inflow_sum_1h += history_amt

        receiver_inflow_count_1h += 1
        receiver_inflow_sum_1h += amount

        # Commit states to memory
        self.state[sender].append((ts, amount, 'sender', receiver))
        self.state[receiver].append((ts, amount, 'receiver', sender))

        enriched_payload = transaction.copy()
        enriched_payload.update({
            'tx_count_1h': tx_count_1h,
            'tx_amount_sum_1h': tx_amount_sum_1h,
            'tx_count_24h': tx_count_24h,
            'beneficiary_diversity_24h': beneficiary_diversity_24h,
            'pass_through_ratio_1h': pass_through_ratio_1h,
            'receiver_inflow_count_1h': receiver_inflow_count_1h,
            'receiver_inflow_sum_1h': receiver_inflow_sum_1h
        })

        return enriched_payload