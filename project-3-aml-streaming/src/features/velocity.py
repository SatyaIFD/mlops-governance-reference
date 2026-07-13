import pandas as pd
from collections import deque
from datetime import timedelta

class StreamingStateManager:
    """
    In-memory stateful lookback engine for real-time transaction feature engineering.
    Tracks windowed behavioral velocities and relational pass-through dynamics.
    """
    def __init__(self):
        # Maps account_id -> deque of tuples: (timestamp, amount, role, counterparty)
        # role can be 'sender' (outflow) or 'receiver' (inflow)
        self.state = {}

    def _prune_expired_records(self, account_id, current_time):
        """Removes entries older than 24 hours for a given account to prevent memory leaks."""
        if account_id not in self.state:
            return
        
        cutoff_time = current_time - timedelta(hours=24)
        # Keep only elements within the 24-hour lookback boundary
        while self.state[account_id] and self.state[account_id][0][0] < cutoff_time:
            self.state[account_id].popleft()

    def update_and_enrich(self, transaction: dict) -> dict:
        """
        Processes a single incoming raw transaction payload, calculates stateful 
        behavioral velocity features, and updates internal lookback structures.
        
        Expected payload fields: 'Timestamp', 'Sender_account', 'Receiver_account', 'Amount'
        """
        ts = pd.to_datetime(transaction['Timestamp'])
        sender = str(transaction['Sender_account'])
        receiver = str(transaction['Receiver_account'])
        amount = float(transaction['Amount'])

        # Initialize state slots for new accounts discovered in the stream
        if sender not in self.state:
            self.state[sender] = deque()
        if receiver not in self.state:
            self.state[receiver] = deque()

        # 1. Prune historical state older than 24h relative to the current transaction time
        self._prune_expired_records(sender, ts)
        self._prune_expired_records(receiver, ts)

        # 2. Extract context window benchmarks for the Sender (Outflow Metrics)
        h1_cutoff = ts - timedelta(hours=1)
        
        tx_count_1h = 0
        tx_amount_sum_1h = 0.0
        tx_count_24h = 0
        unique_beneficiaries_24h = set()

        # Scan sender's historical queue to calculate outflow velocity metrics
        for history_ts, history_amt, role, counterparty in self.state[sender]:
            if role == 'sender':
                tx_count_24h += 1
                unique_beneficiaries_24h.add(counterparty)
                if history_ts >= h1_cutoff:
                    tx_count_1h += 1
                    tx_amount_sum_1h += history_amt

        # Include the current pending transaction in the active count metrics
        tx_count_1h += 1
        tx_amount_sum_1h += amount
        tx_count_24h += 1
        unique_beneficiaries_24h.add(receiver)

        # 3. Calculate Advanced Relational Metrics
        # Beneficiary Diversity: Unique receivers over total transactions in 24 hours
        beneficiary_diversity_24h = len(unique_beneficiaries_24h) / tx_count_24h

        # Pass-Through Velocity: Calculate inflow vs outflow balance changes over the last hour
        inflow_sum_1h = 0.0
        outflow_sum_1h = tx_amount_sum_1h # Already includes current transaction

        for history_ts, history_amt, role, _ in self.state[sender]:
            if role == 'receiver' and history_ts >= h1_cutoff:
                inflow_sum_1h += history_amt

        # Compute symmetrical pass-through ratio
        pass_through_ratio_1h = (2.0 * min(inflow_sum_1h, outflow_sum_1h)) / (inflow_sum_1h + outflow_sum_1h + 1e-5)

        # 4. Commit current transaction states to memory for future ticks
        self.state[sender].append((ts, amount, 'sender', receiver))
        self.state[receiver].append((ts, amount, 'receiver', sender))

        # 5. Build and return the enriched transactional feature vector
        enriched_payload = transaction.copy()
        enriched_payload.update({
            'tx_count_1h': tx_count_1h,
            'tx_amount_sum_1h': tx_amount_sum_1h,
            'tx_count_24h': tx_count_24h,
            'beneficiary_diversity_24h': beneficiary_diversity_24h,
            'pass_through_ratio_1h': pass_through_ratio_1h
        })

        return enriched_payload

if __name__ == '__main__':
    print("Executing quick diagnostic test for StreamingStateManager feature engine...")
    manager = StreamingStateManager()
    
    # Simulate a classic layered pass-through money laundering trail
    mock_stream = [
        {'Timestamp': '2026-07-12 10:00:00', 'Sender_account': 'ACC_A', 'Receiver_account': 'MULE_1', 'Amount': 5000.0},
        {'Timestamp': '2026-07-12 10:15:00', 'Sender_account': 'MULE_1', 'Receiver_account': 'ACC_B', 'Amount': 4950.0},
        {'Timestamp': '2026-07-12 10:20:00', 'Sender_account': 'MULE_1', 'Receiver_account': 'ACC_C', 'Amount': 4950.0},
    ]

    for tx in mock_stream:
        res = manager.update_and_enrich(tx)
        print(f"\nTime: {res['Timestamp']} | Account: {res['Sender_account']} -> {res['Receiver_account']}")
        print(f" -> Pass-Through Ratio (1h): {res['pass_through_ratio_1h']:.4f}")
        print(f" -> Beneficiary Diversity (24h): {res['beneficiary_diversity_24h']:.4f}")
