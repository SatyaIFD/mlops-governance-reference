import csv
import time
from pathlib import Path

def transaction_stream_generator(csv_path: Path, delay_seconds: float = 0.01):
    """
    Yields individual transaction payloads one by one from the data source,
    simulating a live streaming infrastructure event loop.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing base transaction data ledger at: {csv_path}")
        
    print(f"Opening data stream channel from: {csv_path.name}")
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Enforce micro-delay to simulate a living network wire tick
            if delay_seconds > 0:
                time.sleep(delay_seconds)
                
            # Yield standard transaction payload format
            yield {
                'Timestamp': f"{row['Date']} {row['Time']}",
                'Sender_account': row['Sender_account'],
                'Receiver_account': row['Receiver_account'],
                'Amount': float(row['Amount']),
                'Laundering_type': row.get('Laundering_type', 'NONE'),
                'Is_laundering': int(row['Is_laundering'])
            }

if __name__ == '__main__':
    print("Testing Stream Generator output matrix (fetching first 5 live ticks)...")
    data_dir = Path("../data").resolve()
    target_csv = data_dir / "SAML-D.csv"
    
    # Instantiate the iterable generator with zero delay for quick debugging
    stream = transaction_stream_generator(target_csv, delay_seconds=0.0)
    
    for i, tx_event in enumerate(stream):
        if i >= 5:
            break
        print(f"Tick {i+1} ──> {tx_event['Timestamp']} | Sender: {tx_event['Sender_account']} | Amt: ${tx_event['Amount']}")