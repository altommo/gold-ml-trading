import re
import pandas as pd

# Parse trades.txt to extract Gold spot trades
with open('../../../trades.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Split into trade blocks (each starts with an ID number)
lines = content.split('\n')

trades = []
i = 0
while i < len(lines):
    line = lines[i].strip()

    # Check if this is a trade ID (just a number)
    if line.isdigit():
        trade_id = line

        # Next line has instrument and buy/sell
        if i + 1 < len(lines):
            instrument_line = lines[i + 1].strip()

            # Only Gold spot trades (not options)
            if instrument_line.startswith('Gold') and 'Call' not in instrument_line and 'Put' not in instrument_line:
                parts = instrument_line.split('\t')
                if len(parts) >= 2:
                    instrument = parts[0]
                    direction = parts[1]

                    # Amount line
                    if i + 2 < len(lines):
                        amount_line = lines[i + 2].strip()
                        amount_parts = amount_line.split('\t')
                        if len(amount_parts) >= 2:
                            amount_str = amount_parts[0]
                            amount = float(amount_str.replace(' Ounces', ''))
                            open_time = amount_parts[1]

                            # Close time and P/L line
                            if i + 3 < len(lines):
                                close_line = lines[i + 3].strip()
                                close_parts = close_line.split('\t')
                                if len(close_parts) >= 3:
                                    close_time = close_parts[0]
                                    gross_pnl = close_parts[1].replace('£', '').replace('−', '-').replace(',', '').strip()
                                    open_rate = float(close_parts[2])

                                    # Next line has close rate
                                    if i + 4 < len(lines):
                                        rate_line = lines[i + 4].strip()
                                        rate_parts = rate_line.split('\t')
                                        if len(rate_parts) >= 1:
                                            close_rate = float(rate_parts[0])

                                            trades.append({
                                                'trade_id': trade_id,
                                                'instrument': instrument,
                                                'direction': direction,
                                                'amount': amount,
                                                'open_time': open_time,
                                                'close_time': close_time,
                                                'gross_pnl': float(gross_pnl),
                                                'open_rate': open_rate,
                                                'close_rate': close_rate
                                            })
        i += 10  # Skip to next trade block
    else:
        i += 1

df = pd.DataFrame(trades)
print(f"Found {len(df)} Gold spot trades")
print(df.head(20))

# Save
df.to_csv('data/gold_spot_trades.csv', index=False)
print("\nSaved to data/gold_spot_trades.csv")
