import pandas as pd 
import numpy as np
data = {
    'transaction_amount': np.random.randint(100, 10000, 1000),
    'transaction_time': np.random.choice(['12:00', '15:30', '03:00', '22:45'], 1000),
    'location': np.random.choice(['ATM_001', 'ATM_002', 'ATM_003'], 1000),
    'device_id': np.random.choice(['DEV_001', 'DEV_002', 'DEV_003'], 1000),
    'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
}
df = pd.DataFrame(data)
df.to_csv('transactions.csv', index=False)
