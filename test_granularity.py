import pandas as pd
import numpy as np
from test_agent import analyze_data
import json

# Create sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31')
data = pd.DataFrame({
    'date': dates,
    'product': np.random.choice(['A', 'B', 'C'], size=len(dates)),
    'sales': np.random.normal(100, 20, size=len(dates))
})

# Save to CSV
data.to_csv('test_data.csv', index=False)

# Test query with ambiguous granularity
result = analyze_data(
    query="What are the sales trends for product A? Please consider different time granularities in your analysis.",
    file_path='test_data.csv'
)

print("\nAnalysis Result:")
print("================")
print(f"Status: {result['status']}")
if result['status'] == 'success':
    print("\nResponse:")
    print(json.dumps(result['response'], indent=2))
else:
    print(f"Error: {result['error']}")

print("\nDebug Output:")
print("=============")
print(result.get('debug_output', 'No debug output available')) 