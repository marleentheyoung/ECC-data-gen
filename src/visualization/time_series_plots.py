import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/marleendejonge/Desktop/ECC-data-generation/outputs/paris_agreement_timeseries_combined_Q.csv')
df['date'] = pd.to_datetime(df['date'])

fig, ax = plt.subplots(figsize=(14, 8))

# Main time series
ax.plot(df['date'], df['count_sum'], marker='o', linewidth=2.5, markersize=4, color='#2E86AB')

# Add key events
events = {
    '2015-12-12': 'Paris Agreement\nAdoption',
    '2017-06-01': 'US Withdrawal\nAnnouncement',
    '2021-01-20': 'US Re-entry'
}

for date_str, label in events.items():
    event_date = pd.to_datetime(date_str)
    ax.axvline(event_date, color='red', linestyle='--', alpha=0.7)
    ax.text(event_date, ax.get_ylim()[1]*0.8, label, 
            rotation=90, ha='right', va='top', fontsize=10, color='red')

ax.set_title('Paris Agreement Attention in Earnings Calls', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Number of Relevant Climate Snippets', fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()