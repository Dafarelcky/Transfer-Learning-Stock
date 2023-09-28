import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data into a DataFrame
df = pd.read_csv('D:\\RSBP\\Transfer_Learning_Stock\\usdeur_fix.csv')

# Create a line plot
plt.plot(df['timestamp'], df['value'])
plt.xlabel('timestamp')
plt.ylabel('value')
plt.title('USD - EUR')
plt.show()
