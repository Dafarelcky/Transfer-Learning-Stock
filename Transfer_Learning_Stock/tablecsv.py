import pandas as pd

# Replace this with the path to your text file
text_file_path = 'D:\\RSBP\\Transfer_Learning_Stock\\usdeur_datasets.txt'

# Read the text data and split it into rows and columns
with open(text_file_path, 'r') as text_file:
    lines = text_file.read().split('\n')
    data = [line.split(',') for line in lines if line]

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Replace this with the desired Excel file path
csv_file_path = 'D:\\RSBP\\Transfer_Learning_Stock\\usdeur_fix.csv'

# Write the DataFrame to an Excel file
df.to_csv(csv_file_path, index=False, header=False)

print(f"Data has been saved to {csv_file_path}")
