import pandas as pd

file_path = r'D:\ids\datasets\ftp_ssh.csv'
chunk_size = 70000

# Initialize an empty Series to store aggregated counts
total_label_counts = pd.Series(dtype='int64')

# Process the file in chunks
chunks = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip', chunksize=chunk_size)

print(f"Processing {file_path} in chunks of {chunk_size}...")

for i, chunk in enumerate(chunks):
    # Get counts for the current chunk (including NaNs)
    chunk_counts = chunk['Label'].value_counts(dropna=False)

    # Add the current chunk's counts to the running total
    total_label_counts = total_label_counts.add(chunk_counts, fill_value=0)

    print(f"Processed chunk {i+1} ({ (i+1) * chunk_size } rows so far...)")

print("\n--- Final Aggregated Label Count Breakdown ---")
print(total_label_counts.sort_values(ascending=False).astype(int))