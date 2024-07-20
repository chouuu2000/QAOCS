import pandas as pd

# Load the Lumos5G trace dataset
lumos5g_df = pd.read_csv('Lumos5G-v1.0.csv')

run_num = 1
lumos5g_transformed = pd.DataFrame(columns=['Timestamp [s]', 'Bandwidth [MB]'])

for index, row in lumos5g_df.iterrows():
    if row['run_num'] > run_num:
        # Use float_format to remove extra space
        lumos5g_transformed.to_csv(f'lumos5g_trace_{run_num}.txt', sep=' ', index=False, header=False, float_format='%.1f')
        lumos5g_transformed = pd.DataFrame(columns=['Timestamp [s]', 'Bandwidth [MB]'])
        print(f"Dumped lumos5g_trace_{run_num}.txt")
        run_num += 1
    new_row = pd.DataFrame([[row['seq_num'], row['Throughput']]], columns=['Timestamp [s]', 'Bandwidth [MB]'])
    lumos5g_transformed = pd.concat([lumos5g_transformed, new_row], ignore_index=True)
    


# Write the last run
if not lumos5g_transformed.empty:
    lumos5g_transformed.to_csv(f'lumos5g_trace_{run_num}.txt', sep=' ', index=False, header=False, float_format='%.1f')
    print(f"Dumped lumos5g_trace_{run_num}.txt")