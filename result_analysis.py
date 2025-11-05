import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

def read_and_preprocess_qaocs_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        df['TIME'] /= 1000  # 將毫秒轉換為秒
        return df
    except Exception as e:
        logging.error(f"Error reading QAOCS file {file_path}: {e}")
        return None

def read_and_preprocess_other_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        df['DURATION'] *= 1000  # 將秒轉換為毫秒
        df['BUFFER_STATE'] *= 1000  # 將秒轉換為毫秒
        df['REBUF'] *= 1000  # 將秒轉換為毫秒
        # VMAF is normalize by Duration/4 we need to convert it to the original scale
        df['VMAF'] = df['VMAF']*4 / (df['DURATION']/1000)
        return df
    except Exception as e:
        logging.error(f"Error reading other method file {file_path}: {e}")
        return None

def plot_metric(data_dict, metric, save_path):
    plt.figure(figsize=(12, 6))
    for method, df in data_dict.items():
        plt.plot(df['TIME'], df[metric], label=method)
    plt.xlabel('Time (seconds)')
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison')
    plt.legend()
    plt.savefig(f'{save_path}{metric.lower()}_comparison.png')
    plt.close()

def calculate_average_metrics(df, method):
    Switch_count = (df["QUALITY_INDEX"]-df["QUALITY_INDEX"].shift(1).fillna(0)!=0).sum()
    avg_duration = df['DURATION'].mean()
    avg_size = df['BYTES'].mean()
    avg_buffer_state = df['BUFFER_STATE'].mean()
    avg_vmaf = (df['VMAF'].mean()*df['DURATION']/1000).sum()/ (df['DURATION']/1000).sum()
    avg_bitrate = df['BITRATE'].mean()
    #stall_ratio = df["REBUF"].sum()/df["DURATION"].sum()*100
    # count the number of rebuffering events / total segment number
    stall_ratio = (df["REBUF"]>0).sum()/len(df)*100
    stall_time = df["REBUF"].sum()
        # stall event counting logic
    if "N-policy" in method or "D-policy" in method:
        rebuf_flags = (df["REBUF"] > 0).astype(int).values
        stall_events = np.sum((rebuf_flags[1:] - rebuf_flags[:-1]) == 1)
        if rebuf_flags[0] == 1:
            stall_events += 1
        stall_ratio = stall_events / len(df) * 100
        stall_event_count = stall_events
    else:
        stall_ratio = (df["REBUF"] > 0).sum() / len(df) * 100
        stall_event_count = (df["REBUF"] > 0).sum()
    #stall_time = df["REBUF"].mean()
    switch_ratio = Switch_count/len(df)
    total_size = df['BYTES'].sum()
    #vmaf smoothness = average of difference between consecutive vmaf scores
    avg_vmaf_smoothness = (df['VMAF'].diff().abs().sum()/(len(df)-1))
    # print(f'Average Duration: {avg_duration/1000:.2f}s')
    # print(f'Average Size: {avg_size/1e6:.2f}MB')
    # print(f'Average Buffer State: {avg_buffer_state/1000:.2f}s')
    # print(f'Average VMAF Score:" {avg_vmaf}')
    # print(f'Average VMAF Smoothness: {avg_vmaf_smoothness:.2f}')
    # print(f'Average Bitrate: {avg_bitrate}bps')
    # print(f'Stall Ratio: {stall_ratio:.2f}%')
    # print(f'Stall Time: {stall_time/1000:.2f}s')
    # print(f'Switch Ratio: {switch_ratio:.2f}%')
    # print(f'Total Size: {total_size/1e6:.2f}MB')

    return {
    'Average Duration(s)': f'{avg_duration/1000:.2f}',
    'Average Size(MB)': f'{avg_size/1e6:.2f}',
    'Average Buffer State(s)': f'{avg_buffer_state/1000:.2f}',
    'Average VMAF Score': avg_vmaf,
    'Average VMAF Smoothness': f'{avg_vmaf_smoothness:.2f}',
    'Average Bitrate(bps)': f'{avg_bitrate}',
    'Stall Ratio(%)': f'{stall_ratio:.2f}',
    'Stall Time(s)': f'{stall_time/1000:.2f}',
    'Switch Ratio(%)': f'{switch_ratio:.2f}',
    'Total Size(MB)': f'{total_size/1e6:.2f}'
        }

def process_video(video_name, methods):
    results = []
    data_dict = {}
    
    for method, path in methods.items():
        for file in os.listdir(path):
            if file.endswith(".csv"):
                file_path = os.path.join(path, file)
                # if method == 'QAOCS':
                #     df = read_and_preprocess_qaocs_csv(file_path)
                # qaocs_methods = ['QAOCS-BBA', 'QAOCS-RBA', 'QAOCS-BOLA', 'QAOCS-RMPC', 'QAOCS-PENSIEVE', 
                #                 'RBA', 'MPC','BOLA','PENSIEVE','BBA', 
                #                 'stallfrequency', 'stalltime', 'binaryindicator', '5', '10', '20','30', '40', '50' ]
                # if method in qaocs_methods:
                #     df = read_and_preprocess_qaocs_csv(file_path)
                # else:
                #     df = read_and_preprocess_other_csv(file_path)
                df = read_and_preprocess_qaocs_csv(file_path)
                if df is not None:
                    data_dict[method] = df
                    metrics = calculate_average_metrics(df, method)
                    results.append({'File': file, 'Method': method, **metrics})
    
    save_path = f'./Figures/{video_name}/'
    os.makedirs(save_path, exist_ok=True)
    
    plot_metric(data_dict, 'VMAF', save_path)
    plot_metric(data_dict, 'BITRATE', save_path)
    plot_metric(data_dict, 'BUFFER_STATE', save_path)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'./pq_test/{video_name}_performance_comparison.csv', index=False)
    #results_df.to_csv(f'./Training_output_abr_without_qaocs/{video_name}_performance_comparison.csv', index=False)
    logging.info(f"{video_name} Performance comparison results saved.")

if __name__ == '__main__':
    #video_list = ['BBB_360p24', 'LOL_3D', 'sport_highlight', 'sport_long_take', 'TOS_360p24', 'video_game', 'underwater']
    #video_list = ['LOL_3D', 'sport_highlight', 'video_game']
    video_list = ['LOL_3D']
    for video_name in video_list:
        methods = {
            #'QAOCS': f'./Training_output/QAOCS/{video_name}/',
            #'Segue': f'./Training_output/segue_wide_eye/{video_name}/',
            #'Segue': f'./Training_output/segue_GOP-4/{video_name}/',
            # 'GOP-5': f'./Training_output/segue_GOP-5/{video_name}/',
            #'GOP-4': f'./Training_output/segue_GOP-4/{video_name}/',
            #'GOP-4': f'./Training_output/segue_constant_4/{video_name}/',
            # 'GOP-3': f'./Training_output/segue_GOP-3/{video_name}/',
            # 'Constant-5': f'./Training_output/segue_constant_5/{video_name}/',
            #'Constant-4': f'./Training_output/segue_constant_4/{video_name}/',
            #'Constant-4': f'./Training_output/segue_wide_eye/{video_name}/',
            # 'Constant-3': f'./Training_output/segue_constant_3/{video_name}/'
        # for abr test
            # 'QAOCS-RBA': f'./Training_output_abr/rba/{video_name}/',
            # 'QAOCS-RMPC': f'./Training_output_abr_rmpc/mpc/{video_name}/',
            # 'QAOCS-BOLA': f'./Training_output_abr/bola/{video_name}/',
            # 'QAOCS-BBA': f'./Training_output_abr/bba/{video_name}/',
            # 'QAOCS-PENSIEVE': f'./Training_output_abr/pensieve/{video_name}/',
            # 'Segue': f'./Training_output_abr/segue_GOP-4/{video_name}/',
            # 'Constant-4': f'./Training_output_abr/segue_wide_eye/{video_name}/',
            # 'GOP-4': f'./Training_output_abr/segue_constant_4/{video_name}/'
        # for abr test without qaocs
            # 'QAOCS-RBA': f'./Training_output_abr_without_qaocs/rba/{video_name}/',
            # 'QAOCS-MPC': f'./Training_output_abr_without_qaocs_rmpc/mpc/{video_name}/',
            # 'QAOCS-BOLA': f'./Training_output_abr_without_qaocs/bola/{video_name}/',
            # 'QAOCS-BBA': f'./Training_output_abr_without_qaocs/bba/{video_name}/',
            # 'QAOCS-PENSIEVE': f'./Training_output_abr_without_qaocs/pensieve/{video_name}/',

            # 'RBA': f'./Training_output_abr_without_qaocs/rba_nonqaocs/{video_name}/',
            # 'MPC': f'./Training_output_abr_without_qaocs/mpc_nonqaocs/{video_name}/',
            # 'BOLA': f'./Training_output_abr_without_qaocs/bola_nonqaocs/{video_name}/',
            # 'BBA': f'./Training_output_abr_without_qaocs/bba_nonqaocs/{video_name}/',
            # 'PENSIEVE': f'./Training_output_abr_without_qaocs/pensieve_nonqaocs/{video_name}/',

            # 'SEGUE-RBA': f'./Training_output_abr_without_qaocs/segue_rba/{video_name}/',
            # 'SEGUE-MPC': f'./Training_output_abr_without_qaocs/segue_mpc/{video_name}/',
            # 'SEGUE-BOLA': f'./Training_output_abr_without_qaocs/segue_bola/{video_name}/',
            # 'SEGUE-BBA': f'./Training_output_abr_without_qaocs/segue_bba/{video_name}/',
            # 'SEGUE-PENSIEVE': f'./Training_output_abr_without_qaocs/segue_pensieve/{video_name}/',
        #for reward compare
            # 'stallfrequency': f'./reward_compare/stallfrequency/{video_name}/',
            # 'stalltime': f'./reward_compare/stalltime/{video_name}/',
            # 'binaryindicator': f'./reward_compare/binaryindicator/{video_name}/',
        #for pq_test
            #'pq-policy-5':f'./pq_test/pq-policy/5/{video_name}/',
            #'pq(10, 20)':f'./pq_test/pq-policy/10/{video_name}/',
            'pq(20, 30)':f'./pq_test/pq-policy/20/{video_name}/',
            'pq(30, 40)':f'./pq_test/pq-policy/30/{video_name}/',
            'pq(40, 50)':f'./pq_test/pq-policy/40/{video_name}/',
            #'pq-policy-50':f'./pq_test/pq-policy/50/{video_name}/',
        # for N-policy
             'N-policy-3':f'./pq_test/N-policy/3/{video_name}/',
             'N-policy-5':f'./pq_test/N-policy/5/{video_name}/',
             'N-policy-7':f'./pq_test/N-policy/7/{video_name}/',
        # for D-policy
             'D-policy-5':f'./pq_test/D-policy/5/{video_name}/',
             'D-policy-10':f'./pq_test/D-policy/10/{video_name}/',
             'D-policy-15':f'./pq_test/D-policy/15/{video_name}/',
        # for single threshold
            #'Single(30)':f'./pq_test/single/30/{video_name}/',
             #'Single(40)':f'./pq_test/single/40/{video_name}/',
             #'Single(50)':f'./pq_test/single/50/{video_name}/',
        }
        
        process_video(video_name, methods)