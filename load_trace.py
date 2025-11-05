import os
import json

COOKED_TRACE_FOLDER = './train/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER, order_json_path="./trace_order.json"):
    # with open(order_json_path, 'r') as f:
    #     cooked_files = json.load(f)
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names

def load_trace_evaluate(trace_filename):
    cooked_bw = []
    cooked_time = []
    assert os.path.exists(trace_filename)
    with open(trace_filename, 'rb') as f:
        for line in f:
            parse = line.split()
            cooked_time.append(float(parse[0]))
            cooked_bw.append(float(parse[1]))
    return cooked_time, cooked_bw