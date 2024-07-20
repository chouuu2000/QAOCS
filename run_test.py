import os
import subprocess
import shutil

# 視頻列表
video_list = [
    "bigbuckbunny360p24",
    "Tears_of_Steel_360p24",
    "LOL_3D",
    "sport_highlight",
    "underwater",
    "sport_long_take",
    "video_game"
]

# 主程序
def main():
    for video in video_list:
        # 執行Python腳本
        video_path = f"./Video_source/{video}.mp4"
        command = f"python3 test.py --video_path {video_path}"
        subprocess.run(command, shell=True)
        
        # 移動trace文件
        move_trace_files()

def move_trace_files():
    done_folder = "./test/done"
    test_folder = "./test"
    
    # 確保目錄存在
    if not os.path.exists(done_folder):
        print(f"錯誤: {done_folder} 不存在")
        return
    
    # 獲取done文件夾中的所有文件
    files = os.listdir(done_folder)
    
    # 移動所有trace文件
    for file in files:
        if file.endswith(".txt"):
            src = os.path.join(done_folder, file)
            dst = os.path.join(test_folder, file)
            shutil.move(src, dst)
            print(f"已移動: {file}")

if __name__ == "__main__":
    main()