"""
测试，根据sd-webui的api来显示进度条
工具文件
"""
import time
from utils import Utils
from tqdm import tqdm


def second2time(second):
    """
    将秒数转换为min:sec格式
    :param second: 秒数
    :return: min:sec格式
    """
    minute = int(second // 60)
    second = int(second % 60)
    return "{:02}:{:02}".format(minute, second)  # 格式化输出不足两位的补0


class ProgressBar:
    def __init__(self, thread, root_url, image_data):
        self.thread = thread
        self.response = None
        self.utils = Utils(root_url)
        self.image_data = image_data

    def show_progress(self):
        try:
            total = self.image_data['steps'] * self.image_data['n_iter'] * self.image_data['batch_size']
            # 进度条的总值
            with tqdm(total=total) as pbar:
                while self.thread.is_alive():
                    progress = self.utils.get_progress()  # dict
                    progress_percentage = progress["progress"]
                    eta = progress["eta_relative"]
                    job_count = progress["state"]["job_count"]
                    job_no = progress["state"]["job_no"]

                    # 更新进度条
                    pbar.n = int(progress_percentage * total)
                    pbar.set_description(f"ETA: {second2time(eta)} | {job_no}/{job_count}")
                    pbar.refresh()

                    time.sleep(1)  # 休眠1秒
        except KeyError:
            print("No progress data")
            clock = 0
            while self.thread.is_alive():
                print(f"Time elapsed: {second2time(clock)}")
                time.sleep(1)
                clock += 1
            pass
