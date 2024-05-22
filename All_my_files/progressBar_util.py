"""
测试，根据sd-webui的api来显示进度条
工具文件
"""
import time
import traceback

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
        total = len(self.image_data)
        curr = 0
        for image_data in self.image_data:
            curr += 1
            print(f"当前处理批次： {curr}/{total} ")
            try:
                total = image_data['steps'] * image_data['n_iter'] * image_data['batch_size']
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
                if self.utils.get_process() is not None:
                    # 实时读取输出
                    while self.thread.is_alive():
                        while True:
                            output = self.utils.get_process().stdout.readline()
                            if output:
                                print(output.strip())
                else:
                    print("No progress data")
                    clock = 0
                    while self.thread.is_alive():
                        try:
                            print(f"Time elapsed: {second2time(clock)}")
                            time.sleep(1)
                            clock += 1
                        except Exception:
                            traceback.print_exc()
                            break
