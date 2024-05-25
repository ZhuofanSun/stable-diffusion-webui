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
    def __init__(self, thread, root_url: str, image_data, utils=None):
        self.thread = thread
        if utils is None:
            self.utils = Utils(root_url)
        else:
            self.utils = utils
        self.image_data = image_data

    def show_progress(self, batch, total_batch):
        if not isinstance(self.image_data, dict):
            print("*" * 40, " Image_data类型错误 ", "*" * 40)
            raise ValueError("image_data必须是字典/json")

        try:
            print("等待任务开始...")
            start = time.time()
            steps, job_count = 0, 0
            print()
            while job_count is None or steps is None or steps * job_count <= 0:
                if time.time() - start > 5:
                    break
                progress = self.utils.get_progress()  # dict
                steps = progress["state"]['sampling_steps']
                job_count = progress["state"]["job_count"]
                time.sleep(0.8)

            total = steps * job_count
            print("任务开始。")
            print(f"\n当前处理批次： {batch + 1}/{total_batch} ")
            # 进度条的总值
            with tqdm(total=100) as pbar:
                while self.thread.is_alive():
                    progress = self.utils.get_progress()  # dict
                    progress_percentage = progress["progress"]
                    eta = progress["eta_relative"]
                    job_no = progress["state"]["job_no"]

                    # 更新进度条
                    pbar.n = int(progress_percentage * 100)
                    pbar.set_description(f"预估剩余时间: {second2time(eta)} | {job_no}/{job_count}")
                    pbar.refresh()

                    time.sleep(1)  # 休眠1秒
        except KeyError:
            # controlnet 没有进程api
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
