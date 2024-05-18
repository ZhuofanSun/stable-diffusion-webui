"""
测试，根据sd-webui的api来显示进度条
工具文件
"""
import requests
import time


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
    def __init__(self, port=7860):
        """

        :param port:sd默认端口7860，加--nowebui 端口是7861
        """
        self.port = port
        self.url = r"http://127.0.0.1:" + str(port)
        self.progress_url = self.url + r"/sdapi/v1/progress"
        self.response = None

    def show_progress(self):
        progress = requests.get(self.progress_url).json()
        progress_percentage = progress["progress"]
        eta = progress["eta_relative"]
        print("Progress: ", round(progress_percentage * 100, 1), "%", "\tETA: ", second2time(eta))
        time.sleep(1)  # 休眠一秒
