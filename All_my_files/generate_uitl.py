import json
import base64
import requests
from datetime import date
import os
import threading
from progressBar_util import ProgressBar


class Generate:
    def __init__(self):
        self.response = None  # 储存向sd-webui发送请求后的响应

    def get_today_date(self):
        """获取今天的日期格式yyyy-mm-dd"""
        current_date = date.today()
        # 格式化 yyyy-mm-dd
        return current_date.strftime("%Y-%m-%d")

    def submit_post(self, url: str, data: dict):
        """向url发送post请求"""
        self.response = requests.post(url, data=json.dumps(data))

    def save_encoded_image(self, b64_image: str, output_path: str):
        """
        将base64编码的图片解码并保存到指定路径
        :param b64_image: base64编码的图片，response.json()['images'][i]
        :param output_path: 保存路径
        :return: None
        """
        try:
            with open(output_path, 'wb') as image_file:
                image_file.write(base64.b64decode(b64_image))
                image_file.flush()
                image_file.close()
        except Exception as e:
            print(e)
            pass

    def get_available_index(self, img_name, today):
        """
        根据outputs/yyyy-mm-dd/文件夹下（如果文件夹不存在则创建）的文件名称，返回一个可用的index
        :param img_name: 名称部分
        :return: 一个可用的index，作为图片名称的一部分，让图片名字不重复
        """
        save_image_path = os.path.join('outputs', today)
        os.makedirs(save_image_path, exist_ok=True)  # 创建文件夹
        index = 0  # 从0开始尝试可用的索引
        while os.path.exists(os.path.join(save_image_path, img_name + str(index) + '.png')):
            index += 1
        return index

    def generate(self, url, image_data, images_name):
        """
        ./webui.sh --api 以api模式运行sd
        会返回一个URL，这个URL是一个API接口，可以通过这个API接口来调用模型
        替换下面的txt2img_url为你的API接口，基本不用改。
        不加--nowebui 端口是7860，加了是7861，

        :param url: 用sd的功能url  eg. r'http://127.0.0.1:7860/sdapi/v1/txt2img'
        :param image_data: 这个功能的参数，json格式，'http://127.0.0.1:7860/docs'查看
        :param images_name: 生成的图片名字：eg. miku -> outputs/2024-05-18/miku3.png
        :return: None
        """

        today = self.get_today_date()  # yyyy-mm-dd

        # 图片生成放入新线程，主线程打印progress信息
        # target：在新线程中运行的函数
        # args：元组，包含了传递给target函数的参数
        thread = threading.Thread(target=self.submit_post, args=(url, image_data))
        thread.start()
        while thread.is_alive():
            ProgressBar().show_progress()

        # response = submit_post(txt2img_url, imp_data)  # 在这里生成图片

        print(self.response.json())
        index_available = self.get_available_index(images_name, today)
        for i in range(image_data['batch_size'] * image_data['n_iter']):  # No. 图片 = batch_size * n_iter
            save_image_name = images_name + str(index_available + i) + '.png'  # 文件名
            save_image_path = os.path.join('outputs', today, save_image_name)  # 文件路径

            # 存储文件路径：./outputs/yyyy-mm-dd/xxxx.png
            self.save_encoded_image(self.response.json()['images'][i], save_image_path)  # 编码并保存图片
