import json
import os
import requests
import threading
from progressBar_util import ProgressBar
from utils import Utils


class Generate:
    def __init__(self, root_url=r'http://127.0.0.1:7860'):
        self.response = None  # 储存向sd-webui发送请求后的响应
        self.option_data = {}  # 用于存储选项数据
        self.utils = Utils(root_url)
        self.root_url = root_url

    def set_clip(self, new_clip=None):

        if new_clip is None:
            pass
        else:
            self.option_data["CLIP_stop_at_last_layers"] = new_clip

    def set_model(self, new_model=None):

        if new_model is None:
            pass
        else:
            self.option_data["sd_model_checkpoint"] = new_model

    def set_vae(self, new_vae=None):

        if new_vae is None:
            pass
        else:
            self.option_data["sd_vae"] = new_vae

    def post_option(self):

        if 'sd_vae' not in self.option_data.keys():
            self.option_data['sd_vae'] = None

        requests.post(url=self.utils.get_options_url(), json=self.option_data)

    def submit_post(self, url: str, data: dict):
        """向url发送post请求"""
        try:
            self.response = requests.post(url, data=json.dumps(data))
        except requests.exceptions.RequestException as e:
            # 连接导致的
            print(e)
            print("*" * 40, "  ConnectionRefusedError  ", "*" * 40)
            print(f"向{url}发送post请求失败")
            print("*" * 40, "  ConnectionRefusedError  ", "*" * 40)
            self.utils.kill_script()  # 结束脚本
            self.utils.mem_collect()

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

        today = self.utils.get_today_date()  # yyyy-mm-dd

        # check api:
        if not self.utils.check_check_url(url):
            raise ConnectionError(f"url: {url} 连接失败")

        # 图片生成放入新线程，主线程打印progress信息
        # target：在新线程中运行的函数
        # args：元组，包含了传递给target函数的参数

        thread = threading.Thread(target=self.submit_post, args=(url, image_data))
        thread.start()

        index_available = self.utils.get_available_index(images_name, today)
        print("Generating " + images_name + " images...")
        ProgressBar(thread, self.root_url, image_data).show_progress()
        print("Done!\n")

        if images_name == 'leak':
            return

        print(self.response.json())

        # response = submit_post(txt2img_url, imp_data)  # 在这里生成图片
        try:
            for i in range(image_data['batch_size'] * image_data['n_iter']):  # No. 图片 = batch_size * n_iter
                save_image_name = images_name + str(index_available + i) + '.png'  # 文件名
                save_image_path = os.path.join('outputs', today, save_image_name)  # 文件路径

                # 存储文件路径：./outputs/yyyy-mm-dd/xxxx.png
                self.utils.save_encoded_image(self.response.json()['images'][i], save_image_path)  # 编码并保存图片
        except KeyError:
            for i in range(len(image_data['controlnet_input_images'])):
                save_image_name = images_name + str(index_available + i) + '.png'  # 文件名
                save_image_path = os.path.join('outputs', today, save_image_name)  # 文件路径

                # 存储文件路径：./outputs/yyyy-mm-dd/xxxx.png
                self.utils.save_encoded_image(self.response.json()['images'][i], save_image_path)  # 编码并保存图片
