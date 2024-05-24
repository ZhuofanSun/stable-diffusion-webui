import json
import os
import time

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

    def submit_post(self, url, data):
        """向url发送post请求"""
        try:
            if isinstance(data, dict):
                self.response = requests.post(url, data=json.dumps(data))
                print("切换模型...")
                time.sleep(3)
            else:
                raise TypeError("data must be dict")
        except requests.exceptions.RequestException as e:
            # 连接导致的
            print(e)
            print("*" * 40, "  ConnectionRefusedError  ", "*" * 40)
            print(f"向{url}发送post请求失败")
            print("*" * 40, "  ConnectionRefusedError  ", "*" * 40)
            self.utils.kill_script()  # 结束脚本
            self.utils.mem_collect()

    def generate(self, url, image_data_list, images_name):
        """
        生成image_data_list内所有数据的图片，并保存。生成过程中会显示简易进度条
        :param url: 用sd的功能url  eg. r'http://127.0.0.1:7860/sdapi/v1/txt2img'
        :param image_data_list: 这个功能的参数，json格式，'http://127.0.0.1:7860/docs'查看
        :param images_name: 生成的图片名字：eg. miku -> outputs/2024-05-18/miku3.png
        :return: 所有生成的文件地址列表
        """
        generated_images_path = []  # 用于存储生成的图片
        today = self.utils.get_today_date()  # yyyy-mm-dd

        # check api:
        if not self.utils.check_check_url(url):
            raise ConnectionError(f"url: {url} 连接失败")

        curr_batch = 0
        total = len(image_data_list)

        # 对于每个image_data，生成图片
        for image_data in image_data_list:
            # 图片生成放入新线程，主线程打印progress信息
            # target：在新线程中运行的函数
            # args：元组，包含了传递给target函数的参数
            thread = threading.Thread(target=self.submit_post, args=(url, image_data))
            thread.start()

            # 打印进度信息
            print("Generating " + images_name + " images...")
            ProgressBar(thread, self.root_url, image_data).show_progress(batch=curr_batch, total=total)
            print("Done!\n")

            if images_name == 'leak':  # 内存测试不保存图片
                return

            # 输出部分

            # 在当前批次下获取可用的index开头
            index_available = self.utils.get_available_index(images_name, today)

            print(f"\n当前保存批次：{curr_batch + 1}/{total}")

            try:
                for i in range(image_data['batch_size'] * image_data['n_iter']):  # No. 图片 = batch_size * n_iter
                    # 生成当前图片的 文件名 和 文件路径
                    save_image_name = images_name + str(index_available + i) + '.png'  # 名
                    save_image_path = os.path.join('outputs', today, save_image_name)  # 路径

                    # 编码并保存图片，存储文件路径：./outputs/yyyy-mm-dd/xxxx.png
                    self.utils.save_encoded_image(self.response.json()['images'][i], save_image_path)
                    generated_images_path.append(save_image_path)  # 保存文件路径
            # controlnet-depth 图片生成
            except KeyError:
                for i in range(len(image_data['controlnet_input_images'])):
                    # 如果有错误，打印错误信息
                    if "error" in self.response.json().keys():
                        print("*" * 40, f"Response ERROR: {self.response.json()['error']}", "*" * 40)
                        print("*" * 40, f"DETAILS: {self.response.json()['detail']}", "*" * 40)
                        print("*" * 40, f"第{i + 1}张图片有问题", "*" * 40)
                        continue  # 跳过当前图片

                    # 生成当前图片的 文件名 和 文件路径
                    save_image_name = images_name + str(index_available + i) + '.png'  # 文件名
                    save_image_path = os.path.join('outputs', today, save_image_name)  # 文件路径

                    # 编码并保存图片，存储文件路径：./outputs/yyyy-mm-dd/xxxx.png
                    self.utils.save_encoded_image(self.response.json()['images'][i],
                                                  save_image_path)  # 编码并保存图片

                    generated_images_path.append(save_image_path)  # 虽然可能不会需要，但还是返回吧
                    # 原图保存一份 可选
                    self.utils.save_encoded_image(image_data['controlnet_input_images'][i],
                                                  save_image_path.replace('.png', '_origin.png'))
            print("保存完成！\n")
            curr_batch += 1  # 下一批

        return generated_images_path
