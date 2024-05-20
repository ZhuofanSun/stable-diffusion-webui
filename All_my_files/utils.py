import base64
import os
from datetime import date
import PIL.Image as Image
import io
import requests


class Utils:
    def __init__(self, url_root=r'http://127.0.0.1:7860'):
        self.root = url_root
        self.txt2img_url = url_root + '/sdapi/v1/txt2img'
        self.img2img_rul = url_root + '/sdapi/v1/img2img'
        self.options_url = url_root + '/sdapi/v1/options'
        self.progress_url = url_root + '/sdapi/v1/progress'
        self.controlnet_modelList_url = url_root + '/controlnet/model_list'
        self.controlnet_moduleList_url = url_root + '/controlnet/module_list'
        self.controlnet_controlSettings_url = url_root + '/controlnet/settings'
        self.controlnet_detect_url = url_root + '/controlnet/detect'

    def to_filename_depth(self, file_path):
        filename = os.path.basename(file_path)
        name = os.path.splitext(filename)[0]
        return name + "_depth.png"

    def read_image_to_base64(self, image_path):
        """
        读取图片并转换成base64编码的字符串
        :param image_path: 图片路径
        :return: base64_string
        """
        try:
            with open(image_path, 'rb') as image_file:
                image = image_file.read()
            return base64.b64encode(image).decode('utf-8')
        except Exception as e:
            print(e)
            return None

    def save_base64_to_image(self, base64_string, image_path):
        """
        将base64编码的图片保存到指定路径
        :param base64_string: base64编码的图片
        :param image_path: 保存路径
        :return: None
        """
        try:
            with open(image_path, 'wb') as image_file:
                image_file.write(base64.b64decode(base64_string))
                image_file.flush()
        except Exception as e:
            print(e)
            pass

    def decode_base64_to_image(self, base64_string):
        """
        将base64编码的图片解码，并转换成PIL.Image对象
        :param base64_string: base64编码的图片
        :return: image
        """
        image = Image.open(io.BytesIO(base64.b64decode(base64_string)))
        return image

    def encode_image_to_base64(self, image):
        """
        将PIL.Image对象转换成base64编码的字符串
        :param image: PIL.Image对象
        :return: base64_string
        """
        with io.BytesIO() as output_bytes:
            image.save(output_bytes, format="PNG")
        bytes_data = output_bytes.getvalue()
        return base64.b64encode(bytes_data).decode('utf-8')

    def save_encoded_image(self, b64_image: str, output_path: str):
        """
        将base64编码的图片解码并保存到指定路径
        :param b64_image: base64编码的图片，response.json()['images'][i]
        :param output_path: 保存路径
        :return: None
        """
        try:
            img = self.decode_base64_to_image(b64_image)
            img.save(output_path)
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

    def get_today_date(self):
        """获取今天的日期格式yyyy-mm-dd"""
        current_date = date.today()
        # 格式化 yyyy-mm-dd
        return current_date.strftime("%Y-%m-%d")

    def get_txt2img_url(self):
        return self.txt2img_url

    def get_img2img_url(self):
        return self.img2img_rul

    def get_options_url(self):
        return self.options_url

    def get_controlnet_modelList_url(self):
        return self.controlnet_modelList_url

    def get_controlnet_moduleList_url(self):
        return self.controlnet_moduleList_url

    def get_controlnet_controlSettings_url(self):
        return self.controlnet_controlSettings_url

    def get_controlnet_detect_url(self):
        return self.controlnet_detect_url

    def get_controlnet_modelList(self):
        response = requests.get(self.controlnet_modelList_url)
        return response.json()  # 返回json的数据（字典）

    def get_controlnet_moduleList(self):
        response = requests.get(self.controlnet_moduleList_url)
        return response.json()  # 返回json的数据（字典）

    def get_controlnet_controlSettings(self):
        response = requests.get(self.controlnet_controlSettings_url)
        return response.json()

    def get_txt2img_options(self):
        response = requests.get(self.options_url)
        return response.json()

    def get_txt2img(self):
        response = requests.get(self.txt2img_url)
        return response.json()

    def get_img2img(self):
        response = requests.get(self.img2img_rul)
        return response.json()

    def get_progress_url(self):
        return self.progress_url

    def get_progress(self):
        response = requests.get(self.progress_url)
        return response.json()
