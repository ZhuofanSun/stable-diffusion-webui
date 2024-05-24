import base64
import os
import signal
import time
from datetime import date
import PIL.Image as Image
import io
import requests
import subprocess
import gc
import socket
from urllib.parse import urlparse


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
        self.process = None

    def start_webui(self, script_name="webui.sh", args=["--api"]):
        """
        启动工作目录前一层文件夹里的webui.sh脚本
        :param script_name: 脚本名称
        :param args: 脚本参数
        :return: 进程对象
        """
        print('-' * 20, "正在执行 `../webui.sh --api` ...", '-' * 20)
        print()
        # 获取当前工作目录上一层的目录
        script_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        # 构建脚本路径
        script_path = os.path.join(script_dir, script_name)

        # 启动脚本，并重定向标准输出和标准错误输出
        self.process = subprocess.Popen(['bash', script_path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True,
                                        cwd=script_dir,
                                        preexec_fn=os.setsid  # 将子进程放入新的进程组，否则可能连着pycharm一起关闭
                                        )

        # 实时读取输出
        while True:
            output = self.process.stdout.readline()
            if output:
                print(output.strip())
                if "Running on local URL:" in output:
                    break

        print('-' * 20, "sd-webui api 已启动", '-' * 20)
        time.sleep(5)  # 等待5秒

        # 返回进程对象，以便以后结束脚本
        return self.process

    def kill_process(self, process):
        if process is not None:
            try:
                # 获取进程组 ID
                pgid = os.getpgid(process.pid)
                # 发送 SIGTERM 信号以尝试优雅终止进程
                # 检查进程组名称（可选，根据具体需求）
                # pgid_name = get_process_group_name(pgid)  # 自行实现获取进程组名称的方法
                os.killpg(pgid, signal.SIGTERM)
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # 如果进程没有在指定时间内终止，发送 SIGKILL 信号
                print("Process killed with SIGKILL.")
                os.killpg(pgid, signal.SIGKILL)
                process.wait()
            except Exception as e:
                print(f"Failed to kill process group: {e}")

            print("-" * 20, "Process terminated.", "-" * 20)
            process = None  # 清空 process 引用以释放资源
        else:
            print("-" * 20, "No process to kill.", "-" * 20)

    def kill_script(self, process=None):
        """
        结束进程
        :param process: 进程对象
        :return: None
        """
        print("-" * 40, "即将结束脚本", "-" * 40)

        self.kill_process(process)
        self.kill_process(self.process)

    def wait_3_secondes(self):
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("0")

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

    def mem_collect(self):
        print("\n内存回收：")
        # Print the number of objects known by the collector, before and after a collection
        print("Objects before collection: ", gc.get_count())
        gc.collect()
        print("Objects after collection: ", gc.get_count())

    def check_check_url(self, url=None):
        """
        查看root_url的端口连通性
        :return: True/False
        """
        if url is None:
            url = self.root

        try:
            print(f"检查连接{url}")
            # 解析 URL
            parsed_url = urlparse(url)
            host = parsed_url.hostname
            port = parsed_url.port

            # 使用 socket 检查端口是否开放
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)  # 设置超时时间为1秒
                s.connect((host, port))

            # 使用 requests 检查 URL 的响应
            response = requests.get(url)
            if response.json().get('detail') != "Not Found":
                print("连接成功")
                return True
            else:
                print("连接失败")
                return False

        except (socket.timeout, ConnectionRefusedError):
            print("连接失败")
            return False
        except requests.RequestException:
            return True
        except Exception as e:
            print(e)

    def get_image_size(self, image_path):
        """
        获取图片的尺寸
        :param image_path: 图片路径
        :return: 宽，高
        """
        if not (os.path.exists(image_path) or os.path.isfile(image_path)):
            raise FileNotFoundError(f"文件不存在：{image_path}")
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height

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

    def get_process(self):
        return self.process
