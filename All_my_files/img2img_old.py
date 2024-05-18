import json
import base64
import requests
from datetime import date
import os
import threading
from progressBar_util import ProgressBar

response = None


def get_today_date():
    current_date = date.today()
    # 格式化 yyyy-mm-dd
    return current_date.strftime("%Y-%m-%d")


def submit_post(url: str, data: dict):
    global response
    response = requests.post(url, data=json.dumps(data))


def save_encoded_image(b64_image: str, output_path: str):
    try:
        with open(output_path, 'wb') as image_file:
            image_file.write(base64.b64decode(b64_image))
    except Exception as e:
        print(e)
        pass


def get_available_index(img_name, today):
    """
    图片存储在outputs/yyyy-mm-dd/文件夹下，如果文件夹不存在则创建
    在这个文件夹下文件名是img_name+index.png, index 从0开始。
    函数返回一个可用的index
    :param img_name: 名称部分
    :return: 可用索引
    """
    save_image_path = os.path.join('outputs', today)
    os.makedirs(save_image_path, exist_ok=True)
    index = 0
    while os.path.exists(os.path.join(save_image_path, img_name + str(index) + '.png')):
        index += 1
    return index


def generate():
    """
    ./webui.sh --api会返回一个URL，这个URL是一个API接口，可以通过这个API接口来调用模型
    替换下面的txt2img_url为你的API接口，基本不用改。
    不加--nowebui 端口是7860，加了是7861，
    """
    txt2img_url = r'http://127.0.0.1:7860/sdapi/v1/txt2img'  # r表示原始字符串，没有转译

    """
    浏览器输入localhost:7860可以看到webui,
    http://127.0.0.1:7860/docs可以查看API文档，对应修改下面的json参数
    """

    angelMiku_data = {'prompt': 'masterpiece, best quality, hatsune miku, white gown, angel, angel wings, golden halo, '
                                'dark background, upper body, (closed mouth:1.2), looking at viewer, arms behind back, '
                                'blue theme, stars, starry night,incredibly absurdres,<lora:add_detail:0.6>',  # 正向提示词
                      'negative_prompt': '(low quality, worst quality:1.4), (FastNegativeEmbedding:0.9),((dyeing)),((oil painting)),((impasto))',  # 反向提示词
                      'sd_model_checkpoint': "aamAnyloraAnimeMixAnime_v1",  # 模型名称，在根目录/models/sStable-diffusion/中
                      'sampler_index': 'DPM++ SDE',  # 采样器
                      'scheduler': 'Karras',  # 噪声调度器
                      'batch_size': 4,  # 批大小
                      'n_iter': 1,  # 每批n个
                      'seed': -1,  # 种子
                      'steps': 20,  # 步数
                      'width': 512,
                      'height': 684,
                      'cfg_scale': 5}  # clip skip 默认2

    girInCar_data = {'prompt': '(masterpiece, best quality),1girl sitting in a car ,1girl, jewelry, smile, looking at viewer, car interior, solo,pink hair, purple eyes, steering wheel, blush, long hair, white shirt, off shoulder, black jacket, hair between eyes, long sleeves, , wrist scrunchie,',  # 正向提示词
                     'negative_prompt': '<lora:easynegative:1>,sketches,lowres,low quality,long body,long neck,extra limb,disconnected limbs,extra legs,fused fingers,too many fingers,disfigured,malformed limbs,blurry,',
                     # 反向提示词
                     'sd_model_checkpoint': "counterfeitV30_v30",  # 模型名称，在根目录/models/sStable-diffusion/中
                     'sampler_index': 'DPM++ SDE',  # 采样器
                     'scheduler': 'Karras',  # 噪声调度器
                     'batch_size': 4,  # 批大小
                     'n_iter': 1,  # 每批n个
                     'seed': 601687573,  # 种子
                     'steps': 25,  # 步数
                     'width': 512,
                     'height': 512,
                     'cfg_scale': 10}  # clip skip 默认2

    # 存储文件路径：./outputs/yyyy-mm-dd/xxxx.png
    images_name = "miku"  # xxxx
    today = get_today_date()

    # 创建一个新的线程，target参数是你想要在新线程中运行的函数
    # args参数是一个元组，包含了传递给target函数的参数
    thread = threading.Thread(target=submit_post, args=(txt2img_url, angelMiku_data))
    thread.start()
    while thread.is_alive():
        ProgressBar().show_progress()

    # response = submit_post(txt2img_url, imp_data)  # 在这里生成图片

    print(response.json())
    index_available = get_available_index(images_name, today)
    for i in range(4):  # No. 图片 = batch_size * n_iter
        save_image_name = images_name + str(index_available+i) + '.png'  # 文件名
        save_image_path = os.path.join('outputs', today, save_image_name)  # 文件路径

        save_encoded_image(response.json()['images'][i], save_image_path)  # 编码并保存图片


def main():
    generate()


if __name__ == '__main__':
    main()
