import json
import base64
import requests


def submit_post(url: str, data: dict):
    return requests.post(url, data=json.dumps(data))


def save_encoded_image(b64_image: str, output_path: str):
    with open(output_path, 'wb') as image_file:
        image_file.write(base64.b64decode(b64_image))

"""
./webui.sh --api会返回一个URL，这个URL是一个API接口，可以通过这个API接口来调用模型
替换下面的txt2img_url为你的API接口，基本不用改。 
不加--nowebui 端口是7860，加了是7861，
"""
if __name__ == '__main__':
    txt2img_url = r'http://127.0.0.1:7860/sdapi/v1/txt2img'

"""
浏览器输入localhost:7860可以看到webui,
http://127.0.0.1:7860/docs可以查看API文档，对应修改下面的json参数
"""

data = {'prompt': '1girl, 1boy, student, uniform',  # 正向提示词
        'negative_prompt': '',  # 反向提示词
        'sd_model_checkpoint': "aamAnyloraAnimeMixAnime_v1",  # 模型名称，在根目录/models/sStable-diffusion/中
        'sampler_index': 'DPM++ SDE',  # 采样器以及噪声调节器
        'batch_size': 2,
        'n_iter': 2,
        'seed': -1,
        'steps': 20,
        'width': 512,
        'height': 512,
        'cfg_scale': 8}

response = submit_post(txt2img_url, data)
print(response.json())
for i in range(4):
    save_image_path = r'tmp' + str(i) + '.png'
    save_encoded_image(response.json()['images'][i], save_image_path)