import requests
from generate_uitl import Generate

option_data = {}
txt2img_url = r'http://127.0.0.1:7860/sdapi/v1/txt2img'
options_url = r'http://127.0.0.1:7860/sdapi/v1/options'


def set_clip(new_clip=None):
    global option_data

    if new_clip is None:
        pass
    else:
        option_data["CLIP_stop_at_last_layers"] = new_clip


def set_model(new_model=None):
    global option_data

    if new_model is None:
        pass
    else:
        option_data["sd_model_checkpoint"] = new_model


def set_vae(new_vae=None):
    global option_data

    if new_vae is None:
        pass
    else:
        option_data["sd_vae"] = new_vae


def post_option():
    global option_data, options_url

    if 'sd_vae' not in option_data.keys():
        option_data['sd_vae'] = None

    requests.post(url=options_url, json=option_data)


def txt2img_test_cfg(data):
    global txt2img_url

    for i in range(4, 11):
        data['cfg_scale'] = i
        Generate().generate(url=txt2img_url, image_data=data, images_name='angelMiku_testCFG')


def txt2img_test_model(data):
    global txt2img_url

    for i in ['abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]',
              'cetusMix_Codaedition.safetensors [bd518b9aee]',
              'counterfeitV30_v30.safetensors [17277fbe68]',
              'hassakuHentaiModel_v13.safetensors [7eb674963a]',
              'meinahentai_v4.safetensors [8145104977]',
              'meinamix_meinaV10.safetensors [d967bcae4a]']:
        set_model(i)
        post_option()
        Generate().generate(url=txt2img_url, image_data=data, images_name='angelMiku_testModels')


def txt2img_test_vae(data):
    set_model('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
    for i in [None, 'clearvaeSD15_v23.safetensors', 'klF8Anime2VAE_klF8Anime2VAE.safetensors']:
        set_vae(i)
        post_option()
        Generate().generate(url=txt2img_url, image_data=data, images_name='girInCar_testVAE')


def main():
    """
    浏览器输入localhost:7860可以看到webui,
    http://127.0.0.1:7860/docs可以查看API文档，对应修改下面的json参数
    """
    # Clip 能在webui改，会记录，不知道怎么在api改
    # Update：
    # 这个参数在config.json文件中的CLIP_stop_at_last_layers参数进行设置，
    # 默认好像是1，需要在启动webui服务之前将这个参数设置好。下边那个方法能改但是要重启服务
    # update_clip(3)

    angelMiku_data = {
        # 正向提示词
        'prompt': 'masterpiece, best quality, hatsune miku, white gown, angel, angel wings, golden halo,'
                  'dark background, upper body, (closed mouth:1.2), looking at viewer, arms behind back,'
                  'blue theme, stars, starry night,incredibly absurdres,<lora:add_detail:0.6>',
        # 反向提示词
        'negative_prompt': '(low quality, worst quality:1.4), (FastNegativeEmbedding:0.9),'
                           '((dyeing)),((oil painting)),((impasto))',
        'sampler_index': 'DPM++ SDE',  # 采样器
        'scheduler': 'Karras',  # 噪声调度器
        'batch_size': 1,  # 批大小
        'n_iter': 1,  # 每批n个
        'seed': 1836097496,  # 种子
        'steps': 20,  # 步数
        'width': 512,  # 宽度
        'height': 684,  # 高度
        'cfg_scale': 7  # 引导词规模
    }

    girInCar_data = {
        # 正向提示词
        'prompt': '(masterpiece, best quality),1girl sitting in a car ,1girl, jewelry, smile, looking at viewer, '
                  'car interior, solo,pink hair, purple eyes, steering wheel, blush, long hair, white shirt, '
                  'off shoulder, black jacket, hair between eyes, long sleeves, , wrist scrunchie,',
        # 反向提示词
        'negative_prompt': '<lora:easynegative:1>,sketches,lowres,low quality,long body,long neck,extra limb,'
                           'disconnected limbs,extra legs,fused fingers,too many fingers,'
                           'disfigured,malformed limbs,blurry,',
        'sampler_index': 'DPM++ SDE',  # 采样器
        'scheduler': 'Karras',  # 噪声调度器
        'batch_size': 1,  # 批大小
        'n_iter': 1,  # 每批n个
        'seed': 601687573,  # 种子
        'steps': 25,  # 步数
        'width': 512,  # 宽度
        'height': 512,  # 高度
        'cfg_scale': 10  # 引导词规模
    }

    # set_clip(2)
    # set_model('counterfeitV30_v30.safetensors [17277fbe68]')
    # post_option(options_url)
    # Generate().generate(url=txt2img_url, image_data=angelMiku_data, images_name='angelMiku')

    # txt2img_test_cfg(angelMiku_data)
    # txt2img_test_model(angelMiku_data)
    txt2img_test_vae(girInCar_data)


if __name__ == '__main__':
    main()
