from generate_uitl import Generate
import datafile as data
from utils import Utils
import gc

utils = Utils()

option_data = {}


def txt2img_test_cfg(data):
    generate = Generate()

    for i in range(4, 11):
        data['cfg_scale'] = i
        generate.generate(url=utils.get_txt2img_url(), image_data=data, images_name='angelMiku_testCFG')


def txt2img_test_model(data):
    generate = Generate()

    for i in ['abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]',
              'cetusMix_Codaedition.safetensors [bd518b9aee]',
              'counterfeitV30_v30.safetensors [17277fbe68]',
              'hassakuHentaiModel_v13.safetensors [7eb674963a]',
              'meinahentai_v4.safetensors [8145104977]',
              'meinamix_meinaV10.safetensors [d967bcae4a]']:
        generate.set_model(i)
        generate.post_option()
        Generate().generate(url=utils.get_txt2img_url(), image_data=data, images_name='angelMiku_testModels')


def txt2img_test_vae(data):
    generate = Generate()
    generate.set_model('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
    for i in [None, 'clearvaeSD15_v23.safetensors', 'klF8Anime2VAE_klF8Anime2VAE.safetensors']:
        generate.set_vae(i)
        generate.post_option()
        Generate().generate(url=utils.get_txt2img_url(), image_data=data, images_name='girInCar_testVAE')


def main():
    # 启动webui
    # utils.start_webui()

    """
    浏览器输入localhost:7860可以看到webui,
    http://127.0.0.1:7860/docs可以查看API文档，对应修改下面的json参数
    """
    generate = Generate()

    angelMiku_data = data.get_angel_miku()

    girInCar_data = data.get_girl_in_car()

    imp_data = data.get_imp("/Users/sunzhuofan/sdai/my_sd_webui/outputs/img2img-images/2024-05-17/00000-2888002140.jpg")

    # TODO: 文件夹里图片太多会报错  最多九张
    depth_data = data.get_file_depth("/Users/sunzhuofan/sdai/my_sd_webui/outputs/txt2img-images/2024-05-21", multi=True)
    # tests ----------------------------------------
    # txt2img_test_cfg(angelMiku_data)
    # txt2img_test_model(angelMiku_data)
    # txt2img_test_vae(girInCar_data)

    # normal test ----------------------------------
    # generate.set_clip(2)
    # generate.set_model('cetusMix_Codaedition.safetensors [bd518b9aee]')
    # generate.set_vae('klF8Anime2VAE_klF8Anime2VAE.safetensors')
    # generate.post_option()
    # generate.generate(url=utils.get_txt2img_url(), image_data=imp_data, images_name='imp')

    # depth test --------------------------------------
    generate.generate(url=utils.get_controlnet_detect_url(), image_data=depth_data, images_name='depth')

    print("Running memory leak test")
    leak_data = data.get_leak()
    generate.generate(url=utils.get_txt2img_url(), image_data=leak_data, images_name='leak')

    # 可以用ControlNet.get_modulelist看所有信息
    """
    ## ControlNet 模型 & 预处理器:
    'control_v11f1e_sd15_tile [a371b31b]'  # 放大模型
        'tile_resample'       # 分块控制
    'control_v11f1p_sd15_depth [cfd03158]'  # depth模型
        'depth_zoe'           # 细节-，背景-
        'depth_leres++'       # 细节++，可以调remove near, remove background
        'depth_leres'         # 细节+
        'depth_hand_refiner'  #
        'depth_anything'      # 细节，背景

    'control_v11p_sd15_openpose [cab727d4]'  # openpose模型
        'openpose_full'       # 姿态+面部+手部
        'openpose_faceonly'   # 仅面部
        'openpose_face'       # 姿态+面部
        'openpose'            # 仅姿态
        'openpose_hand'       # 姿态+手部
        'dw_openpose_full'    # dwpose算法，姿态+面部+手
    """


if __name__ == '__main__':
    try:
        main()
        utils.kill_script()
        # Print the number of objects known by the collector, before and after a collection
        print("Objects before collection: ", gc.get_count())
        gc.collect()
        print("Objects after collection: ", gc.get_count())

    except Exception as e:
        print(e)
        utils.kill_script()  # 结束脚本
        # Print the number of objects known by the collector, before and after a collection
        print("Objects before collection: ", gc.get_count())
        gc.collect()
        print("Objects after collection: ", gc.get_count())
    except KeyboardInterrupt as k:
        print(k)
        utils.kill_script()  # 结束脚本
        # Print the number of objects known by the collector, before and after a collection
        print("Objects before collection: ", gc.get_count())
        gc.collect()
        print("Objects after collection: ", gc.get_count())
