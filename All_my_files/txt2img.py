import traceback

from generate_uitl import Generate
import datafile as data
from utils import Utils

utils = Utils()

option_data = {}


def txt2img_test_cfg(data_json):
    generate = Generate()

    for i in range(4, 11):
        data_json['cfg_scale'] = i
        generate.generate(url=utils.get_txt2img_url(), image_data=data_json, images_name='angelMiku_testCFG')


def txt2img_test_model(data_json):
    generate = Generate()

    for i in ['abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]',
              'cetusMix_Codaedition.safetensors [bd518b9aee]',
              'counterfeitV30_v30.safetensors [17277fbe68]',
              'hassakuHentaiModel_v13.safetensors [7eb674963a]',
              'meinahentai_v4.safetensors [8145104977]',
              'meinamix_meinaV10.safetensors [d967bcae4a]']:
        generate.set_model(i)
        generate.post_option()
        Generate().generate(url=utils.get_txt2img_url(), image_data=data_json, images_name='angelMiku_testModels')


def txt2img_test_vae(data_json):
    generate = Generate()
    generate.set_model('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
    for i in [None, 'clearvaeSD15_v23.safetensors', 'klF8Anime2VAE_klF8Anime2VAE.safetensors']:
        generate.set_vae(i)
        generate.post_option()
        Generate().generate(url=utils.get_txt2img_url(), image_data=data_json, images_name='girInCar_testVAE')


def main():
    # 检查webui能否访问，不能访问就启动webui
    if not utils.check_check_url():
        # 没连通就启动webui脚本
        # TODO: 脚本启动过程中
        print("-" * 20, "webui，启动！", "-" * 20)
        utils.start_webui()
    else:
        print("webui 运行中")

    """
    浏览器输入localhost:7860可以看到webui,
    http://127.0.0.1:7860/docs可以查看API文档，对应修改下面的json参数
    """
    generate = Generate()

    # angelMiku_data = data.get_angel_miku()

    # girInCar_data = data.get_girl_in_car()

    # imp_data = data.get_imp(
    #     "/Users/sunzhuofan/sdai/my_sd_webui/outputs/img2img-images/2024-05-17/00000-2888002140.jpg"
    # )

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
        utils.wait_5_secondes()
        utils.kill_script()
        utils.mem_collect()
        print("正常结束？")

    except ConnectionError as e:
        print("*" * 40, "  ConnectionError  ", "*" * 40)
        # 打印全部报错信息
        print(e)
        # 获取报错位置
        print("*" * 40, "连接问题，检查 webui.sh --api 执行情况", "*" * 40)
        utils.wait_5_secondes()
        utils.kill_script()  # 结束脚本
        utils.mem_collect()

    except Exception as e:
        print("*" * 40, "  Exception  ", "*" * 40)
        traceback.print_exc()  # 打印堆栈跟踪信息

        utils.wait_5_secondes()
        utils.kill_script()  # 结束脚本
        utils.mem_collect()

    except KeyboardInterrupt:
        print("*" * 40, "  KeyboardInterrupt  ", "*" * 40)
        utils.wait_5_secondes()  # TODO：不加这个在下面结束脚本时会连着pycharm一起杀掉，抽象
        utils.kill_script()  # 结束脚本
        utils.mem_collect()

