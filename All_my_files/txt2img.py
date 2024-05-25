import traceback

from generate_uitl import Generate
import datafile as data
from utils import Utils

utils = Utils()

option_data = {}


def txt2img_scale():
    generate = Generate(utils=utils)
    girl_in_car = data.get_girl_in_car()
    data.add_ad_hand(girl_in_car[0], "girl's hand", "bad hand")
    data.add_ad_face(girl_in_car[0], "detailed face, detailed eyes")
    data.add_high_resolution(girl_in_car[0], 1.5)

    generate.set_model('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
    generate.set_vae('klF8Anime2VAE_klF8Anime2VAE.safetensors')
    generate.post_option()

    generated_file = generate.generate(url=utils.get_txt2img_url(),
                                       image_data_list=girl_in_car,
                                       images_name="girl_in_car_scale1.5_")

    generated_file = generate.generate(url=utils.get_img2img_url(),
                                       image_data_list=data.get_file_upscale(generated_file[0], 1.3),
                                       images_name="girl_in_car_scale1.95_")

    generated_file = generate.generate(url=utils.get_img2img_url(),
                                       image_data_list=data.get_file_upscale(generated_file[0], 1.3),
                                       images_name="girl_in_car_scale2.5_")

    generate.generate(url=utils.get_img2img_url(),
                      image_data_list=data.get_file_upscale(generated_file[0], 2),
                      images_name="girl_in_car_scale5_")


def txt2img_test_cfg(data_json):
    generate = Generate(utils=utils)

    for i in range(4, 11):
        data_json[0]['cfg_scale'] = i
        generate.generate(url=utils.get_txt2img_url(), image_data_list=data_json, images_name='angelMiku_testCFG')


def txt2img_test_model(data_json):
    generate = Generate(utils=utils)

    for i in ['aamXLAnimeMix_v10.safetensors [d48c2391e0]'
              'abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]',
              'cetusMix_Codaedition.safetensors [bd518b9aee]',
              'counterfeitV30_v30.safetensors [17277fbe68]',
              'meinamix_meinaV10.safetensors [d967bcae4a]']:
        generate.set_model(i)
        generate.post_option()
        Generate().generate(url=utils.get_txt2img_url(), image_data_list=data_json, images_name='angelMiku_testModels')


def txt2img_test_vae(data_json):
    generate = Generate(utils=utils)
    generate.set_model('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
    for i in [None, 'clearvaeSD15_v23.safetensors', 'klF8Anime2VAE_klF8Anime2VAE.safetensors']:
        generate.set_vae(i)
        generate.post_option()
        generate.generate(url=utils.get_txt2img_url(), image_data_list=data_json, images_name='girInCar_testVAE')


def txt2img_test_blue():
    generate = Generate(utils=utils)

    girl_data = data.get_blue_girl()

    leak_data = data.get_leak()

    for _ in range(3):
        generate.set_model('aamXLAnimeMix_v10.safetensors [d48c2391e0]')
        generate.set_vae('None')
        generate.post_option()
        generate.generate(url=utils.get_txt2img_url(), image_data_list=girl_data, images_name="blue_girl")

        generate.set_model('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
        generate.set_vae('None')
        generate.post_option()
        generate.generate(url=utils.get_txt2img_url(), image_data_list=leak_data, images_name="leak")


def main():
    # 检查webui能否访问，不能访问就启动webui
    if not utils.check_check_url():
        # 没连通就启动webui脚本
        # TODO: 脚本启动过程中
        print("-" * 20, "webui，启动！", "-" * 20)
        utils.start_webui()
    else:
        print("webui 运行中")

    generate = Generate(utils=utils)

    # angelMiku_data = data.get_angel_miku()

    # girInCar_data = data.get_girl_in_car()

    # imp_data = data.get_imp(
    #     "/Users/sunzhuofan/sdai/my_sd_webui/outputs/txt2img-images/2024-05-17/00013-3727549206.png"
    # )

    # TODO: 一次调用api最多处理9张，否则报错
    # TODO：进行判断，如果大于5张，就分批处理 <- 一次9张api也有概率报错，干脆5张
    # depth_data = data.get_file_depth("/Users/sunzhuofan/sdai/my_sd_webui/outputs/txt2img-images/test")

    # tests ----------------------------------------
    # txt2img_test_cfg(angelMiku_data)
    # txt2img_test_model(angelMiku_data)
    # txt2img_test_vae(girInCar_data)
    # txt2img_test_blue()

    # normal test ----------------------------------
    # data.add_ad_face(imp_data[0], 'imp,seductive_smile')
    # generate.set_clip(2)
    # generate.set_model('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
    # generate.set_vae('klF8Anime2VAE_klF8Anime2VAE.safetensors')
    # generate.post_option()
    # generate.generate(url=utils.get_txt2img_url(), image_data_list=imp_data, images_name='imp')

    # depth test --------------------------------------
    # generate.generate(url=utils.get_controlnet_detect_url(), image_data_list=depth_data, images_name='depth')

    txt2img_scale()
    print("Running memory leak test")
    leak_data = data.get_leak()
    generate.generate(url=utils.get_txt2img_url(), image_data_list=leak_data, images_name='leak')


if __name__ == '__main__':
    try:
        main()
        utils.wait_3_secondes()
        utils.kill_script()
        utils.mem_collect()
        print("正常结束？")

    except ConnectionError as e:
        print("*" * 40, "  ConnectionError  ", "*" * 40)
        # 打印全部报错信息
        print(e)
        # 获取报错位置
        print("*" * 40, "连接问题，检查 webui.sh --api 执行情况", "*" * 40)
        utils.wait_3_secondes()
        utils.kill_script()  # 结束脚本
        utils.mem_collect()

    except KeyboardInterrupt:
        print("*" * 40, "  KeyboardInterrupt  ", "*" * 40)
        utils.wait_3_secondes()
        utils.kill_script()  # 结束脚本
        utils.mem_collect()

    except Exception as e:
        print("*" * 40, "  Exception  ", "*" * 40)
        traceback.print_exc()  # 打印堆栈跟踪信息
        utils.wait_3_secondes()
        utils.kill_script()  # 结束脚本
        utils.mem_collect()
