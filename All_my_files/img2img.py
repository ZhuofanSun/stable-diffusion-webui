import traceback

from generate_uitl import Generate
import datafile as data
from utils import Utils

utils = Utils()

option_data = {}


def main():
    # 检查webui能否访问，不能访问就启动webui
    if not utils.check_check_url():
        # 没连通就启动webui脚本
        # TODO: 脚本启动过程中
        print("-" * 20, "webui，启动！", "-" * 20)
        utils.start_webui()
    else:
        print("webui 运行中")

    generate = Generate()

    test_data = {
        # 正向提示词
        "prompt": "",
        # 反向提示词
        "negative_prompt": "",
        "sampler_name": "string",  # 采样器
        "scheduler": "string",  # 噪声调度器
        "n_iter": 1,  # 生成批次
        "batch_size": 1,  # 每次张数  提升这个数值会显著增加使用内存
        "seed": -1,  # 种子
        "steps": 20,  # 步数
        "cfg_scale": 7,  # 引导词系数
        "width": 512,
        "height": 512,
        "denoising_strength": 0.75,  # 重绘幅度
        # 所有输入的 base64 图片 : 列表
        "init_images": [
            "string"
        ],
        "resize_mode": 1,  # ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]

        "script_name": "",
        "script_args": [],
        "alwayson_scripts": {},
    }
    # normal test ----------------------------------
    generate.set_clip(2)
    generate.set_model('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
    generate.set_vae('klF8Anime2VAE_klF8Anime2VAE.safetensors')
    generate.post_option()
    # generate.generate(url=utils.get_txt2img_url(), image_data_list=imp_data, images_name='imp')


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

    except Exception as e:
        print("*" * 40, "  Exception  ", "*" * 40)
        traceback.print_exc()  # 打印堆栈跟踪信息
        utils.wait_3_secondes()
        utils.kill_script()  # 结束脚本
        utils.mem_collect()

    except KeyboardInterrupt:
        print("*" * 40, "  KeyboardInterrupt  ", "*" * 40)
        utils.wait_3_secondes()
        utils.kill_script()  # 结束脚本
        utils.mem_collect()
