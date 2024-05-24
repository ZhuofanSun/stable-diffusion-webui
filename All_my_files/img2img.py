import traceback

from generate_uitl import Generate
import datafile as data
from utils import Utils

utils = Utils()

option_data = {}


# TODO: upscale 之前要改模型
# TODO: txt2img -> upscale 捆绑

def main():
    upscaler = ["None", "Lanczos", "Nearest", "4x-UltraSharp", "DAT x2", "DAT x3", "DAT x4", "LDSR",
                "R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B", "ScuNET GAN", "ScuNET PSNR", "SwinIR 4x"]

    # 检查webui能否访问，不能访问就启动webui
    if not utils.check_check_url():
        # 没连通就启动webui脚本
        # TODO: 脚本启动过程中
        print("-" * 20, "webui，启动！", "-" * 20)
        utils.start_webui()
    else:
        print("webui 运行中")

    generate = Generate()
    encoded_image = utils.read_image_to_base64("/Users/sunzhuofan/sdai/my_sd_webui/outputs/txt2img-images/2024-05-24"
                                               "/00164-684726996372598.png")
    test_data = {
        # 正向提示词
        "prompt": "incredibly absurdres,Sharpen,very high resolution,Anti-blur,Clear lines,Noise reduction,clear_edge,",
        # 反向提示词
        "negative_prompt": "(edge_blur:1.3),((dyeing)),((oil painting)),((impasto)),watercolor_(medium),blurry,"
                           "low quality,normal quality,worstquality,bad proportions,bad body,long body,long neck,"
                           "deformed,ugly,disfigured,poorly drawn face,extra limb,disconnected limbs,"
                           "easynegative",
        "sampler_name": "DPM++ 2M",  # 采样器
        "scheduler": "Karras",  # 噪声调度器
        "n_iter": 1,  # 生成批次
        "batch_size": 1,  # 每次张数  提升这个数值会显著增加使用内存
        "seed": -1,  # 种子
        "steps": 20,  # 步数
        "cfg_scale": 7,  # 引导词系数
        "width": 800,  # doesn't matter in upscaler
        "height": 800,  # doesn't matter in upscaler
        "denoising_strength": 0.23,  # 重绘幅度
        # 所有输入的 base64 图片 : 列表
        "init_images": [
            encoded_image
        ],
        "resize_mode": 1,  # ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]

        "script_name": "ultimate sd upscale",
        # /sdapi/v1/script-info 可以看到所有
        "script_args": [
            None,  # _ (not used)
            512,  # tile_width
            0,  # tile_height
            8,  # mask_blur
            32,  # padding
            64,  # seams_fix_width
            0.23,  # seams_fix_denoise  降噪幅度
            32,  # seams_fix_padding
            9,  # upscaler_index     # 9是R-ESRGAN 4x+ Anime6B， 3是4x-UltraSharp 看上面的列表
            'true',  # save_upscaled_image a.k.a Upscaled
            0,  # redraw_mode 0 是linear
            'false',  # save_seams_fix_image a.k.a Seams fix
            8,  # seams_fix_ma2sk_blur
            0,  # seams_fix_type  None
            2,  # target_size_type  # 0是根据img2img2设置, 1是自定义, 2是按输入图片倍率
            2048,  # custom_width
            2048,  # custom_height
            2  # custom_scale  # 1 ~ 16倍
        ],
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": "true",  # 开启ControlNet
                        "model": 'control_v11f1e_sd15_tile [a371b31b]',
                        "module": "tile_resample",  # 预处理器
                        "weight": "1",  # 权重
                        "resize_mode": 'Crop and Resize',  # Just Resize, Crop and Resize, Resize and Fill
                        "low_vram": "true",
                        "processor_res": 800,  # 预处理器分辨率, 决定了识别输入图片线条的粗细
                        # "threshold_a": 0,
                        # "threshold_b": 0,
                        "guidance_start": 0,  # 开始介入的时机
                        "guidance_end": 1,  # 结束介入的时机
                        "pixel_perfect": "false",
                        # "Balanced", "My prompt is more important", "ControlNet is more important"
                        "control_mode": 'Balanced',
                        # "Balanced", "My prompt is more important", "ControlNet is more important"

                    }
                ]
            }
        }
    }
    # normal test ----------------------------------
    generate.set_clip(2)
    generate.set_model('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
    generate.set_vae('klF8Anime2VAE_klF8Anime2VAE.safetensors')
    generate.post_option()
    generate.generate(url=utils.get_img2img_url(), image_data_list=[test_data], images_name='test')


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
