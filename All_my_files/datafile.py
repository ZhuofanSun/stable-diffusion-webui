import os
from utils import Utils

utils = Utils()
"""
model:
'abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]',
'cetusMix_Codaedition.safetensors [bd518b9aee]',
'counterfeitV30_v30.safetensors [17277fbe68]',
'hassakuHentaiModel_v13.safetensors [7eb674963a]',
'meinahentai_v4.safetensors [8145104977]',
'meinamix_meinaV10.safetensors [d967bcae4a]'

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


def get_angel_miku():
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
    return [angelMiku_data]


def get_girl_in_car():
    girlInCar_data = {
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
    return [girlInCar_data]


def get_imp(file_path):
    encoded_image = utils.read_image_to_base64(file_path)

    imp_data = {
        # 正向提示词
        'prompt': '1girl,imp,little_demon,anime sytle,<lora:add_detail:0.6>,4K,'
                  '(best illumination, an extremely delicate and beautiful),hyper detail,'
                  'best quality,high resolution,seductive_smile,naughty_face,',
        # 反向提示词
        'negative_prompt': '<lora:easynegative:1>,blurry,low quality,lowres,normal quality,'
                           'worstquality,bad proportions,bad body,long body,long neck,deformed,ugly,'
                           'extra limb,disconnected limbs,poorly drawn hands,',
        'sampler_index': 'DPM++ 2M',  # 采样器
        'scheduler': 'Karras',  # 噪声调度器
        'n_iter': 4,  # 生成批次
        'batch_size': 1,  # 每次张数  提升这个数值会显著增加使用内存
        'seed': -1,  # 种子
        'steps': 20,  # 步数
        'width': 512,  # 宽度
        'height': 512,  # 高度
        'cfg_scale': 7,  # 引导词规模
        # Control Net
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": "true",  # 开启ControNet
                        "model": 'control_v11f1p_sd15_depth [cfd03158]',
                        "module": "depth_anything",  # 预处理器
                        "weight": "1",  # 权重
                        "image": encoded_image,  # 输入图片，base64编码
                        "resize_mode": 'Crop and Resize',  # Just Resize, Crop and Resize, Resize and Fill
                        "low_vram": "true",
                        "processor_res": 512,  # 预处理器分辨率, 决定了识别输入图片线条的粗细
                        "threshold_a": 0,  # 阈值A leres++ remove near %
                        "threshold_b": 0,  # 阈值B leres++ remove background %
                        "guidance_start": 0,  # 开始介入的时机
                        "guidance_end": 1,  # 结束介入的时机
                        "pixel_perfect": "false",  #
                        "control_mode": 'My prompt is more important',  # Just Resize, Crop and Resize, Resize and Fill

                    }
                ]
            }
        }
    }
    return [imp_data]


def get_file_depth(file_path, multi=False):
    encoded_files_list = []
    if multi:
        # get files in the folder
        all_files = os.listdir(file_path)
        print(f"Encoding {len(all_files)} files...\n")
        # 文件夹下所有文件每5个分一组，组成一个列表
        for i in range(0, len(all_files), 5):
            files = all_files[i:i + 5]  # 列表切片的结束索引超过长度也不会报错，只会返回到最后一个元素

            files_encoded = []
            print("Encoding files: ")
            index = 0
            for file in files:
                print(i + index + 1, end='  ')
                encoded_image = utils.read_image_to_base64(os.path.join(file_path, file))
                files_encoded.append(encoded_image)
                index += 1

            encoded_files_list.append(files_encoded)
            if i + 5 + 1 < len(all_files):
                print(f"\nFile {i + 1} to File {i + 5 + 1} Done!\n")
            else:
                print(f"\nFile {i + 1} to File {len(all_files)} Done!\n")

    else:
        encoded_image = utils.read_image_to_base64(file_path)
        files = [encoded_image]
        encoded_files_list.append(files)

    depth_data_list = []
    for encoded_files in encoded_files_list:
        file_depth_data = {
            "controlnet_module": 'depth_anything',
            "controlnet_input_images": encoded_files,
            "controlnet_processor_res": 512,
            # "controlnet_threshold_a": 0,
            # "controlnet_threshold_b": 0,
            "controlnet_masks": [],
            "low_vram": 'true'
        }
        depth_data_list.append(file_depth_data)

    return depth_data_list


def get_leak():
    """有些预处理器有内存泄漏，关了Control net跑一个似乎就能清理了"""
    leak_data = {
        # 正向提示词
        'prompt': 'a dog',
        # 反向提示词
        'negative_prompt': '',
        'sampler_index': 'DPM++ SDE',  # 采样器
        'scheduler': 'Karras',  # 噪声调度器
        'batch_size': 1,  # 批大小
        'n_iter': 1,  # 每批n个
        'seed': -1,  # 种子
        'steps': 5,  # 步数
        'width': 64,  # 宽度
        'height': 64,  # 高度
        'cfg_scale': 7  # 引导词规模
    }
    return [leak_data]
