import os
from utils import Utils

utils = Utils()
"""
model:
'aamXLAnimeMix_v10.safetensors [d48c2391e0]
'abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]',
'cetusMix_Codaedition.safetensors [bd518b9aee]',
'counterfeitV30_v30.safetensors [17277fbe68]',
'meinamix_meinaV10.safetensors [d967bcae4a]'

## ControlNet 模型 & 预处理器:
'control_v11f1e_sd15_tile [a371b31b]'  # 放大模型
    'tile_resample'       # 分块控制
'control_v11f1p_sd15_depth [cfd03158]'  # depth模型
    'depth_midas'         # 细节-，背景-
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
    
ADetailer 模型：
    'face_yolov8n.pt'       # 2D / 真实人脸 nano
    'face_yolov8s.pt'       # 2D / 真实人脸 small
    'face_yolov8m.pt'       # 2D / 真实人脸 medium
    'hand_yolov8n.pt'       # 2D / 真实人手 nano
    'hand_yolov9c.py'       # 2D / 真实人手
    'person_yolov8n-seg.pt' # 2D / 真实全身 
    'person_yolov8s-seg.pt' # 2D / 真实全身
    'yolov8x-worldv2.pt'    # 2D / 世界
    'mediapipe_face_full'   # 真实人脸
    'mediapipe_face_short'  # 真实人脸
    'mediapipe_face_mesh'   # 立体 / 真实人脸 
    'mediapipe_face_mesh_eyes_only'  # 眼睛
    'None'
"""


def get_template_data():
    template_data = {
        # 正向提示词
        'prompt': '',
        # 反向提示词
        'negative_prompt': '',
        'sampler_index': 'DPM++ SDE',  # 采样器
        'scheduler': 'Karras',  # 噪声调度器
        'batch_size': 1,  # 批大小
        'n_iter': 1,  # 每批n个
        'seed': -1,  # 种子
        'steps': 20,  # 步数
        'width': 512,  # 宽度
        'height': 512,  # 高度
        'cfg_scale': 7,  # 引导词系数
        "alwayson_scripts": {}
    }
    return template_data


def set_prompt(data, prompt):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(prompt, str):
        raise ValueError("prompt必须是字符串")

    data["prompt"] = prompt


def set_negative_prompt(data, negative_prompt):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(negative_prompt, str):
        raise ValueError("negative_prompt必须是字符串")

    data["negative_prompt"] = negative_prompt


def set_sampling_method(data, sampler_index):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(sampler_index, str):
        raise ValueError("sampler_index 必须是字符串")

    data["sampler_index"] = sampler_index


def set_scheduler(data, scheduler=None):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(scheduler, str):
        raise ValueError("scheduler必须是字符串")

    data["scheduler"] = scheduler


def set_batch_size(data, batch_size=None):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(batch_size, int):
        raise ValueError("batch_size必须是整数")

    data["batch_size"] = batch_size


def set_n_iter(data, n_iter=None):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(n_iter, int):
        raise ValueError("n_iter必须是整数")

    data["n_iter"] = n_iter


def set_seed(data, seed=None):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(seed, int):
        raise ValueError("seed必须是整数")

    data["seed"] = seed


def set_steps(data, steps=None):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(steps, int):
        raise ValueError("steps必须是整数")

    data["steps"] = steps


def set_width(data, width=None):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(width, int):
        raise ValueError("width必须是整数")

    data["width"] = width


def set_height(data, height=None):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(height, int):
        raise ValueError("height必须是整数")

    data["height"] = height


def set_cfg_scale(data, cfg_scale=None):
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")
    if not isinstance(cfg_scale, int):
        raise ValueError("cfg_scale必须是整数")

    data["cfg_scale"] = cfg_scale


def add_ad_hand(data, prompt="", neg_prompt="", skip_img2img=False):
    """
    给一个图片data数据添加手部adetailer优化
    :param data: 图片信息json文件
    :param prompt: 附加正面提示词
    :param neg_prompt: 附加负面提示词
    :param skip_img2img: 是否跳过img2img
    :return: 加入adetailer优化后的图片信息json文件
    """
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")

    # 检查data["alwayson_scripts"]["ADetailer"] 是否存在
    if "alwayson_scripts" not in data.keys() or "ADetailer" not in data["alwayson_scripts"].keys():
        data["alwayson_scripts"] = {"ADetailer": {"args": ['true', skip_img2img]}}
    data["alwayson_scripts"]["ADetailer"]["args"][1] = skip_img2img

    ad_arg = {
        "ad_tap_enable": 'true',  # 开启当前tap，这样的优化（args列表加更多的字典）可以开多个
        "ad_model": "hand_yolov8n.pt",  # AD模型
        "ad_prompt": prompt + ",detailed,best quality,"
                              "(beautiful detailed face),beautiful eyes,digital painting,",  # ad的正面提示词，空着就是用图片的
        "ad_negative_prompt": neg_prompt + "negative_hand-neg,easynegative,(low quality:1.4),(oil painting),(brush "
                                           "strokes),(greyscale:1.2),(worst quality:1.4),(monochrome:1.1),lowres,"
                                           "worstquality,mutated,grayscale,sketches,spot_color,chromatic_aberration,",
        "ad_denoising_strength": 0.5,  # 重绘幅度

        "ad_confidence": 0.6,  # 高于ai识别置信度的才会重绘，多目标重绘可以用
        "ad_use_inpaint_width_height": 'false',  # 单独设置inpaint的宽高
        "ad_inpaint_width": 512,
        "ad_inpaint_height": 512,
        "ad_use_steps": 'false',  # 单独设置步数
        "ad_steps": 28,
        "ad_use_cfg_scale": 'false',  # 单独设置cfg
        "ad_cfg_scale": 7.0,
        "ad_use_checkpoint": 'false',  # 单独设置模型
        "ad_checkpoint": "Use same checkpoint",
        "ad_use_vae": 'false',  # 单独设置vae
        "ad_vae": "Use same VAE",
        "ad_use_sampler": 'false',  # 单独设置采样器
        "ad_sampler": "DPM++ 2M Karras",
        "ad_use_clip_skip": 'false',  # 单独设置clip
        "ad_clip_skip": 1,
    }
    # 由于字典是可变对象，直接修改就行
    data["alwayson_scripts"]["ADetailer"]["args"].append(ad_arg)


def add_ad_face(data, prompt="", neg_prompt="", skip_img2img=False):
    """
    给一个图片data数据添加人脸adetailer优化
    :param data: 图片信息json文件
    :param prompt: 附加正面提示词
    :param neg_prompt: 附加负面提示词
    :param skip_img2img: 是否跳过img2img
    :return: 加入adetailer优化后的图片信息json文件
    """
    if not isinstance(data, dict):
        raise ValueError("data必须是字典/json")

    # 检查data["alwayson_scripts"]["ADetailer"] 是否存在
    if "ADetailer" not in data["alwayson_scripts"].keys():
        data["alwayson_scripts"]["ADetailer"] = {"args": ['true', skip_img2img]}

    ad_arg = {
        "ad_tap_enable": 'true',  # 开启当前tap，这样的优化（args列表加更多的字典）可以开多个
        "ad_model": "face_yolov8s.pt",  # AD模型
        "ad_prompt": prompt + ",detailed,best quality,"
                              "(beautiful detailed face),beautiful eyes,digital painting,",  # ad的正面提示词，空着就是用图片的
        "ad_negative_prompt": neg_prompt + ",(low quality:1.4),bad eyes,(oil painting),(brush strokes),"
                                           "(greyscale:1.2),(worst quality:1.4),(monochrome:1.1),lowres,"
                                           "worstquality,mutated,grayscale,sketches,spot_color,"
                                           "chromatic_aberration,"
                                           "black and white,easynegative,",  # 负面同理
        "ad_denoising_strength": 0.35,  # 重绘幅度

        "ad_confidence": 0.6,  # 高于ai识别置信度的才会重绘，多目标重绘可以用
        "ad_use_inpaint_width_height": 'false',  # 单独设置inpaint的宽高
        "ad_inpaint_width": 512,
        "ad_inpaint_height": 512,
        "ad_use_steps": 'false',  # 单独设置步数
        "ad_steps": 28,
        "ad_use_cfg_scale": 'false',  # 单独设置cfg
        "ad_cfg_scale": 7.0,
        "ad_use_checkpoint": 'false',  # 单独设置模型
        "ad_checkpoint": "Use same checkpoint",
        "ad_use_vae": 'false',  # 单独设置vae
        "ad_vae": "Use same VAE",
        "ad_use_sampler": 'false',  # 单独设置采样器
        "ad_sampler": "DPM++ 2M Karras",
        "ad_use_clip_skip": 'false',  # 单独设置clip
        "ad_clip_skip": 1,
    }
    # 由于字典是可变对象，直接修改就行
    data["alwayson_scripts"]["ADetailer"]["args"].append(ad_arg)


def add_high_resolution(data, scale):
    """
    set hr_enable to true
    :param data:input datafile : dict
    :param scale: scale of the image : int
    :return: None
    """
    data["enable_hr"] = "true"
    data["hr_scale"] = scale
    data["hr_upscaler"] = "R-ESRGAN 4x+ Anime6B"  # anime专用
    # data["hr_resize_x"] = 0  # 设置为0时不启用
    # data["hr_resize_y"] = 0  # 设置为0时不启用
    data["denoising_strength"] = 0.4  # hr重绘幅度


def add_controlnet_depth(data, file_path, module="depth_anything", threshold_a=0, threshold_b=0):
    # 'depth_midas'         # 细节-，背景-
    # 'depth_zoe'           # 细节-，背景-
    # 'depth_leres++'       # 细节++，可以调remove near, remove background
    # 'depth_leres'         # 细节+
    # 'depth_anything'      # 细节，背景
    encoded_image = utils.read_image_to_base64(file_path)
    # get file resolution
    res = min(utils.get_image_size(file_path))

    depth_data = {
        "args": [
            {
                "enabled": "true",  # 开启ControlNet
                "model": 'control_v11f1p_sd15_depth [cfd03158]',
                "module": module,  # 预处理器
                "weight": "1",  # 权重
                "image": encoded_image,  # 输入图片，base64编码
                "resize_mode": 'Crop and Resize',  # Just Resize, Crop and Resize, Resize and Fill
                "low_vram": "true",
                "processor_res": res,  # 预处理器分辨率, 决定了识别输入图片线条的粗细
                "guidance_start": 0,  # 开始介入的时机
                "guidance_end": 1,  # 结束介入的时机
                "pixel_perfect": "false",  #
                "control_mode": 'My prompt is more important',  # Just Resize, Crop and Resize, Resize and Fill

            }
        ]
    }
    if module == "depth_leres++":
        depth_data["controlnet"]["args"][0]["threshold_a"] = threshold_a
        depth_data["controlnet"]["args"][0]["threshold_b"] = threshold_b

    data["alwayson_scripts"]["controlnet"] = depth_data


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
        'width': 768,  # 宽度
        'height': 1026,  # 高度
        'cfg_scale': 7,  # 引导词系数
        "alwayson_scripts": {}
    }
    return [angelMiku_data]


def get_girl_in_car():
    girlInCar_data = {
        # 正向提示词
        'prompt': '(masterpiece, best quality),1girl sitting in a car ,1girl, jewelry, smile, looking at viewer, '
                  'car interior, solo,pink hair, purple eyes, steering wheel, blush, long hair, white shirt, '
                  'off shoulder, black jacket, hair between eyes, long sleeves, , wrist scrunchie,',
        # 反向提示词
        'negative_prompt': 'easynegative,sketches,lowres,low quality,long body,long neck,extra limb,'
                           'disconnected limbs,extra legs,fused fingers,too many fingers,'
                           'disfigured,malformed limbs,blurry,',
        'sampler_index': 'DPM++ SDE',  # 采样器
        'scheduler': 'Karras',  # 噪声调度器
        'batch_size': 1,  # 批大小
        'n_iter': 1,  # 每批n个
        'seed': -1,  # 种子
        'steps': 25,  # 步数
        'width': 512,  # 宽度
        'height': 512,  # 高度
        'cfg_scale': 7,  # 引导词系数
        "alwayson_scripts": {}
    }
    return [girlInCar_data]


def get_imp(file_path):
    imp_data = {
        # 正向提示词
        'prompt': '1girl,imp,little_demon,anime sytle,4K,'
                  '(best illumination, an extremely delicate and beautiful),hyper detail,'
                  'best quality,high resolution,seductive_smile,naughty_face,(masterpiece, best quality, '
                  'ultra-detailed, best shadow)',
        # 反向提示词
        'negative_prompt': 'easynegative,blurry,low quality,lowres,normal quality,'
                           'worstquality,bad proportions,bad body,long body,long neck,deformed,ugly,'
                           'extra limb,disconnected limbs,poorly drawn hands,',
        'sampler_index': 'DPM++ 2M',  # 采样器
        'scheduler': 'Karras',  # 噪声调度器
        'n_iter': 2,  # 生成批次
        'batch_size': 1,  # 每次张数  提升这个数值会显著增加使用内存
        'seed': -1,  # 种子
        'steps': 20,  # 步数
        'width': 512,  # 宽度
        'height': 512,  # 高度
        'cfg_scale': 7,  # 引导词系数
        "alwayson_scripts": {}
    }
    add_controlnet_depth(imp_data, file_path)
    print(imp_data)
    return [imp_data]


def get_file_depth(file_path):
    # check file_path is a file or a folder
    multi = True if os.path.isdir(file_path) else False

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
            # "controlnet_threshold_a": 0,  # 0～100
            # "controlnet_threshold_b": 0,  # 0～100
            "controlnet_masks": [],
            "low_vram": 'true'
        }
        depth_data_list.append(file_depth_data)

    return depth_data_list


def get_blue_girl():
    blue_girl = {
        # 正向提示词
        'prompt': 'anime girl,night,blue light behind her,((Galaxy, Lens flare)),short hair,flower field,night sky,'
                  'cinematic shot. Wallpaper. (Blue color schema),detailed background,a city in the distance,'
                  '<lora:add_detail:0.4>,4K,high resolution,',
        # 反向提示词
        'negative_prompt': 'negativeXL_D,cgi,3d render,bad quality,worst quality,text,signature,watermark,extra limbs,',
        'sampler_index': 'Euler a',  # 采样器
        'scheduler': 'Karras',  # 噪声调度器
        'batch_size': 1,  # 批大小
        'n_iter': 1,  # 每批n个
        'seed': -1,  # 种子
        'steps': 28,  # 步数
        'width': 970,  # 宽度
        'height': 970,  # 高度
        'cfg_scale': 7,  # 引导词系数
        "alwayson_scripts": {}
    }
    return [blue_girl]


def get_file_upscale(file_path, scale=2):
    """
    返回图片放大的数据
    :param file_path: 需要放大的图片路径 / 文件夹路径
    :param scale: 放大倍率
    :return: 数据列表
    """

    def fill_file_data(width, height, base64image):
        data = {
            # 正向提示词
            "prompt": "incredibly absurdres,Sharpen,very high resolution,Anti-blur,Clear lines,Noise reduction,"
                      "clear_edge,",
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
            "width": width * scale,  # doesn't matter in upscaler
            "height": height * scale,  # doesn't matter in upscaler
            "denoising_strength": 0.23,  # 重绘幅度
            # 所有输入的 base64 图片 : 列表
            "init_images": [
                base64image
            ],
            "resize_mode": 1,  # ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]

            "script_name": "ultimate sd upscale",
            # /sdapi/v1/script-info 可以看到所有
            "script_args": [
                None,  # _ (not used)
                # 按照这个tile大小分割图像，太大了占内存
                512,  # tile_width
                512,  # tile_height
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
                width * scale,  # custom_width  上面选1才填这个，否则不重要
                height * scale,  # custom_height  上面选1才填这个，否则不重要
                scale  # custom_scale  # 1 ~ 16倍  上面选2才填这个，否则不重要
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
                            "processor_res": min(width, height),  # 预处理器分辨率, 决定了识别输入图片线条的粗细
                            # "threshold_a": 0,
                            # "threshold_b": 0,
                            "guidance_start": 0,  # 开始介入的时机
                            "guidance_end": 1,  # 结束介入的时机
                            "pixel_perfect": "false",
                            # "Balanced", "My prompt is more important", "ControlNet is more important"
                            "control_mode": 'Balanced',

                        }
                    ]
                }
            }
        }
        return data

    upscale_data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件/文件夹不存在：{file_path}")

    if os.path.isdir(file_path):
        for image_path in os.listdir(file_path):
            encoded_image = utils.read_image_to_base64(os.path.join(file_path, image_path))
            image_width, image_height = utils.get_image_size(os.path.join(file_path, image_path))  # 获取图片大小

            # 创造data并加入列表
            upscale_data.append(fill_file_data(image_width, image_height, encoded_image))

    else:
        encoded_image = utils.read_image_to_base64(file_path)
        image_width, image_height = utils.get_image_size(file_path)

        upscale_data.append(fill_file_data(image_width, image_height, encoded_image))

    return upscale_data


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
        'cfg_scale': 7  # 引导词系数
    }
    return [leak_data]
