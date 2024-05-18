from generate_uitl import Generate
import json
import os


def update_clip(new_value):
    # 获取当前文件夹的上一层目录
    parent_dir = os.path.dirname(os.getcwd())
    # 拼接得到config.json的完整路径
    config_path = os.path.join(parent_dir, 'config.json')

    # 打开并读取config.json文件
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 更新CLIP_stop_at_last_layers的值
    config['CLIP_stop_at_last_layers'] = new_value

    # 将更新后的config写回文件
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


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
        # 模型名称 models/sStable-diffusion/
        'sd_model_checkpoint': "aamAnyloraAnimeMixAnime_v1",
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
        # 模型名称 models/sStable-diffusion/
        'sd_model_checkpoint': "counterfeitV30_v30",
        'sampler_index': 'DPM++ SDE',  # 采样器
        'scheduler': 'Karras',  # 噪声调度器
        'batch_size': 4,  # 批大小
        'n_iter': 1,  # 每批n个
        'seed': 601687573,  # 种子
        'steps': 25,  # 步数
        'width': 512,  # 宽度
        'height': 512,  # 高度
        'cfg_scale': 10  # 引导词规模
    }

    txt2img_url = r'http://127.0.0.1:7860/sdapi/v1/txt2img'
    # for i in range(4, 11):
    #     angelMiku_data['cfg_scale'] = i
    #     Generate().generate(url=txt2img_url, image_data=angelMiku_data, images_name='angelMiku_testCFG')
    Generate().generate(url=txt2img_url, image_data=angelMiku_data, images_name='angelMiku_testCFG')


if __name__ == '__main__':
    main()
