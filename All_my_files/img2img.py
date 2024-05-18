from generate_uitl import Generate


def main():
    """
    浏览器输入localhost:7860可以看到webui,
    http://127.0.0.1:7860/docs可以查看API文档，对应修改下面的json参数
    """
    # Clip 跳过层默认2，不知道怎么改
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
        'cfg_scale': 5  # 引导词规模
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
    for i in range(4, 11):
        angelMiku_data['cfg_scale'] = i
        Generate().generate(url=txt2img_url, image_data=angelMiku_data, images_name='angelMiku_testCFG')


if __name__ == '__main__':
    main()
