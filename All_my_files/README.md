# Open-webui & Stable diffusion

***


## 目录
1. **[模型 & 数据集下载](#模型数据集下载)**
    - **[资源站](#资源站)**
    - **[下载](#下载)**
2. **[Ollama](#Ollama)**
    - **[常见指令](#常见指令)**
    - **[添加新模型](#添加新模型)**
3. **[Open-webui](#Open-webui)**
    - **[向docker添加Webui容器](#向docker添加Webui容器)**
    - **[启动/关闭Webui容器](#启动关闭webui容器)**
    - **[在Open-webui中使用Stable-diffusion](#在Open-webui中使用Stable-diffusion)**
4. **[Stable diffusion](#Stable-diffusion)**
    - **[从automatic1111获取更新](#从automatic1111获取更新)**
    - **[Models/Lora/Plugin/VAE](#modelslorapluginvae)**
    - **[图片放大](#controlnet图片放大)**
    - **[采样器&噪声](#采样器噪声)**
    - **[ControlNet](#controlnet)**

***

## 模型&数据集下载

### 资源站

- [Hugging face (需要科学上网)](https://huggingface.co)

- [Hugging face 镜像站](https://hf-mirror.com)

### 下载

- [HF下载全方法汇总](https://zhuanlan.zhihu.com/p/663712983)

- 用git的lfs 直接克隆仓库很蠢：断了只能重新下，而且会下载历史版本（占空间巨大

- #### huggingface-cli+hf_transfer
  
    默认从hf本站下载，要用镜像站的话加上：
    
    ```shell
    export HF_ENDPOINT=https://hf-mirror.com  # 仅在当前shell生效
    ```
    
    ##### cli
    
    ```shell
    pip3 install -U huggingface_hub  # 安装依赖，并且要求Python>=3.8
    ```

    ```shell
    huggingface-cli download <库名称> --local-dir <本地文件夹名称>  # 下载模型
    ```

    ```shell
    huggingface-cli download bigscience/bloom-560m --local-dir bloom-560m  # 下载bloom-560m库
    ```

    ##### hf_transfer (网不稳定不要用)
    
    ```shell
    pip3 install -U hf-transfer  # 安装依赖
    ```
    
    ```shell
    export HF_HUB_ENABLE_HF_TRANSFER=1  # 仅在当前shell窗口生效
    ```
    
    ```shell
    huggingface-cli download --resume-download <库> --local-dir <本地>  # 方法同cli
    ```
    

***

## Ollama

### 常见指令

```shell
ollama list  # 查看所有模型
```

```shell
ollama remove <your-model-name>  # 删除模型
```

```shell
ollama run <your-model-name>  # (下载并)运行模型
```

### 添加新模型
- ollama指令可以直接下载的模型：https://ollama.com/library

    ```shell
    ollama run <model-name>  # 可以直接下载并运行
    ```

- 添加下载好的gguf模型：

    1. 创建名为Modelfile的文件，在里面写入（按实际路径改）：
        ```shell
        FROM ./your-model-path.gguf
        ```
    2. 读取模型
        ```shell
        ollama create <your-model-name> -f Modelfile
        ```
   3. 运行模型
      ```shell
      ollama run <your-model-name>
      ```

***

## Open-webui
### 向docker添加Webui容器

```shell
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main  # 只有首次运行，在docker中添加服务
```

### 启动&关闭Webui容器
```shell
docker run open-webui
docker stop open-webui
# 启动后在localhost:3000就可以使用webui了
```
### 在Open-webui中使用Stable-diffusion

- 添加stable diffusion webui的api：  open-webui -> 设置 -> 图像 -> URL

```shell
./webui.sh --api  # 本地部署stable diffusion (加上--nowebui不启动webui, 注意端口会变)
```
- webui服务使用的端口是7860
- 注意：因为open-webui是在docker部署的，sd是在宿主机部署，open-webui调用sd接口相当于docker内部调用宿主机服务，所以在浏览器虽然用127.0.0.1:7860能访问sd，但是这里填写127.0.0.1:7860是无法请求的，
- **要想在open-webui里使用sd的webui接口，要使用http://host.docker.internal:7860**
- 需要让LLM模型把你想使用的提示词回复给你，然后点击生成图片按钮，就可以生成了。

***

## Stable-diffusion
### 从[AUTOMATIC1111](https://github.com/AUTOMATIC1111)获取更新

```shell
git remote add upstream git@github.com:AUTOMATIC1111/stable-diffusion-webui.git # 添加远程上游，指向原始库
git fetch upstream  # 从上游仓库获取最新的更改
git merge upstream/merge_branch # 合并上游的更改

# 本地分支:
merge_branch 合并远程用的
master 本地主分支
```

### Models/Lora/Plugin/VAE
-  **[Civitai](https://civitai.com) 下载**
- add_detail  -- **lora**
- easynegative  -- **lora **
- age_slider  -- **lora **
- counterfeit  -- **sd **
- abyssorangemix3AOM3a1b -- **sd **
- cetusMix -- **sd**
- meinamix_v10 -- **sd**
- clearVAE -- **VAE 细节+色彩**
- k2-f8-anime2 -- **VAE 色彩**
- [sd-webui-prompt-all-in-one](https://github.com/Physton/sd-webui-prompt-all-in-one)-- **提示词管理**
- [stable-diffusion-webui-localization-zh_CN](https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN)  -- **简中语言包**

### ControlNet图片放大
- [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) -- 插件
- [ultimate-upscale-for-automatic1111](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111) -- 插件
- [4x-UltraSharp](https://mega.nz/folder/qZRBmaIY#nIG8KyWFcGNTuMX_XNbJ_g/file/vRYVhaDA) -- 放到models/ESRGAN/里
- [Controlnet模型](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11f1e_sd15_tile.pth) -- 放到models/ControlNet/里： control_v11f1e_sd15_tile.pth

1. **移除正向提示词（是局部放大在拼接，ai会搞混）**
2. **保留负面提示词**
3. **重绘幅度(Denoising)：0.2 ~ 0.3**  
4. **Control Net**
    1. **预处理器(Preprocessor): tile_resample分块控制**
    2. **模型：control_v11f1e_sd15_tile [a371b31b]**
5. **脚本script：Ultimate SD upscale**
    1. **Target size type：Scale from image size -- 放大倍数(长宽各放大n倍)**
6. **放大效果不理想可以考虑分多几次放大**

### 采样器&噪声

- **名字中带有a，及SDE的为祖先采样器**
	不收敛，重复率低
- **Euler、Euler a** 
	快速获得简单的结果
- **DPM++ 2M Karras**
	推荐的算法，速度快，质量好，推荐步数 **20~30** 步
- **DPM++ SDE Karras**
	图像质量好但是不收敛，速度慢，推荐步数 **10~15** 步
- **DPM++ 2M SDE Karras**
	2M和SDE的结合算法，速度和2M相仿，推荐步数 **20~30** 步
- **DPM++ 2M SDE Exponential***
	画面柔和，细节更少一些，推荐步数 **20~30** 步

### ControlNet

1. **[图片放大](#controlnet图片放大)**

    

2. **深度(Depth)**

    深度图中距离镜头越近的物品越亮，相似距离基本无法区分 -- 非常有效的替换/提取背景，提取人物动作等

    预处理器（用来生成深度图）：

    - **depth_leres & depth_leres++**
      整个图片细节更丰富，包括背景。++版可以设置remove near/remove background % [(都报错)](https://github.com/Mikubill/sd-webui-controlnet/issues/1649)
      用

      ```shell
      All_my_files/__init__.py
      ```
      替换
      ```shell
      extensions/sd-webui-controlnet/annotator/leres/__init__.py
      ```
      可以解决。报错原因是 新版Pythorch 用了Metal Performance Shaders (MPS) 后端，而以前的程序一部分用了cpu，一部分用了mps，在init程序里统一了就行

    - **depth_midas & depth_zoe**
      细节较少，背景基本忽略，但是黑白对比更丰富 (zoe报错)

    - **depth_anything**
      最新模型，目标是建立一个简单而强大的基础模型，在任何情况下处理任何图像近距离采集不如leres精细
      模型（必须）：control_v11f1p_sd15_depth.pth，同名yaml文件


3. **姿态(OpenPose)**

      专门用来提取/使用人物动作，捕捉人物骨骼，以及表情。非常有效的解决AI画不好手的问题，但是也挺麻烦

      editor插件能编辑骨骼，识别的也还行，但是好麻烦。

