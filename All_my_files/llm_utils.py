import os
import re
import gc
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import torch


# 参考：
# https://github.com/QwenLM/Qwen1.5/blob/main/README.md
# https://github.com/QwenLM/Qwen/blob/main/README_CN.md
# 正向提示词
# 正向提示词

def read_init_message(path):
    # 读取文件的所有内容并返回
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_model_and_tokenizer(model_dir, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        offload_folder="save_folder",
        low_cpu_mem_usage=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    time.sleep(1)
    return model, tokenizer


def generate_response(model, tokenizer, messages, device, max_new_tokens=512):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def read_init_message(file_path):
    with open(file_path, 'r') as file:
        return file.read()


act_model, act_tokenizer = load_model_and_tokenizer("qwen1.5-7b", "mps")


def run_locally():
    global act_model, act_tokenizer
    model_dir = "qwen1.5-7b"
    device = "mps"

    try:

        init_message = read_init_message(f"{model_dir}/init.txt")

        prompt = input("\n想画一幅什么样的图片？\n")
        if prompt == '':
            prompt = "夜晚的一片花田，有漂亮的夜空，远处有繁华的城市，整个画面的色调是蓝色"
        print("\n正在生成回答...\n")

        messages = [
            {"role": "system", "content": init_message},
            {"role": "user", "content": prompt}
        ]
        response = generate_response(act_model, act_tokenizer, messages, device)
        print(response)

        while True:
            feed_back = input("\n要改动吗？(yes/y or no/n)\n")
            if feed_back.lower() in ["yes", "y"]:
                change_prompt = input("\n想改动什么\n")
                messages = [
                    {"role": "system", "content": response},
                    {"role": "user", "content": change_prompt}
                ]
                response = generate_response(act_model, act_tokenizer, messages, device)
                print(response)
            else:
                break

    except Exception as e:
        print(f"出现错误: {e}")

    finally:
        del act_model
        del act_tokenizer

        gc.collect()
        torch.cuda.empty_cache()
        torch.mps.empty_cache()
    return response


def runwith_ollama_api():
    """
    测试
    通过 ollama api 运行，需要提前运行ollama serve
    并且保证ollama run qwen:7b可以运行
    :return: None
    """
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',  # required but ignored
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': '说：这是一个测试。除了这句话什么都不要输出',
            }
        ],
        model='qwen:7b',
    )
    print(chat_completion)


def extract_info(text):
    # 使用正则表达式提取prompt, negative prompt和people
    # 使用正则表达式提取 prompt, negative prompt 和 people
    prompt_match = re.search(r'(?i)\*\*(?:Prompt|Positive Prompt):\*\*\s*(.*?)(?=\*\*Negative Prompt:|$)', text,
                             re.DOTALL)
    negative_prompt_match = re.search(r'(?i)\*\*Negative Prompt:\*\*\s*(.*?)(?=\*\*People:|$)', text, re.DOTALL)
    people_match = re.search(r'(?i)\*\*People:\*\*\s*(True|False)', text, re.IGNORECASE)

    # 提取匹配结果
    prompt = prompt_match.group(1).strip().split(',') if prompt_match else []
    negative_prompt = negative_prompt_match.group(1).strip().split(',') if negative_prompt_match else []
    people = people_match.group(1).strip().lower() == 'true' if people_match else False

    # 去除多余的空格
    prompt = [p.strip() for p in prompt]
    negative_prompt = [np.strip() for np in negative_prompt]

    return prompt, negative_prompt, people


def ask_to_fill_data(data: dict):
    prompt_lst, negative_prompt_lst, people = extract_info(run_locally())
    keys = data.keys()
    print('Please fill the following data:')
    for key in keys:
        if key == 'prompt':
            # join prompt_lst with ','
            prompt = ', '.join(prompt_lst)
            data[key] = prompt
        elif key == 'negative_prompt':
            # join negative_prompt_lst with ','
            negative_prompt = ', '.join(negative_prompt_lst)
            data[key] = negative_prompt
        elif key == 'alwayson_scripts':
            continue
        elif key == 'sampler_index':
            sampler_option = ['DPM++ 2M', 'Euler a']  # 可以往里面添加采样器种类
            print("Choose from following:")
            for i, option in enumerate(sampler_option):
                print(f'{i}: {option}')
            sampler_index = input(f'{key}: ')
            if isinstance(sampler_index, int) and (sampler_index == 1 or sampler_index == 0):
                data[key] = sampler_option[sampler_index]
            else:
                data[key] = sampler_option[0]
        elif key == 'scheduler':
            continue
        else:
            input_value = input(f'{key}: ')
            if input_value == '':
                continue
            else:
                data[key] = input_value
    return people


if __name__ == "__main__":
    # runwith_ollama_api()
    import datafile as data

    template_data = data.get_template_data()
    print(ask_to_fill_data(template_data))
    print(template_data)
    input("Press Enter to continue...")
