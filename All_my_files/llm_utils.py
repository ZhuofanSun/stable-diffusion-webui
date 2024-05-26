import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI


# 参考：
# https://github.com/QwenLM/Qwen1.5/blob/main/README.md
# https://github.com/QwenLM/Qwen/blob/main/README_CN.md

def run_locally():
    """
    本地运行，需要提前下载模型
    :return: None
    """
    model_dir = "qwen1.5-7b"

    device = "mps"  # the device to load the model onto
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    prompt = "说：这是一个测试。除了这句话什么都不要输出"
    messages = [
        {"role": "system", "content": "你是一个乐于帮助，并能严格按照要求回答的助手"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("回答： ")
    print(response)


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


# runwith_ollama_api()
run_locally()
