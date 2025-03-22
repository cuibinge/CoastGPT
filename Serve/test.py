# 这段代码的主要功能是向指定的工作器发送请求，获取模型生成的回复。它可以通过命令行参数指定控制器地址、工作器地址、模型名称、最大生成令牌数和用户消息。
# 如果未指定工作器地址，代码会从控制器获取可用模型列表，并根据指定的模型名称获取对应的工作器地址。最后，代码会向工作器发送请求，并流式打印模型生成的回复。
import argparse
import json
import requests
from llava.conversation import default_conversation

def main():
    # 检查命令行参数中是否指定了工作器地址
    if args.worker_address:
        # 如果指定了工作器地址，则直接使用该地址
        worker_addr = args.worker_address
    else:
        # 如果未指定工作器地址，则从控制器获取工作器地址
        controller_addr = args.controller_address
        # 向控制器发送请求，刷新所有工作器的状态
        ret = requests.post(controller_addr + "/refresh_all_workers")
        # 向控制器发送请求，获取可用模型列表
        ret = requests.post(controller_addr + "/list_models")
        # 从响应中提取模型列表
        models = ret.json()["models"]
        # 对模型列表进行排序
        models.sort()
        # 打印可用模型列表
        print(f"Models: {models}")

        # 向控制器发送请求，获取指定模型的工作器地址
        ret = requests.post(controller_addr + "/get_worker_address",
            json={"model": args.model_name})
        # 从响应中提取工作器地址
        worker_addr = ret.json()["address"]
        # 打印工作器地址
        print(f"worker_addr: {worker_addr}")

    # 如果工作器地址为空，则直接返回
    if worker_addr == "":
        return

    # 复制默认的对话模板
    conv = default_conversation.copy()
    # 在对话中添加用户的消息
    conv.append_message(conv.roles[0], args.message)
    # 根据对话内容生成提示信息
    prompt = conv.get_prompt()

    # 设置请求头，指定用户代理
    headers = {"User-Agent": "LLaVA Client"}
    # 构建请求负载，包含模型名称、提示信息、最大生成令牌数、温度和停止标记
    pload = {
        "model": args.model_name,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "stop": conv.sep,
    }
    # 向工作器发送 POST 请求，请求流式生成回复
    response = requests.post(worker_addr + "/worker_generate_stream", headers=headers,
            json=pload, stream=True)

    # 打印提示信息，将分隔符替换为换行符
    print(prompt.replace(conv.sep, "\n"), end="")
    # 遍历响应的每一行数据
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            # 解码并解析 JSON 数据
            data = json.loads(chunk.decode("utf-8"))
            # 提取回复文本，去除分隔符
            output = data["text"].split(conv.sep)[-1]
            # 打印回复文本
            print(output, end="\r")
    # 打印换行符
    print("")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加控制器地址参数，默认值为 http://localhost:21001
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    # 添加工作器地址参数
    parser.add_argument("--worker-address", type=str)
    # 添加模型名称参数，默认值为 facebook/opt-350m
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # 添加最大生成令牌数参数，默认值为 32
    parser.add_argument("--max-new-tokens", type=int, default=32)
    # 添加用户消息参数，默认值为一个故事请求
    parser.add_argument("--message", type=str, default=
        "Tell me a story with more than 1000 words.")
    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main()