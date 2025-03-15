import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib

# 构建一个日志记录器，用于记录与 Gradio 网页服务器相关的信息，日志文件名为 gradio_web_server.log
logger = build_logger("gradio_web_server", "gradio_web_server.log")

# 设置请求头，表明这是一个 LLaVA 客户端发起的请求
headers = {"User-Agent": "LLaVA Client"}

# 定义按钮更新的默认状态
no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

# 定义模型优先级字典，为不同的模型分配优先级字符串
priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

# 获取对话日志文件名的函数
def get_conv_log_filename():
    """
    该函数用于获取对话日志的文件名。目前函数体为空，需要根据具体需求实现。
    """
    pass

# 获取可用模型列表的函数
def get_model_list():
    """
    该函数用于获取可用的模型列表。目前函数体为空，需要根据具体需求实现。
    """
    pass

# 获取浏览器窗口 URL 参数的 JavaScript 代码
get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""

# 加载演示界面的函数
def load_demo(url_params, request: gr.Request):
    """
    该函数用于加载演示界面。
    :param url_params: 从浏览器窗口 URL 中获取的参数
    :param request: Gradio 请求对象
    """
    pass

# 刷新模型列表并加载演示界面的函数
def load_demo_refresh_model_list(request: gr.Request):
    """
    该函数用于刷新可用模型列表并加载演示界面。
    :param request: Gradio 请求对象
    """
    pass

# 对最后一次响应进行投票的函数
def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    """
    该函数用于对最后一次响应进行投票（如点赞、踩、标记等）。
    :param state: 当前对话状态
    :param vote_type: 投票类型（如 upvote、downvote、flag 等）
    :param model_selector: 模型选择器的值
    :param request: Gradio 请求对象
    """
    pass

# 对最后一次响应进行点赞的函数
def upvote_last_response(state, model_selector, request: gr.Request):
    """
    该函数用于对最后一次响应进行点赞。
    :param state: 当前对话状态
    :param model_selector: 模型选择器的值
    :param request: Gradio 请求对象
    """
    pass

# 对最后一次响应进行踩的函数
def downvote_last_response(state, model_selector, request: gr.Request):
    """
    该函数用于对最后一次响应进行踩。
    :param state: 当前对话状态
    :param model_selector: 模型选择器的值
    :param request: Gradio 请求对象
    """
    pass

# 对最后一次响应进行标记的函数
def flag_last_response(state, model_selector, request: gr.Request):
    """
    该函数用于对最后一次响应进行标记，通常用于标记不适当的内容。
    :param state: 当前对话状态
    :param model_selector: 模型选择器的值
    :param request: Gradio 请求对象
    """
    pass

# 重新生成最后一次响应的函数
def regenerate(state, image_process_mode, request: gr.Request):
    """
    该函数用于重新生成最后一次响应。
    :param state: 当前对话状态
    :param image_process_mode: 图像处理模式
    :param request: Gradio 请求对象
    """
    pass

# 清除对话历史的函数
def clear_history(request: gr.Request):
    """
    该函数用于清除当前的对话历史。
    :param request: Gradio 请求对象
    """
    pass

# 添加文本和图像到对话的函数
def add_text(state, text, image, image_process_mode, request: gr.Request):
    """
    该函数用于将用户输入的文本和图像添加到对话中。
    :param state: 当前对话状态
    :param text: 用户输入的文本
    :param image: 用户上传的图像
    :param image_process_mode: 图像处理模式
    :param request: Gradio 请求对象
    """
    pass

# 通过 HTTP 请求获取模型响应的函数
def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    """
    该函数通过 HTTP 请求向模型服务器发送请求，并获取模型的响应。
    :param state: 当前对话状态
    :param model_selector: 模型选择器的值
    :param temperature: 生成文本时的温度参数，控制随机性
    :param top_p: 生成文本时的核采样参数
    :param max_new_tokens: 生成的最大新令牌数
    :param request: Gradio 请求对象
    """
    pass

# 定义标题的 Markdown 文本
title_markdown = ("""
# 🛰️ RemoteChat: Advanced Remote Sensing and Spatial Intelligence Model
[[Project Page]()] [[Code]()] [[Model]()] | 📚 [[]()] [[GeoChat-v1]()]
""")

# 定义使用条款的 Markdown 文本
tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")

# 定义学习更多信息的 Markdown 文本
learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

# 定义 CSS 样式，用于设置按钮的最小宽度
block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

# 构建 Gradio 演示界面的函数
def build_demo(embed_mode):
    """
    该函数用于构建 Gradio 演示界面。
    :param embed_mode: 是否以嵌入模式运行
    """
    pass

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加服务器主机地址参数，默认为 0.0.0.0
    parser.add_argument("--host", type=str, default="0.0.0.0")
    # 添加服务器端口参数
    parser.add_argument("--port", type=int)
    # 添加控制器 URL 参数，默认为 http://localhost:21001
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    # 添加并发请求数量参数，默认为 10
    parser.add_argument("--concurrency-count", type=int, default=10)
    # 添加模型列表加载模式参数，可选值为 once 或 reload，默认为 once
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    # 添加是否共享演示界面参数
    parser.add_argument("--share", action="store_true")
    # 添加是否进行内容审核参数
    parser.add_argument("--moderate", action="store_true")
    # 添加是否以嵌入模式运行参数
    parser.add_argument("--embed", action="store_true")
    # 解析命令行参数
    args = parser.parse_args()
    # 记录命令行参数信息到日志
    logger.info(f"args: {args}")

    # 获取可用模型列表
    models = get_model_list()

    # 记录命令行参数信息到日志
    logger.info(args)
    # 构建 Gradio 演示界面
    demo = build_demo(args.embed)
    # 设置演示界面的队列并发数量，并关闭 API 开放
    demo.queue(
        concurrency_count=args.concurrency_count,
        api_open=False
    ).launch(
        # 设置服务器主机地址
        server_name=args.host,
        # 设置服务器端口
        server_port=args.port,
        # 是否共享演示界面
        share=True
    )