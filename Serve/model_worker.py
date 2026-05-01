"""
model_worker.py 文件在项目里扮演着模型执行器的角色，它的主要功能是加载模型，处理来自客户端的请求并生成回复，同时与控制器进行通信以维护自身状态。
"""
import argparse
import asyncio
import json
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from llava.constants import WORKER_HEART_BEAT_INTERVAL
from llava.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread

# 定义1GB的字节数
GB = 1 << 30

# 生成一个唯一的工作者ID
worker_id = str(uuid.uuid4())[:6]
# 构建一个日志记录器，用于记录模型工作者的相关信息
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
# 全局计数器，用于统计请求数量
global_counter = 0

# 模型信号量，用于控制并发请求数量
model_semaphore = None

# 心跳工作线程函数，用于定期向控制器发送心跳信息
def heart_beat_worker(controller):
    while True:
        # 等待指定的心跳间隔时间
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        # 调用控制器的发送心跳信息方法
        controller.send_heart_beat()

# 模型工作者类，负责加载模型、处理请求和与控制器通信
class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device):
        # 控制器的地址
        self.controller_addr = controller_addr
        # 工作者的地址
        self.worker_addr = worker_addr
        # 工作者的ID
        self.worker_id = worker_id
        # 去除模型路径末尾的斜杠
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        # 如果没有指定模型名称，则根据模型路径生成
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        # 设备类型，如'cuda'或'cpu'
        self.device = device
        # 记录加载模型的日志信息
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        # 加载预训练模型、分词器、图像处理器和上下文长度
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)
        # 判断模型是否为多模态模型
        self.is_multimodal = 'llava' in self.model_name.lower()

        # 如果不禁止注册，则向控制器注册并启动心跳线程
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    # 向控制器注册工作者
    def register_to_controller(self):
        logger.info("Register to controller")
        # 构建注册请求的URL
        url = self.controller_addr + "/register_worker"
        # 构建注册请求的数据
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        # 发送POST请求进行注册
        r = requests.post(url, json=data)
        # 确保注册请求成功
        assert r.status_code == 200

    # 向控制器发送心跳信息
    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")
        # 构建心跳请求的URL
        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                # 发送POST请求发送心跳信息
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                # 获取控制器返回的是否存在的信息
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                # 记录心跳错误信息
                logger.error(f"heart beat error: {e}")
            # 等待5秒后重试
            time.sleep(5)

        # 如果工作者在控制器中不存在，则重新注册
        if not exist:
            self.register_to_controller()

    # 获取请求队列的长度
    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    # 获取工作者的状态信息
    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    # 流式生成文本的方法
    @torch.inference_mode()
    def generate_stream(self, params):
        # 获取分词器、模型和图像处理器
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        # 获取请求的提示信息
        prompt = params["prompt"]
        # 保存原始提示信息
        ori_prompt = prompt
        # 获取请求中的图像信息
        images = params.get("images", None)
        # 初始化图像令牌数量
        num_image_tokens = 0
        # 如果是多模态模型且有图像信息
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                # 检查图像数量是否与提示信息中的<image>令牌数量匹配
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                # 从Base64编码加载图像
                images = [load_image_from_base64(image) for image in images]
                # 处理图像
                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    # 将图像移动到模型设备上
                    images = [image.to(self.model.device, dtype=torch.float16) for image in images]
                else:
                    images = images.to(self.model.device, dtype=torch.float16)

                # 替换提示信息中的<image>令牌
                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                # 计算图像令牌数量
                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
            # 构建图像参数
            image_args = {"images": images}
        else:
            images = None
            image_args = {}

        # 获取温度参数
        temperature = float(params.get("temperature", 1.0))
        # 获取Top-p参数
        top_p = float(params.get("top_p", 1.0))
        # 获取模型的最大上下文长度
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        # 获取最大生成的新令牌数量
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        # 获取停止字符串
        stop_str = params.get("stop", None)
        # 判断是否进行采样
        do_sample = True if temperature > 0.001 else False

        # 将提示信息转换为输入ID
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        # 构建停止条件
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # 构建流式生成器
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        # 计算最大生成的新令牌数量
        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        # 如果最大生成的新令牌数量小于1，则返回错误信息
        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        # 启动一个线程进行模型生成
        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()

        # 初始化生成的文本
        generated_text = ori_prompt
        # 流式生成文本
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            # 生成流式响应
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    # 流式生成文本的包装方法，处理异常情况
    def generate_stream_gate(self, params):
        try:
            # 调用generate_stream方法进行流式生成
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            # 处理值错误异常
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            # 处理CUDA错误异常
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            # 处理其他未知异常
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"

# 创建FastAPI应用实例
app = FastAPI()

# 释放模型信号量的方法
def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()

# 处理流式生成请求的API端点
@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    # 增加全局计数器
    global_counter += 1
    # 获取请求的JSON数据
    params = await request.json()

    if model_semaphore is None:
        # 初始化模型信号量
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    # 获取模型信号量
    await model_semaphore.acquire()
    # 发送心跳信息
    worker.send_heart_beat()
    # 调用工作者的流式生成方法
    generator = worker.generate_stream_gate(params)
    # 创建后台任务
    background_tasks = BackgroundTasks()
    # 添加释放模型信号量的后台任务
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    # 返回流式响应
    return StreamingResponse(generator, background=background_tasks)

# 处理获取工作者状态请求的API端点
@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加主机地址参数
    parser.add_argument("--host", type=str, default="localhost")
    # 添加端口号参数
    parser.add_argument("--port", type=int, default=21002)
    # 添加工作者地址参数
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    # 添加控制器地址参数
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    # 添加模型路径参数
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # 添加模型基础路径参数
    parser.add_argument("--model-base", type=str, default=None)
    # 添加模型名称参数
    parser.add_argument("--model-name", type=str)
    # 添加设备类型参数
    parser.add_argument("--device", type=str, default="cuda")
    # 添加多模态模式参数
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    # 添加限制模型并发请求数量参数
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    # 添加流式间隔参数
    parser.add_argument("--stream-interval", type=int, default=1)
    # 添加禁止注册参数
    parser.add_argument("--no-register", action="store_true")
    # 添加加载8位模型参数
    parser.add_argument("--load-8bit", action="store_true")
    # 添加加载4位模型参数
    parser.add_argument("--load-4bit", action="store_true")
    # 解析命令行参数
    args = parser.parse_args()
    # 记录命令行参数信息
    logger.info(f"args: {args}")

    if args.multi_modal:
        # 记录多模态模式的警告信息
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    # 创建模型工作者实例
    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device)
    # 启动FastAPI应用
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")