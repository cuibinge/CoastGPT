"""
A controller manages distributed workers.
It sends worker addresses to clients.
"""
import argparse
import asyncio
import dataclasses
from enum import Enum, auto
import json
import logging
import time
from typing import List, Union
import threading

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from llava.constants import CONTROLLER_HEART_BEAT_EXPIRATION
from llava.utils import build_logger, server_error_msg

# 创建一个名为 "controller" 的日志记录器，将日志信息写入 "controller.log" 文件
logger = build_logger("controller", "controller.log")

# 定义调度方法的枚举类
class DispatchMethod(Enum):
    # 随机选择工作节点的调度方法
    LOTTERY = auto()
    # 选择队列最短的工作节点的调度方法
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        """
        根据字符串名称返回对应的调度方法枚举值
        :param name: 调度方法的字符串名称
        :return: 对应的调度方法枚举值
        """
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")

# 定义工作节点信息的数据类
@dataclasses.dataclass
class WorkerInfo:
    # 工作节点支持的模型名称列表
    model_names: List[str]
    # 工作节点的处理速度
    speed: int
    # 工作节点的队列长度
    queue_length: int
    # 是否检查工作节点的心跳
    check_heart_beat: bool
    # 工作节点最后一次心跳的时间
    last_heart_beat: str

def heart_beat_controller(controller):
    """
    心跳检查线程的目标函数，定期移除长时间没有心跳的工作节点
    :param controller: 控制器实例
    """
    while True:
        # 每隔 CONTROLLER_HEART_BEAT_EXPIRATION 秒执行一次心跳检查
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stable_workers_by_expiration()

# 定义控制器类
class Controller:
    def __init__(self, dispatch_method: str):
        """
        初始化控制器
        :param dispatch_method: 调度方法的字符串名称
        """
        # 存储工作节点信息的字典，键为工作节点名称，值为 WorkerInfo 实例
        self.worker_info = {}
        # 根据字符串名称获取对应的调度方法枚举值
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        # 创建一个心跳检查线程，目标函数为 heart_beat_controller
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,))
        # 启动心跳检查线程
        self.heart_beat_thread.start()

        # 记录控制器初始化信息
        logger.info("Init controller")

    def register_worker(self, worker_name: str, check_heart_beat: bool,
                        worker_status: dict):
        """
        注册一个工作节点
        :param worker_name: 工作节点的名称
        :param check_heart_beat: 是否检查工作节点的心跳
        :param worker_status: 工作节点的状态信息
        :return: 注册成功返回 True，失败返回 False
        """
        if worker_name not in self.worker_info:
            # 记录注册新工作节点的信息
            logger.info(f"Register a new worker: {worker_name}")
        else:
            # 记录更新现有工作节点信息的信息
            logger.info(f"Register an existing worker: {worker_name}")

        # 如果没有提供工作节点的状态信息，则尝试获取
        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        # 如果获取工作节点状态信息失败，则返回 False
        if not worker_status:
            return False

        # 将工作节点信息存储到 worker_info 字典中
        self.worker_info[worker_name] = WorkerInfo(
            worker_status["model_names"], worker_status["speed"], worker_status["queue_length"],
            check_heart_beat, time.time())

        # 记录工作节点注册完成的信息
        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True

    def get_worker_status(self, worker_name: str):
        """
        获取工作节点的状态信息
        :param worker_name: 工作节点的名称
        :return: 工作节点的状态信息，如果获取失败则返回 None
        """
        try:
            # 向工作节点发送 POST 请求，获取其状态信息
            r = requests.post(worker_name + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            # 记录获取工作节点状态信息失败的错误信息
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None

        # 如果请求返回的状态码不是 200，则记录错误信息并返回 None
        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None

        # 返回工作节点的状态信息
        return r.json()

    def remove_worker(self, worker_name: str):
        """
        移除一个工作节点
        :param worker_name: 工作节点的名称
        """
        # 从 worker_info 字典中删除指定的工作节点信息
        del self.worker_info[worker_name]

    def refresh_all_workers(self):
        """
        刷新所有工作节点的信息，移除长时间没有响应的工作节点
        """
        # 备份旧的工作节点信息
        old_info = dict(self.worker_info)
        # 清空工作节点信息字典
        self.worker_info = {}

        # 遍历旧的工作节点信息，尝试重新注册每个工作节点
        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                # 记录移除长时间没有响应的工作节点的信息
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        """
        列出所有工作节点支持的模型名称
        :return: 所有工作节点支持的模型名称列表
        """
        # 用于存储所有模型名称的集合
        model_names = set()

        # 遍历所有工作节点的信息，将每个工作节点支持的模型名称添加到集合中
        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)

        # 将集合转换为列表并返回
        return list(model_names)

    def get_worker_address(self, model_name: str):
        """
        根据模型名称获取可用的工作节点地址
        :param model_name: 模型名称
        :return: 可用的工作节点地址，如果没有可用节点则返回空字符串
        """
        if self.dispatch_method == DispatchMethod.LOTTERY:
            # 存储支持指定模型的工作节点名称列表
            worker_names = []
            # 存储支持指定模型的工作节点处理速度列表
            worker_speeds = []
            # 遍历所有工作节点的信息，找出支持指定模型的工作节点
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.speed)
            # 将工作节点处理速度列表转换为 numpy 数组
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            # 计算工作节点处理速度的总和
            norm = np.sum(worker_speeds)
            # 如果总和小于一个阈值，则返回空字符串
            if norm < 1e-4:
                return ""
            # 对工作节点处理速度进行归一化
            worker_speeds = worker_speeds / norm
            if True:  # Directly return address
                # 根据归一化后的处理速度随机选择一个工作节点
                pt = np.random.choice(np.arange(len(worker_names)),
                    p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # Check status before returning
            while True:
                # 根据归一化后的处理速度随机选择一个工作节点
                pt = np.random.choice(np.arange(len(worker_names)),
                    p=worker_speeds)
                worker_name = worker_names[pt]

                # 检查工作节点的状态
                if self.get_worker_status(worker_name):
                    break
                else:
                    # 如果工作节点状态异常，则移除该节点
                    self.remove_worker(worker_name)
                    worker_speeds[pt] = 0
                    # 重新计算工作节点处理速度的总和
                    norm = np.sum(worker_speeds)
                    if norm < 1e-4:
                        return ""
                    # 重新对工作节点处理速度进行归一化
                    worker_speeds = worker_speeds / norm
                    continue
            return worker_name
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            # 存储支持指定模型的工作节点名称列表
            worker_names = []
            # 存储支持指定模型的工作节点队列长度与处理速度的比值列表
            worker_qlen = []
            # 遍历所有工作节点的信息，找出支持指定模型的工作节点
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_qlen.append(w_info.queue_length / w_info.speed)
            # 如果没有支持指定模型的工作节点，则返回空字符串
            if len(worker_names) == 0:
                return ""
            # 找出队列长度与处理速度比值最小的工作节点的索引
            min_index = np.argmin(worker_qlen)
            # 获取队列长度与处理速度比值最小的工作节点的名称
            w_name = worker_names[min_index]
            # 将该工作节点的队列长度加 1
            self.worker_info[w_name].queue_length += 1
            # 记录选择的工作节点信息
            logger.info(f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}")
            return w_name
        else:
            # 如果调度方法无效，则抛出异常
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    def receive_heart_beat(self, worker_name: str, queue_length: int):
        """
        接收工作节点的心跳信息
        :param worker_name: 工作节点的名称
        :param queue_length: 工作节点的队列长度
        :return: 如果工作节点存在则返回 True，否则返回 False
        """
        if worker_name not in self.worker_info:
            # 记录接收到未知工作节点心跳信息的信息
            logger.info(f"Receive unknown heart beat. {worker_name}")
            return False

        # 更新工作节点的队列长度
        self.worker_info[worker_name].queue_length = queue_length
        # 更新工作节点的最后一次心跳时间
        self.worker_info[worker_name].last_heart_beat = time.time()
        # 记录接收到工作节点心跳信息的信息
        logger.info(f"Receive heart beat. {worker_name}")
        return True

    def remove_stable_workers_by_expiration(self):
        """
        移除长时间没有心跳的工作节点
        """
        # 计算过期时间
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        # 存储需要移除的工作节点名称列表
        to_delete = []
        # 遍历所有工作节点的信息，找出长时间没有心跳的工作节点
        for worker_name, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                to_delete.append(worker_name)

        # 移除长时间没有心跳的工作节点
        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def worker_api_generate_stream(self, params):
        """
        处理客户端的流式生成请求
        :param params: 请求参数
        :return: 流式响应生成器
        """
        # 根据请求中的模型名称获取可用的工作节点地址
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            # 记录没有可用工作节点的信息
            logger.info(f"no worker: {params['model']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            # 生成错误响应
            yield json.dumps(ret).encode() + b"\0"

        try:
            # 向工作节点发送 POST 请求，进行流式生成
            response = requests.post(worker_addr + "/worker_generate_stream",
                json=params, stream=True, timeout=5)
            # 遍历响应的每一行数据
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    # 生成流式响应数据
                    yield chunk + b"\0"
        except requests.exceptions.RequestException as e:
            # 记录工作节点超时的信息
            logger.info(f"worker timeout: {worker_addr}")
            ret = {
                "text": server_error_msg,
                "error_code": 3,
            }
            # 生成错误响应
            yield json.dumps(ret).encode() + b"\0"

    # Let the controller act as a worker to achieve hierarchical
    # management. This can be used to connect isolated sub networks.
    def worker_api_get_status(self):
        """
        获取所有工作节点的状态信息
        :return: 所有工作节点的状态信息字典
        """
        # 用于存储所有工作节点支持的模型名称的集合
        model_names = set()
        # 所有工作节点的处理速度总和
        speed = 0
        # 所有工作节点的队列长度总和
        queue_length = 0

        # 遍历所有工作节点的名称
        for w_name in self.worker_info:
            # 获取工作节点的状态信息
            worker_status = self.get_worker_status(w_name)
            if worker_status is not None:
                # 将工作节点支持的模型名称添加到集合中
                model_names.update(worker_status["model_names"])
                # 累加工作节点的处理速度
                speed += worker_status["speed"]
                # 累加工作节点的队列长度
                queue_length += worker_status["queue_length"]

        # 返回所有工作节点的状态信息字典
        return {
            "model_names": list(model_names),
            "speed": speed,
            "queue_length": queue_length,
        }

# 创建 FastAPI 应用实例
app = FastAPI()

# 定义注册工作节点的 API 端点
@app.post("/register_worker")
async def register_worker(request: Request):
    """
    处理工作节点注册请求
    :param request: 请求对象
    :return: 无
    """
    # 获取请求中的 JSON 数据
    data = await request.json()
    # 调用控制器的 register_worker 方法进行工作节点注册
    controller.register_worker(
        data["worker_name"], data["check_heart_beat"],
        data.get("worker_status", None))

# 定义刷新所有工作节点信息的 API 端点
@app.post("/refresh_all_workers")
async def refresh_all_workers():
    """
    处理刷新所有工作节点信息的请求
    :return: 无
    """
    # 调用控制器的 refresh_all_workers 方法刷新所有工作节点信息
    models = controller.refresh_all_workers()

# 定义列出所有模型名称的 API 端点
@app.post("/list_models")
async def list_models():
    """
    处理列出所有模型名称的请求
    :return: 包含所有模型名称的字典
    """
    # 调用控制器的 list_models 方法列出所有模型名称
    models = controller.list_models()
    return {"models": models}

# 定义获取工作节点地址的 API 端点
@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    """
    处理获取工作节点地址的请求
    :param request: 请求对象
    :return: 包含工作节点地址的字典
    """
    # 获取请求中的 JSON 数据
    data = await request.json()
    # 调用控制器的 get_worker_address 方法获取工作节点地址
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}

# 定义接收工作节点心跳信息的 API 端点
@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    """
    处理接收工作节点心跳信息的请求
    :param request: 请求对象
    :return: 包含工作节点是否存在的字典
    """
    # 获取请求中的 JSON 数据
    data = await request.json()
    # 调用控制器的 receive_heart_beat 方法接收工作节点心跳信息
    exist = controller.receive_heart_beat(
        data["worker_name"], data["queue_length"])
    return {"exist": exist}

# 定义处理客户端流式生成请求的 API 端点
@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request):
    """
    处理客户端流式生成请求
    :param request: 请求对象
    :return: 流式响应对象
    """
    # 获取请求中的 JSON 数据
    params = await request.json()
    # 调用控制器的 worker_api_generate_stream 方法处理流式生成请求
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator)

# 定义获取所有工作节点状态信息的 API 端点
@app.post("/worker_get_status")
async def worker_api_get_status(request: Request):
    """
    处理获取所有工作节点状态信息的请求
    :return: 所有工作节点的状态信息字典
    """
    # 调用控制器的 worker_api_get_status 方法获取所有工作节点状态信息
    return controller.worker_api_get_status()

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加主机地址参数，默认值为 "localhost"
    parser.add_argument("--host", type=str, default="localhost")
    # 添加端口号参数，默认值为 21001
    parser.add_argument("--port", type=int, default=21001)
    # 添加调度方法参数，可选值为 "lottery" 和 "shortest_queue"，默认值为 "shortest_queue"
    parser.add_argument("--dispatch-method", type=str, choices=[
        "lottery", "shortest_queue"], default="shortest_queue")
    # 解析命令行参数
    args = parser.parse_args()
    # 记录命令行参数信息
    logger.info(f"args: {args}")

    # 创建控制器实例
    controller = Controller(args.dispatch_method)
    # 使用 uvicorn 启动 FastAPI 应用
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")