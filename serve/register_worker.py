"""
Manually register workers.
这段代码的主要功能是手动将一个工作器（worker）注册到控制器（controller）上。通过命令行参数指定控制器的地址和工作器的名称，以及是否需要进行心跳检查。代码会构建一个 POST 请求，
将这些信息发送到控制器的注册接口，若请求成功（状态码为 200），则表示工作器注册成功。
"""
import argparse
import requests

if __name__ == "__main__":
    # 创建一个参数解析器对象，用于解析命令行参数
    parser = argparse.ArgumentParser()

    # 添加一个命令行参数 --controller-address，用于指定控制器的地址
    # 类型为字符串，在命令行中使用该参数时需要提供一个有效的地址
    parser.add_argument("--controller-address", type=str)

    # 添加一个命令行参数 --worker-name，用于指定工作器的名称
    # 类型为字符串，在命令行中使用该参数时需要提供一个有效的名称
    parser.add_argument("--worker-name", type=str)

    # 添加一个命令行参数 --check-heart-beat，用于指定是否检查心跳
    # 这是一个布尔类型的参数，使用该参数时表示需要检查心跳
    parser.add_argument("--check-heart-beat", action="store_true")

    # 解析命令行参数
    args = parser.parse_args()

    # 构建向控制器注册工作器的 URL
    url = args.controller_address + "/register_worker"

    # 构建要发送给控制器的 JSON 数据
    data = {
        # 工作器的名称，从命令行参数中获取
        "worker_name": args.worker_name,
        # 是否检查心跳，从命令行参数中获取
        "check_heart_beat": args.check_heart_beat,
        # 工作器的状态，这里初始化为 None
        "worker_status": None,
    }

    # 发送 POST 请求到控制器的注册接口，将数据以 JSON 格式发送
    r = requests.post(url, json=data)

    # 断言请求的状态码是否为 200，如果不是则抛出异常
    # 状态码 200 表示请求成功
    assert r.status_code == 200