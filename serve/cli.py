import argparse
import torch

# 从 llava 库的 constants 模块导入图像相关的标记常量
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# 从 llava 库的 conversation 模块导入对话模板和分隔符样式
from llava.conversation import conv_templates, SeparatorStyle
# 从 llava 库的 model.builder 模块导入加载预训练模型的函数
from llava.model.builder import load_pretrained_model
# 从 llava 库的 utils 模块导入禁用 PyTorch 初始化的函数
from llava.utils import disable_torch_init
# 从 llava 库的 mm_utils 模块导入处理图像、图像标记分词、获取模型名称和停止条件的工具函数
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import matplotlib.pyplot as plt
# 从 matplotlib.patches 模块导入绘制多边形、圆形和矩形的类
from matplotlib.patches import Polygon, Circle, Rectangle
# 从 matplotlib.collections 模块导入 PatchCollection 类，用于管理多个图形
from matplotlib.collections import PatchCollection
from PIL import Image
import numpy as np
import requests
from PIL import Image
from io import BytesIO
# 从 transformers 库导入文本流式输出器
from transformers import TextStreamer
import math
import cv2


# 定义一个函数，用于按比例缩放边界框
def scale_bounding_box(box, scale_factor=1.2):
    # 提取边界框的左上角和右下角坐标
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = box

    # 计算边界框的宽度和高度
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y

    # 按比例缩放宽度和高度
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # 计算缩放后边界框的新坐标
    new_top_left_x = top_left_x - int((new_width - width) / 2)
    new_top_left_y = top_left_y - int((new_height - height) / 2)
    new_bottom_right_x = new_top_left_x + new_width
    new_bottom_right_y = new_top_left_y + new_height

    # 返回缩放后的边界框坐标
    return [new_top_left_x, new_top_left_y, new_bottom_right_x, new_bottom_right_y]


# 定义一个函数，用于加载图像
def load_image(image_file):
    # 检查图像文件是否为网络链接
    if image_file.startswith('http://') or image_file.startswith('https://'):
        # 如果是网络链接，发送请求获取图像内容
        response = requests.get(image_file)
        # 打开图像并转换为 RGB 格式
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        # 如果是本地文件，直接打开并转换为 RGB 格式
        image = Image.open(image_file).convert('RGB')
    return image


# 定义一个函数，用于将边界框和角度转换为多边形坐标
def bbox_and_angle_to_polygon(x1, y1, x2, y2, a):
    # 计算边界框的中心坐标
    x_ctr = (x1 + x2) / 2
    y_ctr = (y1 + y2) / 2

    # 计算边界框的宽度和高度
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    # 将角度转换为弧度
    angle_rad = math.radians(a)

    # 计算旋转后边界框四个角的坐标
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    x1_rot = cos_a * (-w / 2) - sin_a * (-h / 2) + x_ctr
    y1_rot = sin_a * (-w / 2) + cos_a * (-h / 2) + y_ctr

    x2_rot = cos_a * (w / 2) - sin_a * (-h / 2) + x_ctr
    y2_rot = sin_a * (w / 2) + cos_a * (-h / 2) + y_ctr

    x3_rot = cos_a * (w / 2) - sin_a * (h / 2) + x_ctr
    y3_rot = sin_a * (w / 2) + cos_a * (h / 2) + y_ctr

    x4_rot = cos_a * (-w / 2) - sin_a * (h / 2) + x_ctr
    y4_rot = sin_a * (-w / 2) + cos_a * (h / 2) + y_ctr

    # 返回多边形的坐标
    polygon_coords = np.array((x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot))

    return polygon_coords


# 主函数，处理命令行参数并运行对话流程
def main(args):
    # 禁用 PyTorch 的初始化，以提高性能
    disable_torch_init()

    # 从模型路径中获取模型名称
    model_name = get_model_name_from_path(args.model_path)
    # 加载预训练模型、分词器、图像处理器和上下文长度
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           args.load_8bit, args.load_4bit,
                                                                           device=args.device)

    # 根据模型名称推断对话模式
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # 检查用户指定的对话模式和自动推断的对话模式是否一致
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        # 如果不一致，打印警告信息并使用用户指定的对话模式
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                              args.conv_mode,
                                                                                                              args.conv_mode))
    else:
        # 如果一致或用户未指定，使用自动推断的对话模式
        args.conv_mode = conv_mode

    # 从对话模板中复制当前对话模式的模板
    conv = conv_templates[args.conv_mode].copy()
    # 根据模型名称确定对话的角色
    if "mpt" in model_name.lower():
        # 如果是 MPT 模型，固定角色为 'user' 和 'assistant'
        roles = ('user', 'assistant')
    else:
        # 否则，使用对话模板中定义的角色
        roles = conv.roles

    # 加载图像文件
    image = load_image(args.image_file)
    # 处理图像，将其转换为模型可以接受的张量形式
    image_tensor = process_images([image], image_processor, args)
    # 如果图像张量是列表形式，将每个张量移动到模型设备并转换为 float16 类型
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        # 否则，直接将张量移动到模型设备并转换为 float16 类型
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # 进入循环，持续接收用户输入并生成回复
    while True:
        try:
            # 提示用户输入，并显示用户角色
            inp = input(f"{roles[0]}: ")
        except EOFError:
            # 如果用户输入 EOF（如 Ctrl+D），将输入设为空字符串
            inp = ""
        if not inp:
            # 如果输入为空，打印退出信息并结束循环
            print("exit...")
            break

        # 打印助手角色提示，准备输出回复
        print(f"{roles[1]}: ", end="")

        # 如果图像还未处理
        if image is not None:
            # 第一条消息，需要插入图像标记
            if model.config.mm_use_im_start_end:
                # 如果模型配置使用图像起始和结束标记，插入相应标记
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                # 否则，只插入默认图像标记
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            # 将包含图像标记的消息添加到对话中
            conv.append_message(conv.roles[0], inp)
            # 处理完图像后，将图像置为 None
            image = None
        else:
            # 后续消息，直接添加到对话中
            conv.append_message(conv.roles[0], inp)
        # 在对话中添加助手的占位消息
        conv.append_message(conv.roles[1], None)
        # 从对话模板中获取完整的提示信息
        prompt = conv.get_prompt()

        # 将提示信息转换为输入 ID 张量，并移动到 CUDA 设备
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # 根据对话分隔符样式确定停止字符串
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # 将停止字符串作为关键词
        keywords = [stop_str]
        # 创建停止条件对象
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # 创建文本流式输出器，用于实时输出模型生成的文本
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        # 插入调试断点，方便调试
        import pdb;
        pdb.set_trace()
        # 在推理模式下生成回复
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        # 解码生成的输出 ID，去除输入部分和特殊标记
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        # 将生成的回复添加到对话中
        conv.messages[-1][-1] = outputs
        # bboxes=[]
        # print(inp)
        # # if ('[refer]') or ("[grounding]") in inp:
        # output=outputs.replace('</s>','')
        # print(output)
        # # import pdb;pdb.set_trace()
        # bboxes = np.array([int(x) for y in output.replace("|", "").split("}") for x in y.replace("><", ",").replace(">", "").replace("<", "").replace("}", "").replace("{", "").split(',') if x !=""]).astype(np.float32)
        # remainder = len(bboxes)%5
        # if remainder >0:
        #     bboxes = bboxes[:-remainder]
        # # bboxes[1]=100-bboxes[1]
        # # bboxes=bboxes
        # # scaled_bbox=scale_bounding_box(bboxes[:-1], scale_factor=1.3)
        # # bboxes=scaled_bbox.append(bboxes[-1])
        # # bboxes = bboxes.reshape(-1, 5)
        # bboxes=bboxes.tolist()
        # bboxes=[int(bbox*5.04) for bbox in bboxes]
        # bboxes = np.array([bbox_and_angle_to_polygon(bboxes[0],bboxes[1],bboxes[2],bboxes[3],bboxes[4])])
        # # print(bboxes)
        # bboxes=bboxes.reshape(4,2)

        # image = cv2.imread(args.image_file)
        # # print(image.shape)
        # image=cv2.resize(image,(504,504))
        # plt.imshow(image)
        # # import pdb;pdb.set_trace()
        # polygons=[Polygon(bboxes)]
        # plt.axis('off')
        # ax = plt.gca()
        # ax.set_autoscale_on(False)
        # p = PatchCollection(polygons, facecolors='none', edgecolors=[(0,255,0)], linewidths=2)
        # ax.add_collection(p)
        # # print('hello')
        # plt.savefig('/share/data/drive_3/kartik/LLaVA/output_images/output.jpg')

        # 如果启用了调试模式，打印提示信息和生成的回复
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


# 程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加模型路径参数，默认值为 "facebook/opt-350m"
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # 添加模型基础路径参数，默认值为 None
    parser.add_argument("--model-base", type=str, default=None)
    # 添加图像文件路径参数，该参数为必需项
    parser.add_argument("--image-file", type=str, required=True)
    # 添加设备参数，默认值为 "cuda"
    parser.add_argument("--device", type=str, default="cuda")
    # 添加对话模式参数，默认值为 None
    parser.add_argument("--conv-mode", type=str, default=None)
    # 添加温度参数，用于控制生成文本的随机性，默认值为 0.2
    parser.add_argument("--temperature", type=float, default=0.2)
    # 添加最大新生成标记数参数，默认值为 512
    parser.add_argument("--max-new-tokens", type=int, default=512)
    # 添加是否加载 8 位量化模型的参数，默认不加载
    parser.add_argument("--load-8bit", action="store_true")
    # 添加是否加载 4 位量化模型的参数，默认不加载
    parser.add_argument("--load-4bit", action="store_true")
    # 添加调试模式参数，默认不启用
    parser.add_argument("--debug", action="store_true")
    # 添加图像宽高比处理方式参数，默认值为 'pad'
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    # 解析命令行参数
    args = parser.parse_args()
    # 调用主函数开始运行程序
    main(args)