import re
import logging

logger = logging.getLogger("Visual Grounding evaluation")


class VQMetricsCalculator:

    def __init__(self):
        pass

    def calculate_iou(self, box1, box2):
        """
        计算两个边界框的交并比（IOU）。
        :param box1: 第一个边界框的坐标 (x1, y1, x2, y2)
        :param box2: 第二个边界框的坐标 (x3, y3, x4, y4)
        :return: IOU 值
        """
        # 边界框1
		x1, y1, x2, y2 = box1
		# 边界框2
		x3, y3, x4, y4 = box2

		# 计算两个边界框的交集
		intersection_x1 = max(x1, x3)
		intersection_y1 = max(y1, y3)
		intersection_x2 = min(x2, x4)
		intersection_y2 = min(y2, y4)

		# 计算交集区域的宽度和高度
		intersection_width = intersection_x2 - intersection_x1 + 1
		intersection_height = intersection_y2 - intersection_y1 + 1
		# 如何两个边界框无重叠，设置交集面积为0
		if intersection_width <= 0 or intersection_height <= 0:
			intersection_area = 0
		# 计算交集区域的面积
		else:
			intersection_area = intersection_width * intersection_height
    
		# 计算边界框1和边界框2的面积
		box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
		box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

		# 计算两个边界框的并集面积
		union_area = box1_area + box2_area - intersection_area

		# 避免除以零
		if union_area == 0:
			return 0.0

		# 计算IOU值
		iou = intersection_area / union_area

		return iou

    def evaluate(self, predictions):
        """
        评估预测结果，根据阈值计算准确率
        :param predictions: 包含预测结果和真实结果的列表
        :return: 准确率、失败样本数和包含失败样本的准确率
        """
        # 正则表达式，用于从文本中提取边界框坐标
        pattern = r"\[([0-9., ]+)\]"
		# 存储解析后的结果
        parse_result = []  
		# 统计失败样本数
        fail_instance = 0  

        # 遍历每个预测结果
        for item in predictions:
            # 提取预测的边界框坐标
            pred_match = re.findall(pattern, item["pred"])
            # 提取真实的边界框坐标
            target_match = re.findall(pattern, item["target"])

            # 如果预测或真实结果中没有边界框坐标，则标记为失败样本
            if len(pred_match) == 0 or len(target_match) == 0:
                fail_instance += 1
                continue

            try:
                # 将提取的边界框坐标转换为浮点数列表
                pred_result = [list(map(float, match.split(","))) for match in pred_match]
                target_result = [list(map(float, match.split(","))) for match in target_match]
            except:
                # 如果转换失败，则标记为失败样本
                fail_instance += 1
                continue

            new_pred_result = []
            new_target_result = []
            # 遍历预测和真实的边界框坐标
            for pred, target in zip(pred_result, target_result):
                # 如果预测的边界框坐标长度不为4，则截断或标记为失败样本
                if len(pred) == 4:
                    new_pred_result.append(pred)
                    new_target_result.append(target)
				# 如果预测的边界框坐标长度大于4，截断
                elif len(pred) > 4:
                    while len(pred) != 4:
                        pred.pop()
                    new_pred_result.append(pred)
                    new_target_result.append(target)
				# 如果预测的边界框坐标长度小于4，标记为失败样本
                else:
                    fail_instance += 1

            # 将有效的预测和真实边界框坐标添加到解析结果中
            if len(new_pred_result) > 0:
                parse_result.append(
                    dict(
                        filename=item["filename"],
                        pred=new_pred_result,
                        target=new_target_result,
                    )
                )
		# 正确预测的样本数
        count = 0  
		# 总样本数
        total = 0  
        # 遍历解析后的结果
        for item in parse_result:
            preds = item["pred"]
            targets = item["target"]

            # 遍历每个预测和真实的边界框坐标
            for pred, target in zip(preds, targets):
                # 计算IOU值
                iou_score = self.calculate_iou(pred, target)
                # 如果IOU值大于0.5，则认为预测正确
                if iou_score > 0.5:
                    count += 1
                total += 1

        # 计算准确率
        accuracy = count / total * 100 if total > 0 else 0
        # 计算包含失败样本的准确率
        accuracy_fail = count / (total + fail_instance) * 100 if (total + fail_instance) > 0 else 0

        # 记录日志
        logger.info(f"Accuracy: {accuracy}%")
        logger.info(f"Fail Sample: {fail_instance}")
        logger.info(f"Accuracy With Fail Sample: {accuracy_fail}%")

        return accuracy, fail_instance, accuracy_fail