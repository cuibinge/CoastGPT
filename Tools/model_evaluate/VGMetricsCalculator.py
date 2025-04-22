import re
import logging

logger = logging.getLogger("Visual Grounding evaluation")


class VGMetricsCalculator:

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

        # 交际区域的面积
        intersection_area = max(0, intersection_x2 - intersection_x1) * max(
            0, intersection_y2 - intersection_y1
        )

        # 计算两个边界框的面积
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)

        # 计算两个边界框的并集面积
        union_area = box1_area + box2_area - intersection_area

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
        parse_result = []
        fail_instance = 0

        for item in predictions:
            pred_match = re.findall(pattern, item["pred"])
            if len(pred_match) == 0:
                fail_instance += 1

            try:
                pred_result = [list(map(float, match.split(","))) for match in pred_match]
            except:
                fail_instance += 1
                continue

            target_match = re.findall(pattern, item["target"])
            target_result = [list(map(float, match.split(","))) for match in target_match]

            new_pred_result = []
            new_target_result = []

            for pred, target in zip(pred_result, target_result):
                if len(pred) == 4:
                    new_pred_result.append(pred)
                    new_target_result.append(target)
                elif len(pred) > 4:
                    while len(pred) != 4:
                        pred.pop()
                    new_pred_result.append(pred)
                    new_target_result.append(target)
                else:
                    fail_instance += 1

            if len(new_pred_result) > 0:
                parse_result.append(
                    dict(
                        filename=item["filename"],
                        pred=new_pred_result,
                        target=new_target_result,
                    )
                )

        count = 0
        total = 0

        for item in parse_result:
            preds = item["pred"]
            targets = item["target"]

            for pred, target in zip(preds, targets):
                iou_score = self.calculate_iou(pred, target)
                if iou_score > 0.5:
                    count += 1
                    print(item["filename"])
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