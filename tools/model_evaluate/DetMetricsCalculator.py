import torch
import numpy as np
from collections import defaultdict


class DetMetricsCalculator:
    """
    计算目标检测任务的评估指标，包括 mAP、precision 和 recall。
    """

    def __init__(self, iou_threshold=0.5):
        """
        初始化指标计算器。
        Args:
            iou_threshold (float): 计算TP时的IoU阈值，默认为0.5。
        """
        self.iou_threshold = iou_threshold

    def calculate_detection_metrics(self, predictions, ground_truths):
        """
        计算目标检测的评估指标。
        Args:
            predictions (list): 所有预测结果，每个元素是一个字典，包含 'boxes'、'labels'、'scores'。
            ground_truths (list): 所有真实值，每个元素是一个字典，包含 'boxes'、'labels'。
        Returns:
            dict: 包含评估指标的字典，如 mAP、precision、recall。
        """
        # 初始化指标字典
        metrics = {
            'mAP': 0.0,
            'precision': [],
            'recall': []
        }

        # 按类别统计TP、FP、FN
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            pred_scores = pred['scores']
            gt_boxes = gt['boxes']
            gt_labels = gt['labels']

            # 按类别匹配预测框和真实框
            for class_id in torch.unique(torch.cat((pred_labels, gt_labels))):
                class_id = class_id.item()

                # 过滤当前类别的预测框和真实框
                class_pred_mask = pred_labels == class_id
                class_gt_mask = gt_labels == class_id

                class_pred_boxes = pred_boxes[class_pred_mask]
                class_pred_scores = pred_scores[class_pred_mask]
                class_gt_boxes = gt_boxes[class_gt_mask]

                # 计算IoU矩阵
                ious = self._calculate_ious(class_pred_boxes, class_gt_boxes)

                # 按分数排序预测框
                sorted_indices = torch.argsort(class_pred_scores, descending=True)
                class_pred_boxes = class_pred_boxes[sorted_indices]

                # 标记哪些真实框已被匹配
                gt_matched = torch.zeros(len(class_gt_boxes), dtype=torch.bool, device=pred_boxes.device)

                for pred_idx, pred_box in enumerate(class_pred_boxes):
                    # 找到与当前预测框IoU最大的真实框
                    if len(class_gt_boxes) == 0:
                        # 没有真实框，当前预测为FP
                        class_stats[class_id]['fp'] += 1
                        continue

                    max_iou, gt_idx = torch.max(ious[pred_idx], dim=0)

                    if max_iou >= self.iou_threshold and not gt_matched[gt_idx]:
                        # TP
                        class_stats[class_id]['tp'] += 1
                        gt_matched[gt_idx] = True
                    else:
                        # FP
                        class_stats[class_id]['fp'] += 1

                # FN是未被匹配的真实框数量
                class_stats[class_id]['fn'] += len(class_gt_boxes) - gt_matched.sum().item()

        # 计算每个类别的Precision和Recall
        for class_id, stats in class_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            metrics['precision'].append(precision)
            metrics['recall'].append(recall)

        # 计算mAP（这里简化为平均Precision）
        metrics['mAP'] = np.mean(metrics['precision']) if metrics['precision'] else 0.0

        return metrics

    def _calculate_ious(self, pred_boxes, gt_boxes):
        """
        计算预测框和真实框之间的IoU矩阵。
        Args:
            pred_boxes (torch.Tensor): 预测框张量，形状为 [N, 4]。
            gt_boxes (torch.Tensor): 真实框张量，形状为 [M, 4]。
        Returns:
            torch.Tensor: IoU矩阵，形状为 [N, M]。
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return torch.zeros((len(pred_boxes), len(gt_boxes)), device=pred_boxes.device)

        # 计算所有预测框和真实框的IoU
        pred_boxes = pred_boxes.unsqueeze(1)  # [N, 1, 4]
        gt_boxes = gt_boxes.unsqueeze(0)  # [1, M, 4]

        # 计算交集面积
        intersection_min = torch.max(pred_boxes[:, :, :2], gt_boxes[:, :, :2])
        intersection_max = torch.min(pred_boxes[:, :, 2:], gt_boxes[:, :, 2:])
        intersection = torch.clamp(intersection_max - intersection_min, min=0)
        intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

        # 计算并集面积
        pred_area = (pred_boxes[:, :, 2] - pred_boxes[:, :, 0]) * (pred_boxes[:, :, 3] - pred_boxes[:, :, 1])
        gt_area = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0]) * (gt_boxes[:, :, 3] - gt_boxes[:, :, 1])
        union_area = pred_area + gt_area - intersection_area

        # 计算IoU
        ious = intersection_area / (union_area + 1e-6)  # 防止除零

        return ious
    
    def format_metrics(self, metrics, class_names):
        """
        格式化目标检测的评估指标，便于打印和日志记录。
        Args:
            metrics (dict): 包含评估指标的字典。
            class_names (list): 类别名称列表。
        Returns:
            str: 格式化后的指标字符串。
        """
        # 准备每个类别的指标数据
        class_metrics = []
        for i, class_name in enumerate(class_names):
            if i < len(metrics['precision']) and i < len(metrics['recall']):
                class_metrics.append([
                    class_name,
                    metrics['precision'][i],
                    metrics['recall'][i]
                ])
            else:
                class_metrics.append([
                    class_name,
                    0.0,
                    0.0
                ])

        # 计算每列的宽度
        class_name_width = max(len(name) for name in class_names) if class_names else 10
        metrics_names = ["Class", "Precision", "Recall"]
        columns_width = [
            max(class_name_width, len(metrics_names[0])),
            max(max(len(f"{v:.3f}") for v in metrics['precision']), len(metrics_names[1])),
            max(max(len(f"{v:.3f}") for v in metrics['recall']), len(metrics_names[2]))
        ]

        # 格式化表头
        header_format = f"{{:<{columns_width[0]}}}  {{:<{columns_width[1]}}}  {{:<{columns_width[2]}}}"
        header = header_format.format(*metrics_names)
        separator = "-" * (sum(columns_width) + 8)  # 8 是列之间的空格和分隔符的总长度

        # 格式化每个类别的行
        rows = []
        for cm in class_metrics:
            row_format = f"{{:<{columns_width[0]}}}  {{:<{columns_width[1]}.3f}}  {{:<{columns_width[2]}.3f}}"
            rows.append(row_format.format(cm[0], cm[1], cm[2]))

        # 格式化总体指标行
        overall_metrics = [
            "Overall",
            metrics['mAP'],
            np.mean(metrics['recall']) if metrics['recall'] else 0.0
        ]
        overall_format = f"{{:<{columns_width[0]}}}  {{:<{columns_width[1]}.3f}}  {{:<{columns_width[2]}.3f}}"
        overall_row = overall_format.format(*overall_metrics)

        # 组合所有行
        table = [header, separator] + rows + [separator, overall_row, separator]
        table_str = "\n".join(table)

        return table_str