import torch
import numpy as np
from collections import defaultdict


class SegMetricsCalculator:
    """
    计算语义分割任务的评估指标，包括 IoU、Precision、Recall、F1-score 等。
    """

    def __init__(self, num_classes, class_names):
        """
        初始化指标计算器。
        Args:
            num_classes (int): 类别数量。
            class_names (list): 类别名称列表。
        """
        self.num_classes = num_classes
        self.class_names = class_names

    def compute_segmentation_metrics(self, confusion_matrix):
        """
        根据混淆矩阵计算语义分割的评估指标。
        Args:
            confusion_matrix (torch.Tensor): 混淆矩阵，形状为 [num_classes, num_classes]。
        Returns:
            dict: 包含评估指标的字典。
        """
        # 初始化指标字典
        metrics = {
            'IoU': [],
            'Precision': [],
            'Recall': [],
            'F1': []
        }

        # 计算每个类别的指标
        for i in range(self.num_classes):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp

            # 计算 IoU
            iou = tp / (tp + fp + fn + 1e-6)
            metrics['IoU'].append(iou.item())

            # 计算 Precision
            precision = tp / (tp + fp + 1e-6)
            metrics['Precision'].append(precision.item())

            # 计算 Recall
            recall = tp / (tp + fn + 1e-6)
            metrics['Recall'].append(recall.item())

            # 计算 F1-score
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            metrics['F1'].append(f1.item())

        # 计算平均指标
        metrics['mIoU'] = np.nanmean(metrics['IoU'])
        metrics['mPrecision'] = np.nanmean(metrics['Precision'])
        metrics['mRecall'] = np.nanmean(metrics['Recall'])
        metrics['mF1'] = np.nanmean(metrics['F1'])

        return metrics

    def format_metrics(self, metrics):
        """
        格式化评估指标，便于打印和日志记录。
        Args:
            metrics (dict): 包含评估指标的字典。
        Returns:
            str: 格式化后的指标字符串。
        """
        # 准备每个类别的指标数据
        class_metrics = []
        for i, class_name in enumerate(self.class_names):
            class_metrics.append([
                class_name,
                metrics['IoU'][i],
                metrics['F1'][i],
                metrics['Precision'][i],
                metrics['Recall'][i]
            ])

        # 计算每列的宽度
        class_name_width = max(len(name) for name in self.class_names) if self.class_names else 10
        metrics_names = ["Class", "IoU", "F1", "Precision", "Recall"]
        columns_width = [
            max(class_name_width, len(metrics_names[0])),
            max(max(len(f"{v:.3f}") for v in metrics['IoU']), len(metrics_names[1])),
            max(max(len(f"{v:.3f}") for v in metrics['F1']), len(metrics_names[2])),
            max(max(len(f"{v:.3f}") for v in metrics['Precision']), len(metrics_names[3])),
            max(max(len(f"{v:.3f}") for v in metrics['Recall']), len(metrics_names[4]))
        ]

        # 格式化表头
        header_format = f"{{:<{columns_width[0]}}}  {{:<{columns_width[1]}}}  {{:<{columns_width[2]}}}  {{:<{columns_width[3]}}}  {{:<{columns_width[4]}}}"
        header = header_format.format(*metrics_names)
        separator = "-" * (sum(columns_width) + 8)  # 8 是列之间的空格和分隔符的总长度

        # 格式化每个类别的行
        rows = []
        for cm in class_metrics:
            row_format = f"{{:<{columns_width[0]}}}  {{:<{columns_width[1]}.3f}}  {{:<{columns_width[2]}.3f}}  {{:<{columns_width[3]}.3f}}  {{:<{columns_width[4]}.3f}}"
            rows.append(row_format.format(cm[0], cm[1], cm[2], cm[3], cm[4]))

        # 格式化总体指标行
        overall_metrics = [
            "Overall",
            metrics['mIoU'],
            metrics['mF1'],
            metrics['mPrecision'],
            metrics['mRecall']
        ]
        overall_format = f"{{:<{columns_width[0]}}}  {{:<{columns_width[1]}.3f}}  {{:<{columns_width[2]}.3f}}  {{:<{columns_width[3]}.3f}}  {{:<{columns_width[4]}.3f}}"
        overall_row = overall_format.format(*overall_metrics)

        # 组合所有行
        table = [header, separator] + rows + [separator, overall_row, separator]
        table_str = "\n".join(table)

        return table_str