import json
import logging
import os

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import wandb
logger = logging.getLogger("train")


class CapMetricsCalculator:
    """
    图像描述生成任务的评估指标计算器。
    该类用于计算和格式化图像描述生成模型的各种评估指标，如 BLEU、METEOR、ROUGE_L 和 CIDEr。
    它使用 pycocoevalcap 库中的 COCOEvalCap 工具来进行指标计算。
    """

    def __init__(self, coco_gt, coco_res, tb_writer=None, use_wandb=False):
        """
        初始化图像描述生成指标计算器。

        参数:
            coco_gt (COCO): 包含真实标注数据的 COCO 对象。
            coco_res (COCO): 包含生成结果的 COCO 对象。
            tb_writer (SummaryWriter, 可选): 用于记录指标的 TensorBoard 写入器。默认为 None。
            use_wandb (bool, 可选): 是否使用 WandB 进行记录。默认为 False。
        """
        self.coco_gt = coco_gt
        self.coco_res = coco_res
        self.tb_writer = tb_writer
        self.use_wandb = use_wandb
        self.coco_eval = None
        self.metrics = None

    def calculate_metrics(self):
        """
        使用 COCOEvalCap 计算图像描述生成指标。

        返回:
            dict: 包含所有计算出的指标的字典。
        """
        self.coco_eval = COCOEvalCap(self.coco_gt, self.coco_res)
        self.coco_eval.evaluate()

        self.metrics = self.coco_eval.eval
        return self.metrics

    def format_metrics(self, cur_stat):
        """
        将计算出的指标格式化为逐行排列的字符串，并记录这些指标。

        参数:
            cur_stat (int): 当前训练状态（例如，轮次或迭代次数）。

        返回:
            str: 格式化后的指标字符串。
        """
        if self.metrics is None:
            raise ValueError("Metrics have not been calculated yet. Please call calculate_metrics first.")

        metrics_str = "Information on each evaluation metric:\n"
        for metric, score in self.metrics.items():
            metrics_str += f"{metric}: {score:.4f}\n"
            if self.tb_writer:
                self.tb_writer.add_scalar(f"eval/{metric}", score, cur_stat)
            if self.use_wandb:
                wandb.log({metric: score})

        logger.info(metrics_str)
        return metrics_str