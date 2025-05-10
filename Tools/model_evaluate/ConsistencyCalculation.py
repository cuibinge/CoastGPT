import numpy as np
import cv2
from sklearn.metrics import jaccard_score, accuracy_score
from skimage.morphology import label

class RemoteSensingEvaluator:
    def __init__(self):
        """
        初始化评估器
        """
        self.spatial_connectivity_scores = []
        self.boundary_accuracy_scores = []
        self.semantic_consistency_scores = []
        self.class_diversity_scores = []

    def evaluate_sample(self, y_pred, y_true):
        """
        评估单个样本
        :param y_pred: 模型预测结果 (height, width)
        :param y_true: 真实标签 (height, width)
        """
        # 空间连通性
        labeled_array, num_features = label(y_pred, connectivity=2, return_num=True)
        self.spatial_connectivity_scores.append(num_features)

        # 边界精度
        boundary_pred = cv2.Canny(y_pred.astype(np.uint8), 100, 200)
        boundary_true = cv2.Canny(y_true.astype(np.uint8), 100, 200)
        intersection = np.logical_and(boundary_pred, boundary_true)
        union = np.logical_or(boundary_pred, boundary_true)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        self.boundary_accuracy_scores.append(iou)

        # 语义一致性
        iou = jaccard_score(y_true.flatten(), y_pred.flatten(), average='macro')
        self.semantic_consistency_scores.append(iou)

        # 类别多样性
        accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
        self.class_diversity_scores.append(accuracy)

    def evaluate_dataset(self, y_preds, y_trues):
        """
        评估整个数据集
        :param y_preds: 模型预测结果 (batch_size, height, width)
        :param y_trues: 真实标签 (batch_size, height, width)
        """
        for y_pred, y_true in zip(y_preds, y_trues):
            self.evaluate_sample(y_pred, y_true)

    def get_results(self):
        """
        获取评估结果
        :return: 评估结果字典
        """
        results = {
            "Spatial Connectivity": np.mean(self.spatial_connectivity_scores),
            "Boundary Accuracy": np.mean(self.boundary_accuracy_scores),
            "Semantic Consistency": np.mean(self.semantic_consistency_scores),
            "Class Diversity": np.mean(self.class_diversity_scores)
        }
        return results

    def reset(self):
        """
        重置评估器
        """
        self.spatial_connectivity_scores = []
        self.boundary_accuracy_scores = []
        self.semantic_consistency_scores = []
        self.class_diversity_scores = []
