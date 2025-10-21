"""
该脚本用于对模型视觉定位的预测结果进行评估：
"""
import json
import re
import numpy as np
import argparse
from sklearn.metrics import average_precision_score

def parse_boxes(s):
    """解析边界框字符串，支持多个框的情况"""
    boxes = []
    # 匹配所有方括号内的坐标
    matches = re.findall(r'\[([^\]]+)\]', s)
    for match in matches:
        try:
            # 分割坐标字符串并转换为浮点数
            coords = [float(x.strip()) for x in match.split(',')]
            if len(coords) == 4:
                boxes.append(coords)
        except:
            continue
    return boxes

def calculate_iou(bbox_pred, bbox_gt):
    """计算两个边界框的 IoU"""
    # 确保坐标有效性
    x1_pred, y1_pred, x2_pred, y2_pred = bbox_pred
    x1_gt, y1_gt, x2_gt, y2_gt = bbox_gt
    
    # 计算交集坐标
    xi1 = max(x1_pred, x1_gt)
    yi1 = max(y1_pred, y1_gt)
    xi2 = min(x2_pred, x2_gt)
    yi2 = min(y2_pred, y2_gt)
    
    # 计算交集面积
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0
    
    # 计算并集面积
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = pred_area + gt_area - inter_area
    
    return inter_area / union_area

def main(json_file):
    # 存储结果
    all_scores = []
    all_ious = []
    y_true = []  # 用于AP计算的二进制标签
    correct_count = 0
    correct_count25 = 0
    
    # 读取并处理 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for item in data:
        # 解析边界框
        pred_boxes = parse_boxes(item['pred'])
        gt_boxes = parse_boxes(item['target'])
        
        if not pred_boxes or not gt_boxes:
            continue  # 跳过解析失败的条目
        
        # 找到匹配的框对（使用最小索引数量）
        n_pairs = min(len(pred_boxes), len(gt_boxes))
        
        for i in range(n_pairs):
            # 计算 IoU
            iou = calculate_iou(pred_boxes[i], gt_boxes[i])
            # 使用默认置信度1.0（因为新格式没有提供置信度）
            score = 1.0
            
            all_ious.append(iou)
            all_scores.append(score)
            
            # 检查是否大于0.5并计数
            if iou >= 0.5:
                correct_count += 1
                y_true.append(1)
            else:
                y_true.append(0)
            
            if iou >= 0.25:
                correct_count25 += 1
    
    # 计算 Acc@IoU
    accuracy25 = np.mean(np.array(all_ious) >= 0.25) if all_ious else 0
    accuracy = np.mean(np.array(all_ious) >= 0.5) if all_ious else 0
    
    # 计算 AP@0.5
    ap = average_precision_score(y_true, all_scores) if y_true else 0
    
    # 计算平均 IoU
    mean_iou = np.mean(all_ious) if all_ious else 0
    
    # 输出结果
    print(f"Acc@IoU=0.25: {accuracy25:}")
    print(f"Acc@IoU=0.5: {accuracy:}")
    print(f"AP@IoU=0.5: {ap:}")
    print(f"Samples with IoU >= 0.25: {correct_count25}/{len(all_ious)}")
    print(f"Samples with IoU >= 0.5: {correct_count}/{len(all_ious)}")
    print(f"Total samples: {len(all_ious)}")


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='计算视觉定位任务的评估指标')
    parser.add_argument(
        '--json-file', 
        type=str, 
        required=True,
        help='包含预测和真实边界框的JSON文件路径'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用主函数，传入JSON文件路径
    main(args.json_file)