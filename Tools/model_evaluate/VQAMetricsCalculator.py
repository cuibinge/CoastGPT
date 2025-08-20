import json

def VQAMetricsCalculator(json_file):
    # 初始化一个字典来存储每种类型的统计信息
    type_stats = {
        "comp": {"correct": 0, "total": 0},
        "presence": {"correct": 0, "total": 0},
        "rural_urban": {"correct": 0, "total": 0}
    }

    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 遍历每个数据项并统计
    for data in data_list:
        question_type = data.get('types')
        if question_type not in type_stats:
            continue  # 跳过不在预定义类型中的数据

        # 更新总数
        type_stats[question_type]["total"] += 1

        # 检查预测是否正确
        if data.get('pred').lower() == data.get('target').lower():
            type_stats[question_type]["correct"] += 1

    # 计算准确率并准备输出
    result = {}
    for question_type, stats in type_stats.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
        else:
            accuracy = 0.0
        result[question_type] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": accuracy
        }

    # 输出结果到JSON文件
    output_file = "lhrs_lr_test_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

    print(f"结果已保存到 {output_file}")

# 调用函数
VQAMetricsCalculator('eval_save_file.json')  # 替换为你的JSON文件路径