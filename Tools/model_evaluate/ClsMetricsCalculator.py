import json
from sklearn.metrics import classification_report

# 预定义的 AID 类列表
aid_pre_defined_classes = [
    "Airport",
    "Bare Land",
    "Baseball Field",
    "Beach",
    "Bridge",
    "Center",
    "Church",
    "Commercial",
    "Dense Residential",
    "Desert",
    "Farmland",
    "Forest",
    "Industrial",
    "Medium Residential",
    "Meadow",
    "Mountain",
    "Pond",
    "Park",
    "Parking",
    "Playground",
    "Port",
    "Railway Station",
    "Resort",
    "River",
    "School",
    "Sparse Residential",
    "Square",
    "Storage Tanks",
    "Stadium",
    "Viaduct"
]
# 预定义的 EuroSAT 类列表
eurosat_pre_defined_classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]
# 预定义的 UCM 类列表
ucm_pre_defined_classes = [
    "agricultural",
    "airplane",
    "baseball diamond",
    "beach",
    "buildings",
    "chaparral",
    "dense residential",
    "forest",
    "freeway",
    "golf course",
    "harbor",
    "intersection",
    "medium residential",
    "mobile home park",
    "overpass",
    "parking lot",
    "river",
    "runway",
    "sparse residential",
    "storage tanks",
    "tennis court"
]
# 预定义的 WHU-RS19 类列表
whu_pre_defined_classes = [
    "Airport",
    "Beach",
    "Bridge",
    "Commercial",
    "Desert",
    "Farmland",
    "footballField",
    "Forest",
    "Industrial",
    "Meadow",
    "Mountain",
    "Park",
    "Parking",
    "Pond",
    "Port",
    "railwayStation",
    "Residential",
    "River",
    "Viaduct"
]
# 预定义的 NWPU 类列表
nwpu_pre_defined_classes = [
    "airplane",
    "airport",
    "baseball_diamond",
    "basketball_court",
    "beach",
    "bridge",
    "chaparral",
    "church",
    "circular_farmland",
    "cloud",
    "commercial_area",
    "dense_residential",
    "desert",
    "forest",
    "freeway",
    "golf_course",
    "ground_track_field",
    "harbor",
    "industrial_area",
    "intersection",
    "island",
    "lake",
    "medium_residential",
    "meadow",
    "mobile_home_park",
    "mountain",
    "overpass",
    "palace",
    "parking_lot",
    "railway",
    "railway_station",
    "rectangular_farmland",
    "river",
    "roundabout",
    "runway",
    "sea_ice",
    "ship",
    "snowberg",
    "sparse_residential",
    "storage_tank",
    "tennis_court",
    "stadium",
    "thermal_power_station",
    "terrace",
    "wetland"
]
# 读取图像分类任务的结果文件
def load_data(jsonl_file):
	# 存储成对样本的真实值
    ground_truths = []
	# 存储成对样本的预测值
    predictions = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            ground_truth = item.get('ground_truth', '')
            prediction = item.get('pred', '')
            ground_truths.append(ground_truth.strip())
            predictions.append(prediction.strip())
    return ground_truths, predictions

def generate_classification_report(ground_truths, predictions, classes):
    # 生成分类报告
    report = classification_report(
        ground_truths,
        predictions,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    return report

def print_classification_report(report, correct_predictions, total_predictions):
    # 打印分类报告
    print(f"{'Class':<20} Precision  Recall   F1-Score  Support")
    for cls in report:
        if cls == 'accuracy' or cls == 'macro avg' or cls == 'weighted avg':
            continue
        metrics = report[cls]
        print(f"{cls:<20} {metrics['precision']:.3f}   {metrics['recall']:.3f}   {metrics['f1-score']:.3f}   {metrics['support']}")
    
    print(f"\nAccuracy: {correct_predictions/total_predictions}")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.3f}")
    print(f"Macro Avg Recall: {report['macro avg']['recall']:.3f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted Avg Precision: {report['weighted avg']['precision']:.3f}")
    print(f"Weighted Avg Recall: {report['weighted avg']['recall']:.3f}")
    print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.3f}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Incorrect Predictions: {total_predictions - correct_predictions}")
    print(f"Total Predictions: {total_predictions}")

def main():
	# 传入结果文件
    input_jsonl = 'CLS_results.jsonl'
    ground_truths, predictions = load_data(input_jsonl)
    
    # 统计正确的预测数目和总预测数目
    correct_predictions = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == pred)
    total_predictions = len(ground_truths)
    
    # 生成分类报告
    report = generate_classification_report(ground_truths, predictions, aid_pre_defined_classes)
    
    # 打印分类报告和统计信息
    print_classification_report(report, correct_predictions, total_predictions)

if __name__ == "__main__":
    main()