海岸线视觉定位数据集构建流程：
1. 提取海岸线轮廓：运行extract_coastline_footprint.py，把遥感影像对应的海岸线范围切割出来；
2. 影像与矢量分割：运行split_image_coastline_512.py，将遥感影像与矢量海岸线都分割成512×512，步长为256；
3. 有效海岸线筛选：运行 valid_coastline_extractor.py，过滤掉不含有效海岸线的矢量切片，保留带有海岸线的矢量数据。
4. 影像与海岸线匹配：运行match_coastline_with_images.py，筛选并保留能够与海岸线矢量对应的影像切片；
5. 二次筛选与校正：由于部分影像存在黑边，或海岸线与影像存在轻微偏差，需要进一步筛选。运行refine_coastline_image_pairs.py，对影像与海岸线配对结果进行二次校正与筛选。
6. 生成海岸线拐点序列：运行generate_coastline_keypoints.py，将海岸线矢量数据的拐点转换为像素坐标，并进行归一化处理，最终生成海岸线拐点坐标序列。
