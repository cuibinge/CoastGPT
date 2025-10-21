# Evaluation

### Classification

```python
python main_cls.py \
     -c configs/inference.yaml \             # config file
     --model-path ${PathToCheckpoint}.pt \   # path to checkpoint end with .pt
     --data-path ${ImageFolder} \            # path to classification image folder
     --accelerator "gpu" \                   # change if you need ["mps", "cpu", "gpu"]
     --workers 4 \
     --enable-amp True \
     --output ${YourOutputDir} \             # Path to output (result, metric etc.)
     --batch-size 8 
```

### dataset

https://pan.baidu.com/s/1NTqtBH5a0GqWD0VAslkG-g 提取码: i46e 



### Visual Question Answering

```python
## Inference: Generate a prediction result JSON file
python main_vqa_infer.py \
     -c Configs/inference.yaml \              # config file
     --model-path ${PathToCheckpoint}.pt \    # path to checkpoint end with .pt
     --data-path ${Image} \                   # path to image folder
     --accelerator "gpu" \                    # change if you need ["mps", "cpu", "gpu"]
     --workers 2 \
     --enable-amp True \
     --output ${YourOutputDir} \              # Path to output (result, metric etc.)
     --batch-size 1 \                         # It's better to use batchsize 1, since we find batch inference
     --data-target ${ParsedLabelJsonPath} \   # Folder for storing JSON files.
     --data-type "HR"                         # choose from ["HR", "LR"]
        
## Evaluate: Calculate the accuracy of visual question answering tasks
python VQAMetricsCalculator.py \
     --result-file ${PathToResultFile}.json \ # Result JSON file
     --output-json ${YourOutputDir}.json      # A JSON file storing the evaluation results
```

### dataset

https://pan.baidu.com/s/1jrNjjN_3zWDbaWE5WcxcWA 提取码: y2rq 



### Visual Grounding

```python
## Inference: Generate a prediction result JSON file
python main_vg_infer.py \
     -c Configs/inference.yaml \              # config file
     --model-path ${PathToCheckpoint}.pt \    # path to checkpoint end with .pt
     --data-path ${ImageFolder} \             # path to image folder
     --accelerator "gpu" \                    # change if you need ["mps", "cpu", "gpu"]
     --workers 2 \
     --enable-amp True \
     --output ${YourOutputDir} \              # Path to output (result, metric etc.)
     --batch-size 1 \                         # It's better to use batchsize 1, since we find batch inference
     --data-target ${ParsedLabelJsonPath}     # is not stable.
   
## Evaluate: Calculate metrics for visual grounding tasks
python VGMetricsCalculator.py \
     --json-file ${PathToResultFile}.json   # Result JSON file
```

### dataset

https://pan.baidu.com/s/1JspW2spCXQGB8cf-UgfBdg 提取码: em2v 


