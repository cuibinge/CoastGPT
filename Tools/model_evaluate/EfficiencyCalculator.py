import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    total_training_time = 0
    per_epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        per_epoch_times.append(epoch_time)
        total_training_time += epoch_time

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Time: {epoch_time:.2f}s")

    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Average Per-Epoch Training Time: {np.mean(per_epoch_times):.2f}s")

    return total_training_time, per_epoch_times

# 训练模型
total_training_time, per_epoch_times = train_model(model, dataloader, criterion, optimizer, num_epochs=10)

def evaluate_inference(model, dataloader):
    model.eval()
    total_inference_time = 0
    total_samples = 0
    memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            total_samples += inputs.size(0)

    average_inference_time = total_inference_time / total_samples
    throughput = total_samples / total_inference_time

    print(f"Average Inference Time: {average_inference_time:.4f}s")
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"Memory Usage: {memory_usage / (1024 ** 2):.2f} MB")

    return average_inference_time, throughput, memory_usage

# 评估推理效率
average_inference_time, throughput, memory_usage = evaluate_inference(model, dataloader)