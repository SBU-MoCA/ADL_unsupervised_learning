import sys
sys.path.append("/home/mengjingliu/ADL_unsupervised_learning/")
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from model import Model
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ReprogramLLM.data_loader import wrapper_dataLoader
from sklearn.metrics import accuracy_score
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ExponentialLR
import datetime

from torch.cuda.amp import autocast, GradScaler

import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

# Set environment variables
os.environ['MASTER_ADDR'] = 'localhost'    # Adjust as necessary
os.environ['MASTER_PORT'] = '12355'        # Choose an open port
os.environ['RANK'] = '0'                   # Example: process 0
os.environ['WORLD_SIZE'] = '2'             # Example: 2 processes

# Initialize the process group
# dist.init_process_group(backend='nccl', init_method='env://')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Initialize the distributed environment.
dist.init_process_group(backend='nccl')

scaler = GradScaler()

from utils import plot_loss

os.chdir("/home/mengjingliu/ADL_unsupervised_learning/")
# os.environ["CUDA_VISIBLE_DEVICE"] = "1"



configs = {
    "llm_layers": 2,
    "IT": {
        "in_features": 1456,
        "out_features": 64
        
    },
    "OM": {
        "in_features": 262144,
        # "in_features": 64*256,
        "out_features": 5
        
    },
    
    "channels": [1, 32, 64]
    
}

# model_name_or_path = "prajjwal1/bert-mini"
# model_name_or_path = "google-bert/bert-base-uncased"
model_name_or_path = "meta-llama/Llama-2-7b-hf"
title = model_name_or_path.replace('/', '_')

model = Model(model_name_or_path, configs)
model = model.half()


if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    model.cuda()
    model = DDP(model)


else:
    print("CUDA is not available. Training on CPU.")

trained_parameters = []

for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)
optimizer = torch.optim.SGD(trained_parameters, lr=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.98)

criterion = nn.CrossEntropyLoss()  # For classification tasks

path_list = ["/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6",
             "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5",
             "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4",
             "/home/mengjingliu/Vid2Doppler/data/2023_11_17/HAR3",
             "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR2"]
train_loader, test_loader = wrapper_dataLoader(path_list, batch_size=16, if_resize=False, if_replicate_channels=False)

training_losses = []
test_losses = []
test_accuracies = []

test_interval = 1

best_accuracy = 0
for epoch in range(100):  # Loop over the dataset multiple times
    for k in range(test_interval):     # test the model every 10 epoches
        running_loss = 0.0
        for inputs, labels in train_loader:
            
            # if os.path.exists(f"results/best_model_{title}.pth"):
            #     state_dict = torch.load(f"results/best_model_{title}.pth")
            #     model.load_state_dict(state_dict)
            
            optimizer.zero_grad()
            with autocast():
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                # scaler.scale(loss).backward()
            optimizer.step()
            # scaler.unscale_(optimizer)

            # scaler.step(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters())
            # scaler.update()
    
            running_loss += loss.item()
        
        training_losses.append(running_loss / len(train_loader))
            
        scheduler.step()
        print(f'Epoch {epoch * test_interval + k + 1}, Time: {str(datetime.datetime.now())}, Training Loss: {running_loss / len(train_loader)}')
    
    test_loss = 0.0
    outputs = torch.tensor([])
    labels = torch.tensor([])
    with torch.no_grad():
        for input, label in test_loader:
            # input, label = input.to(device), label.to(device)
            # Forward pass
            output = model(input.cuda())
            loss = criterion(output, label.cuda())
            test_loss += loss.item()
            outputs = torch.cat([outputs, output.to("cpu")])
            labels = torch.cat([labels, label])
        
        test_losses.append(test_loss / len(test_loader))
    
    print(f'Epoch {epoch * test_interval + k + 1}, Testing Loss: {test_loss / len(test_loader)}')
    
    accuracy = Accuracy(task="multiclass", num_classes=5)
    accuracy = accuracy(outputs, labels)
    test_accuracies.append(accuracy)
    print(f'Epoch {epoch * test_interval + k + 1}, Testing accuracy: {accuracy}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy

        # do not save the whole model. should only save trainable parameters
        torch.save(model.state_dict(), f'ReprogramLLM/results/best_model_{title}.pth')
        print(f"Epoch {epoch * test_interval + k + 1}: New best model found and saved.")
    
    print("--------------------------------------------------------------")
    
    plot_loss("ReprogramLLM/results", f"loss_{title}.png", np.array(training_losses), np.array(test_losses), np.array(test_accuracies), test_interval)

print('Finished Training')


