import sys
sys.path.append("/home/mengjingliu/ADL_unsupervised_learning/")
import datetime
from torchmetrics import Accuracy
from ReprogramLLM.data_loader import wrapper_dataLoader
from transformers import ViTForImageClassification
import torch
from torch.optim import Adam
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from torchsummary import summary
import torch.nn as nn
import logging

import os
sys.path.append("/home/mengjingliu/ADL_unsupervised_learning/")

from utils import plot_loss

# os.chdir("/home/mengjingliu/ADL_unsupervised_learning")
path = "/home/mengjingliu/ADL_unsupervised_learning/ViT/results/"


path_list = ["/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6",
             "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5",
             "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4",
             "/home/mengjingliu/Vid2Doppler/data/2023_11_17/HAR3",
             "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR2"]
train_loader, test_loader = wrapper_dataLoader(path_list, batch_size=16, if_resize=True, if_replicate_channels=True)

model_name_or_path = "google/vit-base-patch16-224-in21k"
title = "finetuneLoRA_" + model_name_or_path.replace('/', '_')
# Load a pre-trained Vision Transformer model
model = ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=5)

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Move model to device
model.to(device)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# LoRA
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Define the optimizer
# optimizer = Adam(model.parameters(), lr=1e-3)
lr_scheduler = ExponentialLR(optimizer, gamma=0.98)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Number of training epochs
num_epochs = 100
test_interval = 10

logging.basicConfig(
    level=logging.DEBUG,  # Minimum level of messages to log
    filename=os.path.join(path, f'log_{title}.log'),  # File where messages will be written
    filemode='a',  # Append mode, so logs are added to the file. Use 'w' for overwrite mode.
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)


def evaluate(model, test_loader, test_accuracies, test_losses, epoch=0, best_accuracy=0):
    # model.eval() does not freeze the model, it only changes behavior of dropout, batchNorm or layers have different behaviors for training and inference.
    model.eval()        
    test_loss = 0.0
    outputs_all = torch.tensor([])
    labels_all = torch.tensor([])
    with torch.no_grad():       # this context is for inference, it can save GPU memory
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)

            outputs = model(inputs).logits.to("cpu")
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            outputs_all = torch.cat([outputs_all, outputs])
            labels_all = torch.cat([labels_all, labels])
    
    test_losses.append(test_loss / len(test_loader))
    
    accuracy = Accuracy(task="multiclass", num_classes=5).to(device)
    accuracy = accuracy(outputs, labels)
    test_accuracies.append(accuracy.item())
    logging.info(f'Epoch {epoch}, Testing Loss: {test_loss / len(test_loader)}, Testing accuracy: {accuracy}')
  
    if accuracy >= best_accuracy:
        logging.info(f"Epoch {epoch}, New best model found and saved. Previous test accuracy: {best_accuracy}, current test accuracy: {accuracy}")

        best_accuracy = accuracy
        # do not save the whole model. should only save trainable parameters
        torch.save(model.state_dict(), os.path.join(path, f'best_model_{title}.pth'))
    
    logging.info("--------------------------------------------------------------")
    return best_accuracy



training_losses = []
test_losses = []
test_accuracies = []
best_accuracy = 0
# Training loop

evaluate(model, test_loader, test_accuracies, test_losses)

for epoch in range(num_epochs):
    model.train()
    for k in range(test_interval):
        running_loss = 0
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            # Move batch to device
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs.logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Update progress bar
            loop.set_description(f'Epoch {epoch * test_interval + k + 1}')
            loop.set_postfix(loss=loss.item())
        training_losses.append(running_loss / len(train_loader))
            
        lr_scheduler.step()
        logging.info(f'Epoch {epoch * test_interval + k + 1}, Training Loss: {running_loss / len(train_loader)}')
    
    best_accuracy = evaluate(model, test_loader, test_accuracies, test_losses, epoch=epoch * test_interval + k + 1, best_accuracy=best_accuracy)
    
    plot_loss(path, f"loss_{title}.png", np.array(training_losses), np.array(test_losses), np.array(test_accuracies), test_interval)
    
