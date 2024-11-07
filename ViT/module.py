
import logging
import torch
from torchmetrics import Accuracy
import os


def cnt_trainable_params(model):
    cnt = 0
    for param in model.parameters():
        if param.requires_grad:
            shape = param.shape
            n = 1
            for s in shape:
                n *= s
            cnt += n
    
    print(f"Trainable parameters: {cnt}")


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
