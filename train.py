import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from torch import nn
from data_tool.custom_dataset import CustomImageDataset
from torch.utils.data import DataLoader
# from nets.sota_mobilenet import MobileNet
from nets.mobilenetv1 import MobileNet
# from nets.mobilenetv2 import MobileNetV2 as MobileNet
# from nets.mobilenetv3 import MobileNetV3_Large as MobileNet
# from nets.shufflenet0 import ShuffleNetV1 as MobileNet
# from nets.shufflenetv2 import ShuffleNetV2 as MobileNet
# from nets.shufflenet import ShuffleNetG2 as MobileNet
# from nets.squeezenet import SqueezeNet as MobileNet
# from nets.Xception import xception as MobileNet


device = "cuda"

BS=32
epochs = 10
patience_flag=5
train_model=1

patience=0
best_loss=9.0
best_acc=0
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

train_annotations_file='mydata/train_labels.csv'
train_img_dir='mydata/train'
train_model_names=['sota_mobilenet','mobilenetv1','mobilenetv2','mobilenetv3','shufflenetv1','shufflenetv2','SqueezeNet','Xception']

training_data=CustomImageDataset( train_img_dir)
validation_data=CustomImageDataset('mydata/validation')

train_dataloader = DataLoader(training_data, batch_size=BS, shuffle=True)
validate_dataloader = DataLoader(validation_data, batch_size=BS, shuffle=True)

model =MobileNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    current=0
    loss=0
    correct=0
    for batch, (X, y) in enumerate(dataloader):
        if len(X)< BS: break
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), current+ len(X)
        if size-current<BS: current=size
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()/len(X)
        print("\r",f"Epoch {t+1}   "+f"[{current:>5d}/{size:>5d}]  train acc:{(100*correct):.2f}%  train loss:{loss:.4f}  ",end="",flush=True)
    train_loss_list.append(loss)
    train_acc_list.append(correct)

def validate(dataloader, model, loss_fn):
    global best_loss,best_acc,patience
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= num_batches
    val_loss_list.append(val_loss)
    correct /= size
    val_acc_list.append(correct)
    print(f"  val acc:{(100*correct):.2f}%  val loss:{val_loss:.4f}",end='   ')

    if correct>best_acc:
        print('Saving to the best model\n')
        torch.save(model.state_dict(), f'results/results_{train_model_names[train_model]}/'+f'model_{train_model_names[train_model]}.pth')
        best_acc=correct
        patience=0
    else:
        print(f"val acc didn't improve form {best_acc:.4f}\n")
        patience+=1


for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)
    validate(validate_dataloader, model, loss_fn)
    if patience>patience_flag:
        print("early stopping")
        epochs=t+1
        break
print("training finished")

var=np.array(range(1, epochs+1))
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xticks(np.arange(1,epochs+1,round((epochs+1)/10)))
plt.plot(var,np.array(train_loss_list), label="train_loss")
plt.plot( var,np.array(val_loss_list), label="val_loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig(f'results/results_{train_model_names[train_model]}/'+f'loss_{train_model_names[train_model]}.png')
plt.close()

x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xticks(np.arange(1,epochs+1,round((epochs+1)/10)))
plt.plot( var,np.array(train_acc_list), label="train_acc")
plt.plot( var,np.array(val_acc_list), label="val_acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig(f'results/results_{train_model_names[train_model]}/'+f'acc_{train_model_names[train_model]}.png')






