import cv2
import imutils
import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from PIL import Image
from torch import nn
from data_tool.custom_dataset import CustomImageDataset
from torch.utils.data import DataLoader
import time
# from nets.sota_mobilenet  import MobileNet
from nets.mobilenetv1 import MobileNet
# from nets.mobilenetv2 import MobileNetV2 as MobileNet
# from nets.mobilenetv3 import MobileNetV3_Large as MobileNet
# from nets.shufflenet0 import ShuffleNetV1 as MobileNet
# from nets.shufflenetv2 import ShuffleNetV2 as MobileNet
# from nets.shufflenet0 import ShuffleNetV1 as MobileNet
# from nets.squeezenet import SqueezeNet as MobileNet
# from nets.Xception import xception as MobileNet

device = "cuda"
show=0
test_model=1

model = MobileNet().to(device)
test_model_names=['sota_mobilenet','mobilenetv1','mobilenetv2','mobilenetv3','shufflenetv1','shufflenetv2','SqueezeNet','Xception']
classes = ['轻度裂缝','重度裂缝','点状缺口','块状缺口','表层磨损','边缘磨损']
loss_fn = nn.CrossEntropyLoss()
model.load_state_dict(torch.load(f'results/results_{test_model_names[test_model]}/'+f"model_{test_model_names[test_model]}.pth"))

test_data=CustomImageDataset('mydata/test')
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

def change_cv2_draw(image,strs,local,sizes,colour):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("SIMYOU.TTF",sizes, encoding="utf-8")
    draw.text(local, strs, colour, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image

def mytest(dataloader, model, loss_fn):
    t1=time.time()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    matrix = np.zeros((6, 6))
    with torch.no_grad():
        count,fb_hit=0,0
        total = [0, 0, 0, 0, 0, 0]
        right = [0, 0, 0, 0, 0, 0]
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            fin=pred[0].argmax(0)
            if fin!=0 and y!=0: fb_hit+=1
            total[y] += 1
            right[fin] += (1 if fin==y else 0)
            matrix[fin][y]+=1
            predicted, actual = classes[fin], classes[y]
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            count+=1
            if show:
                # if fin!=y and (sum(total)-sum(right))%10==0:
                if fin+y==7 and fin-y in [1,-1]:
                # if count%50==0:
                # if fin != y:
                    array1 = X[0].cpu().numpy()
                    maxValue = array1.max()
                    array1 = array1 * 255 / maxValue
                    mat = np.uint8(array1)
                    mat = mat.transpose(1, 2, 0)
                    labelleft = "预测:{}  实际:{} ".format(predicted, actual)
                    if fin == y:
                        labelright = '正确'
                    else:
                        labelright = '错误'
                    output = imutils.resize(mat, width=400)
                    output = change_cv2_draw(output, labelleft, (10, 25), 20, (0, 0, 0))
                    if predicted == actual:
                        output = change_cv2_draw(output, labelright, (10, 200), 20, (0, 255, 0))
                    else:
                        output = change_cv2_draw(output, labelright, (10, 200), 20, (255, 0, 0))
                    cv2.imshow("Output", output)
                    cv2.waitKey(0)

    test_loss /= num_batches
    correct /= size
    print(f"test Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    t2=time.time()
    if not show:print('{:.3f}'.format((t2-t1)/989))
    for i in range(6):
        print('{}类总数为{}张，分类的正确率为{:.2f}%'.format(classes[i], total[i], right[i] / total[i] * 100))
    print('测试集总体为{}张，分类的正确率为{:.2f}%(准确率)'.format(np.sum(total), np.sum(right) / np.sum(total) * 100))
    print('有异纤类检出为有异纤的概率为{:.2f}%(检出率)'.format(fb_hit / (np.sum(total)-total[0]) * 100))
    print('无异纤类检出为有异纤的概率为{:.2f}%(虚惊率)'.format(100-right[0] / total[0] * 100))
    for i in range(6):
        for j in range(6):
            if i!=j and matrix[i][j]>0:
                print('{}类错误识别为{}类一共{:.0f}次，占比{:.2f}%'.format(classes[j],classes[i],matrix[i][j],matrix[i][j]/(np.sum(total)-np.sum(right))*100))

mytest(test_dataloader, model, loss_fn)



