import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transform
import torchvision.models as models
import argparse
from efficientnet_pytorch import EfficientNet
from torch.nn.parameter import Parameter
import torch.nn.functional as F
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class MixNet(nn.Module):
    def __init__(self, b=3):
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b{}'.format(b))
        self.in_features = self.base._fc.in_features
        self.fc1 = nn.Linear(self.in_features, 1, bias=False)
        self.fc2 = nn.Linear(self.in_features, 5, bias=False)
        self.avg_pool = GeM()
        
    def forward(self, x):
        x = self.base.extract_features(x)
        feat = self.avg_pool(x)
        feat = feat.view(x.size(0), -1)
        rg_output = self.fc1(feat)
        cls_output = self.fc2(feat)
        return rg_output, cls_output

def rg2predict(rg_output):
    # print(rg_output)
    pred = []
    threshold = [0.5, 1.5, 2.5, 3.5]
    for i in range(len(rg_output)):
        if rg_output[i] < threshold[0]:
            pred.append(0)
        elif rg_output[i] >= threshold[0] and rg_output[i] < threshold[1]:
            pred.append(1)
        elif rg_output[i] >= threshold[1] and rg_output[i] < threshold[2]:
            pred.append(2)
        elif rg_output[i] >= threshold[2] and rg_output[i] < threshold[3]:
            pred.append(3)
        else:
            pred.append(4)

    return pred

# my own library
import load_data
def compare_ans(l1, l2):
    cnt = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            cnt += 1
    return cnt

# test model
def test_model(load):
    rg_correct = 0
    cls_correct = 0
    total = 0
    # turn the model into evaluate mode
    network.eval()
    with torch.no_grad():
        for data in load:
            x, y = data
            x,y = x.to(device), y.to(device)
            rg_outputs, cls_outputs = network(x)
            cls_predict = torch.max(cls_outputs.data, 1)[1]
            rg_predict = rg2predict(rg_outputs)
            total += y.size(0)
            rg_correct += compare_ans(y, rg_predict)
            cls_correct += (cls_predict==y).sum().item()
    print("rg accuracy : {:f} %".format(float(rg_correct)/float(total)*100.0))
    print("cls accuracy : {:f} %".format(float(cls_correct)/float(total)*100.0))


# setting hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="model.pkl")
parser.add_argument("--lr", default=0.000001)
parser.add_argument("--epoches", default=5)
parser.add_argument("--mini_batch_size", default=32
)
parser.add_argument("--load_model", default=False)
parser.add_argument("--img_size", default=220)
args = parser.parse_args()

# set the training data transform 
train_img_transform = transform.Compose([
    transform.Resize(args.img_size),
    transform.RandomHorizontalFlip(),
    transform.RandomVerticalFlip(),
    transform.RandomRotation(0, 360),
    transform.ToTensor(),
    transform.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
])

# set the validation data transform
valid_img_transform = transform.Compose([
    transform.Resize((args.img_size, args.img_size)),
    transform.ToTensor(),
    transform.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
])

# load the data and the model
if args.load_model == True:
    network = torch.load('./model/'+args.model)
else:
    network = MixNet()

print(network)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
network.to(device)
train_set = load_data.Load_traindata(transform=train_img_transform, use_pseudo=False)
val_set = load_data.Load_traindata(transform=valid_img_transform, valid=True)
train_load = Data.DataLoader(dataset=train_set, batch_size=args.mini_batch_size, shuffle=True)
val_load = Data.DataLoader(dataset=val_set, batch_size=args.mini_batch_size, shuffle=True)

# set the loss function and the optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
cls_loss_func = nn.CrossEntropyLoss()
rg_loss_func = nn.MSELoss()

# Start training model
for epoch in range(args.epoches):
    print(epoch)
    network.train()
    for step, (x, y) in enumerate(train_load):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        rg_output, cls_output = network(x)
        
        rg_loss = rg_loss_func(rg_output.reshape(y.size()), y.float())
        cls_loss = cls_loss_func(cls_output, y)

        loss = rg_loss + 0.2*cls_loss
        loss.backward()
        optimizer.step()
    scheduler.step()
    print("rg loss:{}".format(rg_loss))
    print("cls loss:{}".format(cls_loss))
    print("total loss:{}".format(loss))
    test_model(val_load)
    # torch.save(network, './model/model'+str(val_acc)+'.pkl')
    if (epoch % 5) == 0:
        torch.save(network, './model/model'+str(epoch)+'.pkl')    
        # test_model(train_load)
# save model
torch.save(network, './model/model.pkl')