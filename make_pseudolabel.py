import os
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.utils.data as Data
import torchvision.transforms as transform
import torch
import torch.nn as nn
import numpy as np
import csv
import torchvision.models as models
import argparse

import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--test_model", default="model.pkl")
file_dir = os.getcwd()
args = parser.parse_args()

# set the testing data transform
img_transform = transform.Compose([
    transform.Resize((200,200)),
    transform.ToTensor(),
    transform.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
])

# load the testing model and data
network = torch.load('./model/'+args.test_model)
print(network)
test_data = load_data.Load_extradata(transform=img_transform)
test_load = Data.DataLoader(dataset=test_data, batch_size=16, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network.to(device)
print(device)

# start testing the model
def test_model():
    ans = []
    ids = []
    network.eval()
    with torch.no_grad():
        for data in test_load:
            x, y = data
            x = x.to(device)
            outputs = network(x)
            predict = torch.max(outputs.data, 1)[1]
            ids.extend(y)
            ans.extend(predict.cpu().numpy())
    return ans, ids

# write the result into csv file
with open('./data/pseudo.csv', 'w', newline='') as csvFile:
    ans, ids = test_model()
    writer = csv.writer(csvFile)
    writer.writerow(['id_code', 'diagnosis'])
    for i in range(len(ans)):
        writer.writerow([ids[i], ans[i]])