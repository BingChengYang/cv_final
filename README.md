# cv_final

* 1.Training

## Training model
使用以下指令訓練模型:
```
python train.py --lr=0.00001 --epoches=10 --mini_batch_size=32 --load_model=False --model="model.pkl --img_size=300"
```
lr代表learning rate的大小，default = 0.00001</br>
epoches代表總共訓練幾個epoch，default = 10</br>
mini_batch_size代表會使用mini_batch的大小，default = 32</br>
load_model代表是否要使用訓練到一半的model，False代表重新訓練一個模型，
True則可以選擇要接下去訓練的model，default = False</br>
model代表使用load_model=True時，要接下去的model名稱，default = "model.pkl"，上傳到kaggle Submit</br>
img_size代表要將訓練圖片重新resize的大小，default = 300</br>
