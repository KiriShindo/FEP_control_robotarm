import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import numpy as np
from torch import optim
import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import torch.nn.init as init
from decoder_1joint import RGBDecoder


# dataset
class CustomDataset(Dataset):
    def __init__(self, features, image_paths, transform=None):
        self.features = features
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature = torch.tensor(feature)
        image_path = self.image_paths[idx]
        
        # 画像を読み込み、必要に応じて前処理を適用
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        
        return feature.to(torch.float32), image


# train_data
train_csv_file = 'train_data/train_internal/train.csv'
train_features = np.loadtxt(train_csv_file,       # 読み込みたいファイルのパス
                  delimiter=",",    # ファイルの区切り文字
                  skiprows=0,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  usecols=(0) # 読み込みたい列番号
                 )

train_image_folder = 'train_data/train_images'
train_image_paths = [os.path.join(train_image_folder, f'{filename}.png') for filename in range(1,9001)]


# test_data
test_csv_file = 'test_data/test_internal/test.csv'
test_features = np.loadtxt(test_csv_file,       # 読み込みたいファイルのパス
                  delimiter=",",    # ファイルの区切り文字
                  skiprows=0,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  usecols=(0) # 読み込みたい列番号
                 )

test_image_folder = 'test_data/test_images'
test_image_paths = [os.path.join(test_image_folder, f'{filename}.png') for filename in range(1,1001)]



train_dataset = CustomDataset(train_features, train_image_paths, transform=transforms.ToTensor())
test_dataset = CustomDataset(test_features, test_image_paths, transform=transforms.ToTensor())



# DataLoader
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


n_epoch = 50

# モニタリング用
random_vectors = torch.tensor([-1.19563037475987])

#print(random_vectors.dtype)
fixed = random_vectors.to("cuda:0")



def train_net(net, train_loader, test_loader, n_iter=n_epoch, device="cpu"):
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.95)
    criterion = nn.MSELoss()

    best_test_perf = np.Inf
    epoch_train_loss = []
    epoch_test_loss = []  

    for epoch in range(n_iter):
        net.train()
        batch_train_loss = 0
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            input_x = xx.to(device)
            target_y = yy.to(device)
            #print(input_x.dtype)
            optimizer.zero_grad()
            output_y = net(input_x)
            #print(output_y.shape)
            loss = criterion(output_y, target_y)
            loss.backward()
            optimizer.step()
            batch_train_loss += loss.item()
        scheduler.step()
        epoch_train_loss.append(batch_train_loss / batch_size)
        # Evaluate test set performance
        test_loss = eval_test_set(net, test_loader, device)
        epoch_test_loss.append(test_loss)
        print("%d/%d" % (epoch+1, n_iter))
        if epoch % 10 == 0:
            generated_img = net(fixed)
            save_image(generated_img, "generated_images/{:03d}.png".format(epoch))
            print("train_loss:%f" % epoch_train_loss[epoch])
            print("val_loss:%f" % epoch_test_loss[epoch])

    
    return epoch_train_loss, epoch_test_loss


def eval_test_set(net, data_loader, device="cpu"):
    net.eval()
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            output_variables = net(x)
            criterion = nn.MSELoss()
            loss = criterion(output_variables, y)
            test_loss = loss.item()
            return test_loss


print(torch.cuda.is_available())
#torch.manual_seed(2898479827)
net = RGBDecoder()
net = net.to("cuda:0")

epoch_train_loss, epoch_test_loss = train_net(net, train_dataloader, test_dataloader, n_iter=n_epoch, device="cuda:0")
params = net.state_dict()
torch.save(params, "net.prm", pickle_protocol=4)





x = np.arange(1, n_epoch + 1, 1)
train_loss = np.zeros(n_epoch)
test_loss = np.zeros(n_epoch)
train_loss = epoch_train_loss
test_loss = epoch_test_loss
np.savetxt("loss/train_loss_1joint.csv", train_loss)
np.savetxt("loss/test_loss_1joint.csv", test_loss)
plt.plot(x, epoch_train_loss, label="train")
plt.plot(x, epoch_test_loss, label="test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.legend()
plt.show()


