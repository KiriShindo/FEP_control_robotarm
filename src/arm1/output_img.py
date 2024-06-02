import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch import optim
import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import random
from torch.autograd import Variable
import torch.nn.init as init
from decoder_1joint import RGBDecoder



# dataset
class CustomDataset2(Dataset):
    def __init__(self, features, image_paths, transform=None):
        self.features = features
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self):
        feature = self.features[:]
        feature = torch.tensor(feature)
        image_path = self.image_paths[:]
        
        # 画像を読み込み、必要に応じて前処理を適用
        image = []
        for i in range(len(feature)):
            img = Image.open(image_path[i])
            img = self.transform(img)
            image.append(img)
        
        return feature.to(torch.float32), image
    



# test_data
test_csv_file = 'test_data/test_internal/test.csv'
test_features = np.loadtxt(test_csv_file,       # 読み込みたいファイルのパス
                  delimiter=",",    # ファイルの区切り文字
                  skiprows=0,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                  usecols=(0) # 読み込みたい列番号
                 )

test_image_folder = 'test_data/test_images'
test_image_paths = [os.path.join(test_image_folder, f'{filename}.png') for filename in range(1,1001)]
print(test_image_paths[:1000])

test_dataset = CustomDataset2(test_features, test_image_paths, transform=transforms.ToTensor())
#print(len(test_dataset.__getitem__(20)))
#print(test_dataset.__getitem__(20)[0])

input_data = test_dataset.__getitem__()[0]
target_data = test_dataset.__getitem__()[1]
input_data = input_data[500:1000]
target_data = target_data[500:1000]


target_data = torch.stack(target_data, dim=0)
print(target_data.shape)



def eval_test_set(net, input_data, target_data, device="cpu"):
    net.eval()
    with torch.no_grad():
        x, y = input_data.to(device), target_data.to(device)
        output_variables = net(x)
        criterion = nn.MSELoss()
        loss = criterion(output_variables, y)
        test_loss = loss.item()
    return test_loss, output_variables.to("cpu")


net = RGBDecoder()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

params = torch.load("net.prm", map_location="cpu")
net.load_state_dict(params)


loss_val, output_variables = eval_test_set(net, input_data, target_data, device="cuda:0")
print('====>Loss over the data: {:.4f}'.format(loss_val))

print('---> Visualizing 20 network outputs (randomly sampled from the evaluation data)...')
output_images = output_variables.data.numpy()
rand_ind = random.sample(range(input_data.shape[0]), 10)
for i in rand_ind:
    # サイズが1の次元を削除
    im_true = np.squeeze(target_data[i,:,:,:])
    im_pred = np.squeeze(output_images[i, :, :, :])
    print(im_true.shape)

    # (c, h, w) => (h, w, c)
    im_true = np.transpose(im_true, (1, 2, 0))
    im_pred = np.transpose(im_pred, (1, 2, 0))
    print(im_true.shape)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(im_true, vmin=0, vmax=1)
    axarr[0].set_title('True image')
    axarr[0].axis('off')

    axarr[1].imshow(im_pred, vmin=0, vmax=1)
    axarr[1].set_title('Predicted image')
    axarr[1].axis('off')
    plt.show()
