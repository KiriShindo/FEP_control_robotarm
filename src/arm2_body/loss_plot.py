import numpy as np
import csv
import matplotlib.pyplot as plt



train_loss = np.loadtxt("loss/train_loss_e1000.csv")
test_loss = np.loadtxt("loss/test_loss_e1000.csv")

plt.plot(train_loss[100:], label="train loss")
plt.plot(test_loss[100:], label="eval loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
