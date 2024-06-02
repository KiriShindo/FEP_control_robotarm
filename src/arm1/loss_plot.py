import numpy as np
import csv
import matplotlib.pyplot as plt



# y_g = np.loadtxt("result_simple/kv60_a0.3_e50/y_g.csv")
# rho_g = np.loadtxt("result_simple/kv60_a0.3_e50/rho_g.csv")

# plt.plot(y_g, label="y_g")
# plt.plot(rho_g, label="rho_g")
# plt.xlabel("time")
# plt.ylabel("MSE")
# plt.legend()
# plt.show()

# print("y_g_mean")
# print(np.mean(y_g[250:750]))
# print("rho_g_mean")
# print(np.mean(rho_g[250:750]))
# print()
# print("y_g_var")
# print(np.max(y_g[250:750]) - np.mean(y_g[250:750]))
# print("rho_g_var")
# print(np.max(rho_g[250:750]) - np.mean(rho_g[250:750]))

train_loss = np.loadtxt("loss/train_loss_1joint.csv")
test_loss = np.loadtxt("loss/test_loss_1joint.csv")

plt.plot(train_loss[1:], label="train loss")
plt.plot(test_loss[1:], label="eval loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
