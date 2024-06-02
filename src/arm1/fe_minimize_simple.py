#!/usr/bin/env python

#from __future__ import division

import numpy as np
#from collections import namedtuple
import cv2
import os

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import mujoco
import mediapy as media
from PIL import Image
import torchvision.transforms as transforms
import cv2
from decoder_1joint import RGBDecoder


import mujoco
import matplotlib.pyplot as plt
import time
import itertools
from typing import Callable, NamedTuple, Optional, Union, List
import mujoco.viewer
import csv
import pandas as pd

import ffmpeg
import imageio

# import random
# import sys

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

targ_num1 = 7000
targ_num2 = 2000



# CSVファイルのパス
csv_file_path = 'train_data/train_internal/train.csv'

# CSVファイルをDataFrameに読み込む
df = pd.read_csv(csv_file_path, header=None)

targ_q1 = df.iloc[targ_num1-1][0]
targ_q2 = df.iloc[targ_num2-1][0]

# 各列の最大値と最小値を取得
max_values = df.max()
min_values = df.min()

print("shape")
print(max_values.shape)

data_max = (np.array(max_values))[0]
data_min = (np.array(min_values))[0]





#Define the joint names and the safe sitting joint states for the safe sitting method:
larm_min = [-1.396]
larm_max = [1.396]

actors = ['joint_1']
actors_ind = [0]

def minmax_normalization(x_input , max_val, min_val):
    # Minmax normalization to get an input x_val in [-1,1] range
    return 2*(x_input-min_val)/(max_val-min_val)-1



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FE_minimize:
    def __init__(self):
        # initialize class variables
        # self.env = env #Mode 0 for simulation, 1 for real NAO.

        # attractor_im and s_v store images from the bottom camera which are processed using the function
        # process_image, the original image size is reduced to the values stored in the properties
        # width and height:
        self.width = 320
        self.height = 240

        self.mu = np.empty((1, 1))
        self.s_v = np.empty((1,3,self.height, self.width)) # Currently observed visual sensation
        self.g_mu = np.empty((1,3,self.height, self.width)) # Predicted visual sensation
        self.pred_error = np.empty((1,3,self.height, self.width))

        # set target
        self.targ_num = targ_num1

        # Only for  active inference
        self.attractor_im = np.empty((1, 3, self.height, self.width))
        #self.attractor_pos = None
        self.attr_error = np.empty((1, 3, self.height, self.width))
        self.a = np.zeros((1, 1))
        self.a_dot = np.zeros((1, 1))
        self.mu_dot = np.zeros((1, 1))
        self.mu_dot_pre = np.zeros((1, 1))
        self.attr_error_sc = np.zeros((1, 1))
        self.vis_error_sc = np.zeros((1, 1))
        self.attr_mse = np.zeros(1)
        self.vis_mse = np.zeros(1)
        self.rhovis_mse = np.zeros(1)
        self.rho_q = np.zeros((1, 1))
        self.mu_q = np.zeros((1, 1))
        self.rho_mu = np.zeros((1, 1))
        

        # Visual forward model:
        self.model_path = 'net.prm'
        # Load network for the visual forward model
        self.load_model()

        # Will be initialized inside the reset_pixelAI function
        self.dt = 0.002
        self.beta = 2 * 1e-2
        self.sigma_v = 3 * 1e-2
        self.sv_mu_gain = 1e-3
        self.sigma_mu = 1
        self.K = 1

        self.a_thres = 2.0
        
        # self.sigma_v_gamma = None

        self.active_inference_mode = 0    # 1 If active inference (actions with attractor dynamics), if 0: perceptual inference
        self.adapt_sigma = 0

    def reset(self, params, start_mu):
        # torch.manual_seed(0)
        # np.random.seed(0)
        self.mu = np.reshape(start_mu, (1,1))
        self.s_v = np.empty((1, 3, self.height, self.width))
        self.g_mu = np.empty((1, 3, self.height, self.width))

        #Read the parameters:
        (self.dt, self.beta, self.sigma_v, self.sv_mu_gain, self.sigma_mu) = params

        self.active_inference_mode = 1
        #self.attractor_pos = np.reshape(attractor_pos, (1,4))
        self.attractor_im = np.empty((1, 3, self.height, self.width))
        self.attr_error = np.empty((1, 3, self.height, self.width))
        self.a = np.zeros((1, 1))
        self.a_dot = np.zeros((1, 1))
        #self.a_thres = 2.0 * (np.pi / 180) / self.dt # Maximum 2 degrees change per timestep

    def set_attractor_im(self, d):
        img_path = "train_data/train_images/%d.png" % self.targ_num
        # if targ_num > 4500:
        #     n = 4500 - int(d.qpos[1] * 243)
        #     if n <= 4500:
        #         n = 4501
        # else:
        #     n = 4500 + int(d.qpos[1] * 243)
        #     if n > 4500:
        #         n = 4499
        # print(n)
        #img_path = "train_data/train_images/%d.png" % n
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)/255
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # (240, 320, 3) => (3, 240, 320)
        img = np.transpose(img, (2, 0, 1))
        # (3, 240, 320) => (1, 3, 240, 320)
        self.attractor_im = img[np.newaxis,:,:,:]
        # camera_im = process_image(attractor_im, self.env)
        # self.attractor_im = camera_im.reshape([1,1,camera_im.shape[0],camera_im.shape[1]])

    def set_visual_sensation(self, r, d):
        # m = mujoco.MjModel.from_xml_string(xml)
        # d = mujoco.MjData(m)
        # r = mujoco.Renderer(m)
        r.update_scene(d, camera="track")
        img_path = "sim/sim.png"
        media.write_image(img_path, r.render())
        # resize
        # img = Image.open(img_path)
        # img = img.resize((80, 60))
        # img.save(img_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)/255
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # (240, 320, 3) => (3, 240, 320)
        img = np.transpose(img, (2, 0, 1))
        # (3, 240, 320) => (1, 3, 240, 320)
        self.s_v = img[np.newaxis,:,:,:]

    def load_model(self):
        print("Loading Deconvolutional Network...")
        self.net = RGBDecoder()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = self.net.to(device)

        params = torch.load(self.model_path, map_location="cpu")
        self.net.load_state_dict(params)
        self.net.eval()
        # self.network.load_state_dict(torch.load(os.path.join(self.model_path,'checkpoint_cpu_version')))
        #self.net.eval()

        #Load data range used for training for minmax-normalization:
        # data_range = np.genfromtxt(os.path.join(self.model_path,"data_range.csv"), delimiter=",")
        # self.data_max = data_range[0,:]
        # self.data_min = data_range[1,:]



    def visual_forward(self):
        #print("A")
        input = torch.FloatTensor(minmax_normalization(self.mu, data_max, data_min)).to(device)
        #print("B")
        #input = torch.FloatTensor(self.mu).to(device)
        input = Variable(input, requires_grad=True)
        # prediction
        out = self.net.forward(input)
        return input, out 

    def get_dF_dmu_vis(self, input, out):
        neg_dF_dg = (1/self.sigma_v) * self.pred_error
        #print(neg_dF_dg.shape)
        # Set the gradient to zero before the backward pass to make sure there is no accumulation from previous backward passes
        # input.grad = torch.zeros(input.size())
        if input.grad is not None:
            input.grad = torch.zeros(input.size(), dtype=input.grad.dtype, device=input.grad.device)
        else:
            # もしくは何らかのデフォルトのデータ型とデバイスを設定する
            input.grad = torch.zeros(input.size(), dtype=torch.float32, device='cuda:0')
        #print(input.grad.shape)
        out.backward(torch.Tensor(neg_dF_dg).to(device),retain_graph=True)
        #print(input.grad)
        return input.grad.cpu().data.numpy() # dF_dmu_vis

    def get_dF_dmu_dyn(self, input, out):
        self.attr_error = self.attractor_im - self.g_mu
        A = self.beta * (self.attr_error).copy()
        # Set the gradient to zero before the backward pass to make sure there is no accumulation from previous backward passes
        # input.grad = torch.zeros(input.size())
        # print(input.grad)
        input.grad = torch.zeros(input.size(), dtype=input.grad.dtype, device=input.grad.device)
        out.backward(torch.Tensor(A*(1/self.sigma_mu)).to(device),retain_graph=True)
        return input.grad.cpu().data.numpy() # mu_dot_dyn (1x1 vector)
    
    # def get_internal_error(self, input, out):
    #     self.attr_error = self.attractor_im - self.g_mu
    #     input.grad = torch.zeros(input.size(), dtype=input.grad.dtype, device=input.grad.device)
    #     out.backward(torch.Tensor(self.beta * self.attr_error).to(device),retain_graph=True)
    #     #print(input.grad)
    #     return input.grad.cpu().data.numpy() # mu_dot_dyn (1x1 vector)

    def get_dF_da_visual(self, dF_dmu_vis):
        # dF_dsv = (1 / self.sigma_v) * self.pred_error (= neg_dF_dg)
        return (-1) * dF_dmu_vis * self.dt

    def iter(self):
        input, out = self.visual_forward()
        # print(out.shape)
        # print(out)
        # print(input.dtype)
        # print(type(input))
        # print(type(input.grad))
        self.g_mu = out.cpu().data.numpy()
        #self.g_mu = (self.g_mu * 255).astype(np.uint8)
        # print((self.g_mu).shape)
        # print(self.g_mu)
        # g_mu = (self.g_mu).copy()
        # g_mu = np.squeeze(g_mu)
        # print(g_mu)
        # g_mu = np.transpose(g_mu, (1, 2, 0))
        # cv2.imwrite("img_log/test.png", g_mu)
        self.pred_error = self.s_v - self.g_mu

        self.attr_mse = np.mean(((self.attr_error).copy())**2)
        self.vis_mse = np.mean(((self.pred_error).copy())**2)
        self.rhovis_mse = np.mean(((self.s_v).copy() - (self.attractor_im).copy())**2)
        


        # dF/dmu using visual information:
        dF_dmu_vis = self.get_dF_dmu_vis(input, out)
        mu_dot_vis = self.sv_mu_gain * dF_dmu_vis    # mu_dot_vis is a 1x4 vector
        self.vis_error_sc = mu_dot_vis.copy()

        # get error between desired degree and current degree(internal)
        # self.internal_deg_err = self.get_internal_error(input, out)


        # dF/dmu with attractor:
        #self.mu_dot_pre = self.mu_dot
        #self.attr_error_sc = (self.get_dF_dmu_dyn(input, out)).copy() - self.mu_dot_pre/self.sigma_mu
        self.attr_error_sc = (self.get_dF_dmu_dyn(input, out)).copy()
        #mu_dot = self.mu_dot_pre + mu_dot_vis + self.attr_error_sc
        mu_dot = mu_dot_vis + self.attr_error_sc
        self.mu_dot = mu_dot.copy()
        
        # Compute the action:
        self.a_dot = self.get_dF_da_visual(dF_dmu_vis)
        self.a_dot = self.K * self.a_dot

        # Update mu:
        self.mu = self.mu + self.dt * mu_dot
        #self.mu = minmax_normalization(self.mu, data_max, data_min)
        #self.mu[0] = [-0.000000761, -1.499666667, -0.00032278, 0.006600549, -0.000220694, -0.000189881, 0.00000136]
        #self.mu = self.mu * 10
        #self.mu = np.clip(self.mu, -2, 2)

        # Update a:
        self.a = self.a + self.dt * self.a_dot
        # Clip the action value in the case of action saturation:
        self.a = np.clip(self.a, -self.a_thres, self.a_thres)

        # if self.adapt_sigma and np.square(self.pred_error).mean() <= 0.01:
        #     self.sigma_v = self.sigma_v * self.sigma_v_gamma
        #     self.adapt_sigma = 0 # Increase Sigma only once!


def move(fe_min, d):
    q = (d.qpos).copy()
    diff_thres = 0.0
    # Actions will only be executed if they are larger than the threshold value!

    diff = fe_min.a[0][0]*fe_min.dt
    # targ = q[i]+ diff
    # d.qpos[i] = targ
    # if abs(diff)>=diff_thres:
    targ = q[0]+ diff
    print("diff")
    print(diff)
    print("targ")
    print(targ)
    if targ<larm_max[0] and targ>larm_min[0]:
        d.ctrl[0] = targ
        #d.ctrl[1] = fe_min.a[0][0]
        #d.ctrl[1] = 0.3
        #d.qvel[1] = fe_min.a[0][0]
        #print(d.qpos[1])
        #print("\n")
        #print("Action has been carried out")
    else:
        targ=np.minimum(larm_max[0],np.maximum(targ, larm_min[0]))
        if abs(targ-q[0])>=diff_thres:
            d.ctrl[0] = targ
            #d.ctrl[1] = 0.3
            #d.ctrl[1] = fe_min.a[0][0]
            #d.qvel[1] = fe_min.a[0][0]
        print('Joints limit reached in: {}'.format(actors[0]))
#print('Last target {}'.format(self.last_target))



def central_execute(fe_min, d, r):



    # self.reset_robot(level_id, c,t)

    #Bring the robot to the attractor position to store attractor image:
    # self.set_arm_angles(np.squeeze(self.pixelAI.attractor_pos))
    # time.sleep(1.0)

    fe_min.set_attractor_im(d)
    #Bring the robot arm to the mu position:
    #self.set_arm_angles(np.squeeze(self.pixelAI.mu))


    fe_min.set_visual_sensation(r, d)

    fe_min.iter()
    move(fe_min, d)




xml = """
<mujoco model="inverted pendulum">
	<compiler inertiafromgeom="true" autolimits="true"/>
    <option gravity="0 0 0" />
	<default>
		<joint armature="0" damping="1" limited="true"/>
		<geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
		<tendon/>
		<motor ctrlrange="-1.396 1.396"/>
	</default>
	<option integrator="RK4" timestep="0.002"/>
	<worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <camera name="track" mode="fixed" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
        <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="1.0 1.0 1.0 1" size="40 40 40" type="plane"/>
        <body name="pole" pos="0 0 0">
            <joint axis="0 1 0" name="joint1" pos="0 0 0" range="-80 80" type="hinge"/>
            <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
            <!--                 <body name="pole2" pos="0.001 0 0.6"><joint name="hinge2" type="hinge" pos="0 0 0" axis="0 1 0"/><geom name="cpole2" type="capsule" fromto="0 0 0 0 0 0.6" size="0.05 0.3" rgba="0.7 0 0.7 1"/><site name="tip2" pos="0 0 .6"/></body>-->
        </body>
	</worldbody>
    <actuator>
        <position name="pos_servo" joint="joint1" kp="1500000"/>
        <!--velocity name="vel_servo" joint="joint1" kv="300"/-->
    </actuator>
</mujoco>
"""


m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m)
#print(len(d.ctrl))

# initialize
d.qpos = [0.0]

log_a = [[] for i in range(1)]
log_q = [[] for i in range(1)]
log_attr_err = [[] for i in range(1)]
log_vis_err = [[] for i in range(1)]
log_adot = [[] for i in range(1)]
log_mu = [[] for i in range(1)]
log_mu_dot = [[] for i in range(1)]
log_vis_mse = []
log_attr_mse = []
log_rhovis_mse = []
log_rho_q = []
log_rho_mu = []
log_mu_q = []
#log_internal_deg_err = [[] for i in range(1)]




# set initial position
#d.qpos[0] = 35
#print(len(d.qpos))

fe_min = FE_minimize()

frames = []


with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  fe_min.mu[0] = (d.qpos[0]).copy()
  n = 0
  sim_len = 3000
  time_len = 120
  flg = 0
  while viewer.is_running() and time.time() - start < sim_len:
    if time.time() - start > time_len/2:
        fe_min.targ_num = targ_num2
        flg = 1
    step_start = time.time()
    #d.qpos[0] = -0.000000761
    #d.qpos[2:] = [-0.00032278, 0.006600549, -0.000220694, -0.000189881, 0.00000136]

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    central_execute(fe_min, d, r)
    if n == 0:
        start = time.time()
    print(d.qpos[0])
    mujoco.mj_step(m, d)
    print(d.qpos[0])
    print("\n")
    r.update_scene(d, camera="track")
    pixels = r.render()
    frames.append(pixels)

    #print(fe_min.a[0][7])

    for i in range(1):
        log_a[i].append(fe_min.a[0][i])

    # for i in range(1):
    #     log_internal_deg_err[i].append(fe_min.internal_deg_err[0][i])


    for i in range(1):
        log_q[i].append(d.qpos[i])

    for i in range(1):
        log_attr_err[i].append(fe_min.attr_error_sc[0][i])

    for i in range(1):
        log_vis_err[i].append(fe_min.vis_error_sc[0][i])

    for i in range(1):
        log_adot[i].append(fe_min.a_dot[0][i])

    for i in range(1):
        log_mu[i].append(fe_min.mu[0][i])
    
    for i in range(1):
        log_mu_dot[i].append(fe_min.mu_dot[0][i])
    
    log_vis_mse.append(fe_min.vis_mse)
    log_attr_mse.append(fe_min.attr_mse)
    log_rhovis_mse.append(fe_min.rhovis_mse)

    for i in range(1):
        if flg == 0:
            log_rho_q.append((targ_q1 - d.qpos[i])**2)
        else:
            log_rho_q.append((targ_q2 - d.qpos[i])**2)
    
    for i in range(1):
        if flg == 0:
            log_rho_mu.append((targ_q1 - fe_min.mu[0][i])**2)
        else:
            log_rho_mu.append((targ_q2 - fe_min.mu[0][i])**2)
    
    for i in range(1):
        log_mu_q.append((fe_min.mu[0][i] - d.qpos[i])**2)

    # r.update_scene(d, camera="track")
    n += 1

    if time.time() - start > time_len:
        mu = fe_min.mu[0]
        q = d.qpos[0]
        break

   

    # Example modification of a viewer option: toggle contact points every two seconds.
    # with viewer.lock():
    #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)


#time_len = time.time() - start
t = np.linspace(0, time_len, len(log_mu[0]))
print(len(t))




# log_q = np.array(log_q)
# log_q_ori = log_q.copy()
# log_q *= 100


# サブプロットの行数と列数
rows = 3
cols = 3

name_list = ['joint_1']


# a_dotのログ
#fig = plt.figure(figsize=(12, 12))
#fig_a_dot = plt.figure(figsize=(12, 12))  # フィギュアのサイズを調整

# for i in range(1, 2):
#     ax = fig.add_subplot(3, 3, i)
#     ax.set_title(name_list[i-1])
#     ax.plot(log_adot[i-1], label="a_dot")


# # レイアウトの調整
# plt.tight_layout()
# plt.legend()
plt.rcParams["font.size"] = 30
plt.plot(t, log_adot[0], label="a_dot")
plt.xlabel("time [sec]", fontsize=30)

# グラフを表示
plt.legend(fontsize=15)
plt.show()


# mu_dot
plt.plot(t, log_mu_dot[0], label="mu_dot")
plt.xlabel("time [sec]", fontsize=30)

# グラフを表示
plt.legend(fontsize=15)
plt.show()


# internal deg err
# plt.plot(t, log_internal_deg_err[0], label="internal degree error")
# plt.xlabel("time [sec]")

# グラフを表示
# plt.legend()
# plt.show()


# muのログ
# fig = plt.figure(figsize=(12, 12))
# #fig_a_dot = plt.figure(figsize=(12, 12))  # フィギュアのサイズを調整

# for i in range(1, 2):
#     ax = fig.add_subplot(3, 3, i)
#     ax.set_title(name_list[i-1])
#     ax.plot(log_mu[i-1], label="mu")
#     ax.plot(log_q[i-1], label="q")


# レイアウトの調整
# plt.tight_layout()
# plt.legend()
# plt.plot(t, log_mu[0], label=r"$\mu$ : internal [rad]")
# plt.plot(t, log_q[0], label=r"$q$ : actual [rad]")
# plt.plot(t, log_a[0], label=r"$a$ : action [rad/sec]")
# plt.hlines(targ_q1, xmin=0, xmax=time_len, linestyles="dashed", colors="red")
# plt.hlines(targ_q2, xmin=0, xmax=time_len, linestyles="dashed", colors="red")
# plt.axhspan(larm_min[0], larm_max[0], facecolor="blue", alpha=0.1, label='arm range')
# plt.text(time_len+0.02*time_len, targ_q1, 'target1', va='center', ha='left', backgroundcolor='white')
# plt.text(time_len+0.02*time_len, targ_q2, 'target2', va='center', ha='left', backgroundcolor='white')
# plt.xlabel("time [sec]")

# # グラフを表示
# plt.xlim(0, time_len)
# #plt.ylim(-1.5, 1.5)
# plt.legend(loc="upper right")
# plt.show()

plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["ytick.major.width"] = 2  
# 1つ目のy軸にデータをプロット
fig, ax1 = plt.subplots()
#ax1.plot(t, log_mu[0], label=r"$\mu$ : internal [rad]")
ax1.plot(t, log_q[0], label=r"$\theta$ : actual [rad]", color="tab:orange", linewidth=3)
ax1.set_xlabel("time [sec]", fontsize=30)
ax1.set_ylabel(r"$\theta$ [rad]", color='black', fontsize=30)
ax1.hlines(targ_q1, xmin=0, xmax=time_len, linestyles="dashed", colors="red", label="target1 [rad]")
ax1.hlines(targ_q2, xmin=0, xmax=time_len, linestyles="dashed", colors="green", label="target2 [rad]")
#ax1.vlines(time_len/2, ymin=-1.5, ymax=1.5, linestyles="dashed", colors="red")
#ax1.annotate('Target Changed', xy=(time_len/2, -1.5), xytext=(time_len/2, -2.0),
             #arrowprops=dict(facecolor='black', arrowstyle='->'))
# ax1.annotate('', xy=(time_len/2, -2.0), xytext=(time_len/2, -1.8),
#              arrowprops=dict(facecolor='black', arrowstyle='->'))
ax1.axhspan(larm_min[0], larm_max[0], facecolor="blue", alpha=0.1, label='arm range [rad]')
# plt.text(time_len+0.02*time_len, targ_q1, 'target1', va='center', ha='left', backgroundcolor='white')
# plt.text(time_len+0.02*time_len, targ_q2, 'target2', va='center', ha='left', backgroundcolor='white')
#plt.text(time_len / 2, 1.7, 'target changed', va='center', ha='center', backgroundcolor='white', color='red')
ax1.legend(loc="lower left", fontsize=22)

# 2つ目のy軸を作成
# ax2 = ax1.twinx()
# ax2.plot(t, log_a[0], label=r"$a$ : action [rad/sec]", color='green', alpha=0.5)
# ax2.set_ylabel(r"$a$ [rad/sec]", color='black', fontsize=30)
# ax2.legend(loc="upper right", fontsize=15)

# グラフの範囲と凡例
plt.xlim(0, time_len)
plt.show()


# q, mu, rho
plt.plot(t, log_mu_q, label=r"$q-\mu$")
plt.plot(t, log_rho_mu, label=r"$\rho-\mu$")
plt.plot(t, log_rho_q, label=r"$\rho-q$")
plt.xlabel("time [sec]", fontsize=30)
plt.ylabel(r"$angle err [rad^{2}]$", fontsize=30)
# グラフを表示
plt.legend(fontsize=15)
plt.show()

# qとaのログ
plt.plot(t, log_a[0], color="blue", label="a")
plt.plot(t, log_q[0], color="red", label="q")
plt.xlabel("time [sec]", fontsize=30)

# グラフを表示
plt.legend(fontsize=22)
plt.show()



# attrのログ
# fig = plt.figure(figsize=(12, 12))
# #fig_a_dot = plt.figure(figsize=(12, 12))  # フィギュアのサイズを調整

# for i in range(1, 2):
#     ax = fig.add_subplot(3, 3, i)
#     ax.set_title(name_list[i-1])
#     ax.plot(log_attr[i-1], label="attr")


# レイアウトの調整
# plt.tight_layout()
# plt.legend()

# 移動平均と移動標準偏差を計算する関数
def moving_average_std(data, window_size):
    moving_averages = []
    moving_stds = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        moving_averages.append(np.mean(window))
        moving_stds.append(np.std(window))
    return moving_averages, moving_stds

#attr_err
plt.plot(t, log_attr_err[0], label="attr_err")
plt.xlabel("time [sec]", fontsize=30)
# グラフを表示
plt.legend(fontsize=15)
plt.show()

# 10データごとの移動平均と標準偏差を計算
ma, std = moving_average_std(log_attr_err[0], 10)

# 時系列グラフをプロット
#plt.plot(log_attr_err[0], label='Original Data')
plt.plot(range(len(log_attr_err[0]) - 10 + 1), ma, label='Moving Average')
plt.fill_between(range(len(log_attr_err[0]) - 10 + 1), np.array(ma) - np.array(std), np.array(ma) + np.array(std), color='b', alpha=0.2)
plt.legend()
plt.show()



#vis_err
plt.plot(t, log_vis_err[0], label="vis_err")
plt.xlabel("time [sec]", fontsize=30)
# グラフを表示
plt.legend(fontsize=15)
plt.show()


# 10データごとの移動平均と標準偏差を計算
ma, std = moving_average_std(log_vis_err[0], 10)

# 時系列グラフをプロット
#plt.plot(log_vis_err[0], label='Original Data')
plt.plot(range(len(log_vis_err[0]) - 10 + 1), ma, label='Moving Average')
plt.fill_between(range(len(log_vis_err[0]) - 10 + 1), np.array(ma) - np.array(std), np.array(ma) + np.array(std), color='b', alpha=0.2)
plt.legend()
plt.show()


# MSE
np.savetxt("result_simple/kv60_a0.3_e50/y_g.csv", log_vis_mse)
np.savetxt("result_simple/kv60_a0.3_e50/rho_g.csv", log_attr_mse)
plt.plot(t, log_vis_mse, label=r"$y_{v}-g(\mu)$")
plt.plot(t, log_attr_mse, label=r"$\rho-g(\mu)$")
plt.plot(t, log_rhovis_mse, label=r"$\rho-y_{v}$")
plt.xlabel("time [sec]", fontsize=30)
plt.ylabel("MSE", fontsize=30)
# グラフを表示
plt.legend(fontsize=15)
plt.show()

attr_err = fe_min.attr_error
attr_im = fe_min.attractor_im
g_mu = fe_min.g_mu
s_v = fe_min.s_v
pred_err = fe_min.pred_error
# g_mu = (g_mu * 255).astype(np.uint8)
#g_mu = cv2.cvtColor(g_mu, cv2.COLOR_BGR2RGB)

attr_err = np.squeeze(attr_err)
attr_im = np.squeeze(attr_im)
g_mu = np.squeeze(g_mu)
s_v = np.squeeze(s_v)
pred_err = np.squeeze(pred_err)

attr_err = np.transpose(attr_err, (1, 2, 0))
attr_im = np.transpose(attr_im, (1, 2, 0))
g_mu = np.transpose(g_mu, (1, 2, 0))
s_v = np.transpose(s_v, (1, 2, 0))
pred_err = np.transpose(pred_err, (1, 2, 0))

plt.imshow(attr_err)
plt.show()
plt.imshow(attr_im)
plt.show()
plt.imshow(g_mu)
plt.show()
plt.imshow(s_v)
plt.show()
plt.imshow(pred_err)
plt.show()

# attr_err = cv2.cvtColor(attr_err, cv2.COLOR_RGB2BGR)
# attr_im = cv2.cvtColor(attr_im, cv2.COLOR_RGB2BGR)
# g_mu = cv2.cvtColor(g_mu, cv2.COLOR_RGB2BGR)
# s_v = cv2.cvtColor(s_v, cv2.COLOR_RGB2BGR)
# pred_err = cv2.cvtColor(pred_err, cv2.COLOR_RGB2BGR)

# print(np.min(attr_err), np.max(attr_err))
# print(np.min(attr_im), np.max(attr_im))
# print(np.min(g_mu), np.max(g_mu))
# print(np.min(s_v), np.max(s_v))
# print(np.min(pred_err), np.max(pred_err))

# cv2.imwrite("img_log/attr_err.png", attr_err)
# cv2.imwrite("img_log/attr_im.png", attr_im)
# cv2.imwrite("img_log/g_mu.png", g_mu)
# cv2.imwrite("img_log/s_v.png", s_v)
# cv2.imwrite("img_log/pred_err.png", pred_err)

slice_start = int(0.8 * len(log_mu[0]))
print("mu_min")
print(np.min(log_mu[0][slice_start:]))
print("mu_max")
print(np.max(log_mu[0][slice_start:]))
print("mu")
print(mu)
print("q")
print(q)
print("last target")
print(targ_q2)

# save movie
media.write_video("movie.mp4", frames, fps=120)

# # 動画ファイルのパスを指定
# video_path = 'output_video.mp4'

# # imageioのwriterを使用して動画ファイルを作成
# with imageio.get_writer(video_path, fps=120) as writer:  # fpsはフレームレート
#     for frame in frames:
#         writer.append_data(frame)