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
from decoder_2joint import RGBDecoder


import mujoco
import matplotlib.pyplot as plt
import time
import itertools
from typing import Callable, NamedTuple, Optional, Union, List
import mujoco.viewer
import csv
import pandas as pd

import ffmpeg
import matplotlib.animation as animation
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

# import random
# import sys

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

targ_num2 = 1
targ_num1 = 5025


# CSVファイルのパス
csv_file_path = 'train_data/train_internal/train.csv'

# CSVファイルをDataFrameに読み込む
df = pd.read_csv(csv_file_path, header=None)

targ1_q1 = df.iloc[targ_num1-1][0]
targ1_q2 = df.iloc[targ_num1-1][1]
targ_q1 = [targ1_q1, targ1_q2]

targ2_q1 = df.iloc[targ_num2-1][0]
targ2_q2 = df.iloc[targ_num2-1][1]
targ_q2 = [targ2_q1, targ2_q2]

# 各列の最大値と最小値を取得
max_values = df.max()
min_values = df.min()

print("shape")
print(max_values.shape)


data_max = (np.array(max_values))
data_min = (np.array(min_values))






#Define the joint names and the safe sitting joint states for the safe sitting method:
larm_min = [-0.698, -0.698]
larm_max = [0.698, 0.698]

actors = ['joint_1', 'joint_2']
actors_ind = [0, 1]

def minmax_normalization(x_input , max_val, min_val):
    # Minmax normalization to get an input x_val in [-1,1] range
    return 2.0*(x_input-min_val)/(max_val-min_val)-1.0



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

        self.mu = np.empty((1, 2))
        self.s_v = np.empty((1,3,self.height, self.width)) # Currently observed visual sensation
        self.g_mu = np.empty((1,3,self.height, self.width)) # Predicted visual sensation
        self.pred_error = np.empty((1,3,self.height, self.width))

        # set target
        self.targ_num = targ_num1

        # Only for  active inference
        self.attractor_im = np.empty((1, 3, self.height, self.width))
        #self.attractor_pos = None
        self.attr_error = np.empty((1, 3, self.height, self.width))
        self.a = np.zeros((1, 2))
        self.a_dot = np.zeros((1, 2))
        self.mu_dot = np.zeros((1, 2))
        self.attr_error_sc = np.zeros((1, 2))
        self.vis_error_sc = np.zeros((1, 2))
        self.attr_mse = np.zeros(1)
        self.vis_mse = np.zeros(1)
        self.rhovis_mse = np.zeros(1)

        

        # Visual forward model:
        self.model_path = 'net_1000.prm'
        # Load network for the visual forward model
        self.load_model()

        # Will be initialized inside the reset_pixelAI function
        self.dt = 0.002
        self.beta = 3 * 1e-3
        self.beta1 = 5 * 1e-3
        self.beta2 = 3 * 1e-3
        self.sigma_v = 3 * 1e-2
        self.sv_mu_gain = 1 * 1e-5
        self.sigma_mu = 1
        self.K = 500

        self.a_thres = 0.3
        
        # self.sigma_v_gamma = None

        self.active_inference_mode = 1    # 1 If active inference (actions with attractor dynamics), if 0: perceptual inference
        self.adapt_sigma = 0

    def reset(self, params, start_mu):
        # torch.manual_seed(0)
        # np.random.seed(0)
        self.mu = np.reshape(start_mu, (1,2))
        self.s_v = np.empty((1, 3, self.height, self.width))
        self.g_mu = np.empty((1, 3, self.height, self.width))

        #Read the parameters:
        (self.dt, self.beta, self.sigma_v, self.sv_mu_gain, self.sigma_mu) = params

        self.active_inference_mode = 1
        #self.attractor_pos = np.reshape(attractor_pos, (1,4))
        self.attractor_im = np.empty((1, 3, self.height, self.width))
        self.attr_error = np.empty((1, 3, self.height, self.width))
        self.a = np.zeros((1, 2))
        self.a_dot = np.zeros((1, 2))
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
        self.attr_img_raw = img.copy()
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
        self.s_v_raw = img.copy()
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

    def get_dF_dmu_dyn(self, input1, out1, input2, out2):
        self.attr_error = self.attractor_im - self.g_mu
        A1 = self.beta1 * (self.attr_error).copy()
        A2 = self.beta2 * (self.attr_error).copy()
        # Set the gradient to zero before the backward pass to make sure there is no accumulation from previous backward passes
        # input.grad = torch.zeros(input.size())
        # print(input.grad)
        input1.grad = torch.zeros(input1.size(), dtype=input1.grad.dtype, device=input1.grad.device)
        out1.backward(torch.Tensor(A1*(1/self.sigma_mu)).to(device),retain_graph=True)
        input2.grad = torch.zeros(input2.size(), dtype=input2.grad.dtype, device=input2.grad.device)
        out2.backward(torch.Tensor(A2*(1/self.sigma_mu)).to(device),retain_graph=True)
        #print(input.grad)
        grad = np.zeros((1, 2))
        grad[0][0] = input1.grad.cpu().data.numpy()[0][0]
        grad[0][1] = input2.grad.cpu().data.numpy()[0][1]
        return grad # mu_dot_dyn (1x1 vector)

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

        self.attr_mse = (((self.attr_error).copy())**2).mean()
        self.vis_mse = (((self.pred_error).copy())**2).mean()
        self.rhovis_mse = ((self.s_v_raw - self.attr_img_raw)**2).mean()



        # dF/dmu using visual information:
        dF_dmu_vis = self.get_dF_dmu_vis(input, out)
        mu_dot_vis = self.sv_mu_gain * dF_dmu_vis    # mu_dot_vis is a 1x4 vector
        self.vis_error_sc = mu_dot_vis.copy()


        # dF/dmu with attractor:
        self.attr_error_sc = (self.get_dF_dmu_dyn(input, out, input, out)).copy()
        print("grad")
        print(self.attr_error_sc)
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
        #self.a[0][1] = 0.0

        # if self.adapt_sigma and np.square(self.pred_error).mean() <= 0.01:
        #     self.sigma_v = self.sigma_v * self.sigma_v_gamma
        #     self.adapt_sigma = 0 # Increase Sigma only once!


def move(fe_min, d):
    q = (d.qpos).copy()
    diff_thres = 0.0
    # Actions will only be executed if they are larger than the threshold value!

    # joint1
    diff = fe_min.a[0][0]*fe_min.dt
    targ = q[0]+ diff
    d.ctrl[0] = targ

    # joint2
    diff = fe_min.a[0][1]*fe_min.dt
    targ = q[1]+ diff
    d.ctrl[1] = targ


    #d.ctrl[1] = 0.0
    # print('Last target {}'.format(self.last_target))

#     diff = fe_min.a[0][0]*fe_min.dt
#     # targ = q[i]+ diff
#     # d.qpos[i] = targ
#     # if abs(diff)>=diff_thres:
#     targ = q[0]+ diff
#     if targ<larm_max[0] and targ>larm_min[0]:
#         d.ctrl[0] = targ
#         #d.qvel[1] = fe_min.a[0][0]
#         #print(d.qpos[1])
#         #print("\n")
#         #print("Action has been carried out")
#     else:
#         targ=np.minimum(larm_max[0],np.maximum(targ, larm_min[0]))
#         if abs(targ-q[0])>=diff_thres:
#             d.ctrl[0] = targ
#             #d.qvel[1] = fe_min.a[0][0]
#         print('Joints limit reached in: {}'.format(actors[0]))
# #print('Last target {}'.format(self.last_target))




        

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
		<motor ctrlrange="-0.698 0.698"/>
	</default>
	<option integrator="RK4" timestep="0.002"/>
	<worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <camera name="track" mode="fixed" pos="0 1 5" xyaxes="1 0 0 0 1 0"/>
        <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="1 1 1 1" size="40 40 40" type="plane"/>
        <body name="foundation" pos="0 0 0.1">
            <geom name="fd" rgba="0.3 0.3 0.3 1" size=".2 .1 .1" type="box"/>
            <body name="pole" pos="0 0 0">
                <joint axis="0 0 1" name="joint1" pos="0 0 0" range="-40 40" type="hinge"/>
                <geom fromto="0 0 0 0 0.6 0" name="cpole" rgba="0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
                <body name="pole2" pos="0 0.6 0">
                    <camera name="end" mode="fixed" pos="0 .8 0" xyaxes="1 0 0 0 0 1"/>
                    <joint axis="0 0 1" name="joint2" pos="0 0 0" range="-40 40" type="hinge"/>
                    <geom fromto="0 0 0 0 0.8 0" name="cpole2" rgba="0.0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
                </body>
            </body>
        </body>
	</worldbody>
  <actuator>
      <position name="pos_servo1" joint="joint1" ctrlrange="-0.698 0.698" kp="50000"/>
      <!--velocity name="vel_servo1" joint="joint1" kv="20"/-->
      <position name="pos_servo2" joint="joint2" ctrlrange="-0.698 0.698" kp="10000"/>
      <!--velocity name="vel_servo2" joint="joint2" kv="30"/-->
  </actuator>
  <keyframe>
    <key name="home" qpos="0.1 -0.2"/>
  </keyframe>
</mujoco>
"""


m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m)
#print(len(d.ctrl))

# initialize
d.qpos = [0.0, 0.0]

log_a = [[] for i in range(2)]
log_q = [[] for i in range(2)]
log_attr_err = [[] for i in range(2)]
log_vis_err = [[] for i in range(2)]
log_adot = [[] for i in range(2)]
log_mu = [[] for i in range(2)]
log_mu_dot = [[] for i in range(2)]
log_vis_mse = []
log_attr_mse = []
log_rhovis_mse = []
log_rho_q = [[] for i in range(2)]
log_rho_mu = [[] for i in range(2)]
log_mu_q = [[] for i in range(2)]



# set initial position
#d.qpos[0] = 35
#print(len(d.qpos))

fe_min = FE_minimize()

frames = []


with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  fe_min.mu[0] = (d.qpos).copy()
#   print("mu")
#   print((fe_min.mu).shape)
  n = 0
  sim_len = 3000
  time_len = 120
  flg = 0
  while viewer.is_running() and time.time() - start < sim_len:
    print(fe_min.targ_num)
    # if time.time() - start > time_len/2:
    #     fe_min.targ_num = targ_num2
    #     flg = 1
    step_start = time.time()
    #d.qpos[0] = -0.000000761
    #d.qpos[2:] = [-0.00032278, 0.006600549, -0.000220694, -0.000189881, 0.00000136]

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    pixels = r.render()
    frames.append(pixels)
    central_execute(fe_min, d, r)
    if n == 0:
        start = time.time()
    print(d.qpos[0])
    mujoco.mj_step(m, d)
    print(d.qpos[0])
    print("\n")

    #print(fe_min.a[0][7])

    for i in range(2):
        log_a[i].append(fe_min.a[0][i])


    for i in range(2):
        log_q[i].append(d.qpos[i])

    for i in range(2):
        log_attr_err[i].append(fe_min.attr_error_sc[0][i])

    for i in range(2):
        log_vis_err[i].append(fe_min.vis_error_sc[0][i])

    for i in range(2):
        log_adot[i].append(fe_min.a_dot[0][i])

    for i in range(2):
        log_mu[i].append(fe_min.mu[0][i])

    for i in range(2):
        log_mu_dot[i].append(fe_min.mu_dot[0][i])
    
    log_vis_mse.append(fe_min.vis_mse)
    log_attr_mse.append(fe_min.attr_mse)
    log_rhovis_mse.append(fe_min.rhovis_mse)


    for i in range(2):
        if flg == 0:
            log_rho_q[i].append((targ_q1[i] - d.qpos[i])**2)
        else:
            log_rho_q[i].append((targ_q2[i] - d.qpos[i])**2)
    
    for i in range(2):
        if flg == 0:
            log_rho_mu[i].append((targ_q1[i] - fe_min.mu[0][i])**2)
        else:
            log_rho_mu[i].append((targ_q2[i] - fe_min.mu[0][i])**2)
    
    for i in range(2):
        log_mu_q[i].append((fe_min.mu[0][i] - d.qpos[i])**2)

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

name_list = ['joint1', 'joint2']

plt.rcParams["font.size"] = 20

# a_dotのログ
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.6)

for i in range(1, 3):
    ax = fig.add_subplot(1, 2, i)
    ax.set_title(name_list[i-1])
    ax.plot(t, log_adot[i-1], label="a_dot")
    ax.set_xlim(0, time_len)

# レイアウトの調整
plt.tight_layout()
# plt.legend()
#plt.plot(log_adot[0], label="a_dot")

# グラフを表示
plt.legend()
plt.show()


# mu_dotのログ
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.6)

for i in range(1, 3):
    ax = fig.add_subplot(1, 2, i)
    ax.set_title(name_list[i-1])
    ax.plot(t, log_mu_dot[i-1], label="mu_dot")
    ax.set_xlim(0, time_len)


# レイアウトの調整
plt.tight_layout()
# plt.legend()
#plt.plot(log_adot[0], label="a_dot")

# グラフを表示
plt.legend()
plt.show()





# muのログ
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.6)

for i in range(1, 3):
    ax = fig.add_subplot(1, 2, i)
    ax.set_title(name_list[i-1])
    ax.plot(t, log_mu[i-1], label=r"$\mu$ : internal [rad]")
    ax.plot(t, log_q[i-1], label=r"$\theta$ : actual [rad]")
    ax.hlines(targ_q1[i-1], xmin=0, xmax=time_len, linestyles="dashed", colors="red", label="target [rad]")
    #ax.hlines(targ_q2[i-1], xmin=0, xmax=time_len, linestyles="dashed", colors="red")
    ax.axhspan(larm_min[i-1], larm_max[i-1], facecolor="blue", alpha=0.1, label='arm range [rad]')
    #ax.text(time_len+0.02*time_len, targ_q1[i-1], 'target1', va='center', ha='left', backgroundcolor='white')
    #ax.text(time_len+0.02*time_len, targ_q2[i-1], 'target2', va='center', ha='left', backgroundcolor='white')
    ax.set_xlabel("time [sec]", fontsize=20)
    ax.set_ylabel(r"$\theta$ [rad]", color='black')
    ax.legend()
    ax.set_xlim(0, time_len)
    ax2 = ax.twinx()
    ax2.plot(t, log_a[i-1], label=r"$a$ : action [rad/sec]", alpha=0.5, color="green")
    ax2.set_ylabel(r"$a$ [rad/sec]", color='black')
    ax2.legend()


# レイアウトの調整
# plt.tight_layout()
# plt.legend()
# plt.plot(log_mu[0], label=r"$\mu$ : internal [rad]")
# plt.plot(log_q[0], label=r"$q$ : actual [rad]")
# plt.plot(log_a[0], label=r"$a$ : action [rad/sec]")
# plt.hlines(targ_q, xmin=0, xmax=len(log_mu[0]), linestyles="dashed", colors="red")
# plt.axhspan(larm_min[0], larm_max[0], facecolor="blue", alpha=0.1, label='arm range')
# plt.text(len(log_mu[0])+0.02*len(log_mu[0]), targ_q, 'target', va='center', ha='left', backgroundcolor='white')


# グラフを表示
plt.xlim(0, time_len)
#plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()



# muのログ
# hoge = np.array(len(log_q[i-1]))
# hoge = 5
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["ytick.major.width"] = 2  
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.6)
marker_interval = 20
labels = [r"$\theta_1, \mu_1$ [rad]", r"$\theta_2, \mu_2$ [rad]"]
lg_lbl_mu = [r"$\mu_1$ : internal [rad]", r"$\mu_2$ : internal [rad]"]
lg_lbl_theta = [r"$\theta_1$ : actual [rad]", r"$\theta_2$ : actual [rad]"]
for i in range(1, 3):
    ax = fig.add_subplot(1, 2, i)
    ax.set_title(name_list[i-1])
    ax.plot(t, log_mu[i-1], label=lg_lbl_mu[i-1])
    ax.plot(t, log_q[i-1], label=lg_lbl_theta[i-1])
    

    ax.hlines(targ_q1[i-1], xmin=0, xmax=time_len, linestyles="dashed", colors="red", label="target [rad]")
    #ax.hlines(targ_q2[i-1], xmin=0, xmax=time_len, linestyles="dashed", colors="red")
    ax.axhspan(larm_min[i-1], larm_max[i-1], facecolor="blue", alpha=0.1, label='arm range [rad]')
    #ax.text(time_len+0.02*time_len, targ_q1[i-1], 'target1', va='center', ha='left', backgroundcolor='white')
    #ax.text(time_len+0.02*time_len, targ_q2[i-1], 'target2', va='center', ha='left', backgroundcolor='white')
    ax.set_xlabel("time [sec]", fontsize=30)
    ax.set_ylabel(labels[i-1], color='black', fontsize=30)
    ax.set_xlim(0, time_len)
    #ax.set_ylim(larm_min[i-1]-0.1, larm_max[i-1]+0.1)
    # ax2 = ax.twinx()
    # ax2.plot(t, log_a[i-1], label=r"$a$ : action [rad/sec]", alpha=0.2, color="green")
    # ax2.set_ylabel(r"$a$ [rad/sec]", color='black')
    # l2 = ax2.legend(loc="upper right", fontsize="15")
    l1 = ax.legend(fontsize="20")


# レイアウトの調整
# plt.tight_layout()
# plt.legend()
# plt.plot(log_mu[0], label=r"$\mu$ : internal [rad]")
# plt.plot(log_q[0], label=r"$q$ : actual [rad]")
# plt.plot(log_a[0], label=r"$a$ : action [rad/sec]")
# plt.hlines(targ_q, xmin=0, xmax=len(log_mu[0]), linestyles="dashed", colors="red")
# plt.axhspan(larm_min[0], larm_max[0], facecolor="blue", alpha=0.1, label='arm range')
# plt.text(len(log_mu[0])+0.02*len(log_mu[0]), targ_q, 'target', va='center', ha='left', backgroundcolor='white')


# グラフを表示
plt.xlim(0, time_len)
#plt.ylim(-1.5, 1.5)
#plt.legend(loc="lower right", fontsize="15")
plt.show()



# q, mu, rho
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.6)
for i in range(1, 3):
    ax = fig.add_subplot(1, 2, i)
    ax.set_title(name_list[i-1])
    ax.plot(t, log_rho_q[i-1], label=r"$\rho-q$")
    ax.plot(t, log_rho_mu[i-1], label=r"$\rho-\mu$")
    ax.plot(t, log_mu_q[i-1], label=r"$\mu-q$")
    ax.set_xlim(0, time_len)
    ax.set_xlabel("time [sec]")
    ax.set_ylabel(r"$angle err [rad^{2}]$")
    ax.legend()
# plt.xlabel("time [sec]")
# plt.ylabel(r"$angle err [rad^{2}]$")
# グラフを表示
#plt.legend()
plt.show()



# qとaのログ
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.6)
for i in range(1, 3):
    ax = fig.add_subplot(1, 2, i)
    ax.set_title(name_list[i-1])
    ax.plot(t, log_a[i-1], color="blue", label="a")
    ax.plot(t, log_q[i-1], color="red", label="q")
    ax.set_xlim(0, time_len)


# レイアウトの調整
# plt.tight_layout()
# plt.legend()
# plt.plot(log_a[0], color="blue", label="a")
# plt.plot(log_q[0], color="red", label="q")

# グラフを表示
plt.legend()
plt.show()



# attr_errのログ
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.6)
#fig_a_dot = plt.figure(figsize=(12, 12))  # フィギュアのサイズを調整

for i in range(1, 3):
    ax = fig.add_subplot(1, 2, i)
    ax.set_title(name_list[i-1])
    ax.plot(t, log_attr_err[i-1], label="attr_err")
    ax.set_xlim(0, time_len)

# グラフを表示
plt.legend()
plt.show()


# vis_errのログ
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.6)
#fig_a_dot = plt.figure(figsize=(12, 12))  # フィギュアのサイズを調整

for i in range(1, 3):
    ax = fig.add_subplot(1, 2, i)
    ax.set_title(name_list[i-1])
    ax.plot(t, log_vis_err[i-1], label="vis_err")
    ax.set_xlim(0, time_len)

# グラフを表示
plt.legend()
plt.show()

plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["ytick.major.width"] = 2
# MSE
plt.rcParams["font.size"] = 30
# plt.plot(t, log_vis_mse, label=r"$y_{v}-g(\mu)$")
# plt.plot(t, log_attr_mse, label=r"$\rho-g(\mu)$")
plt.plot(t, log_rhovis_mse, label=r"$\rho-y$")
plt.xlim(0, time_len)
plt.xlabel("time [sec]")
plt.ylabel("MSE")
# グラフを表示
plt.legend()
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
media.write_video("result/fault_sim/arm_len/movie.mp4", frames, fps=60)






###error field###
targ_num = targ_num1
plt.rcParams["font.size"] = 4
# 画像フォルダのパスを指定
image_folder_normal = "gmu_image"
image_folder_dis = "arm_len_image/0.6_0.8"

# 画像データを格納するリスト
image_list_normal = []
image_list_dis = []

# 画像を読み込んでリストに追加
for i in range(10000):
    image_path = f"{image_folder_normal}/{i+1}.png"  # 画像ファイルのパスを指定
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)  # NumPy配列に変換
    image_list_normal.append(image)

for i in range(10000):
    image_path = f"{image_folder_dis}/{i+1}.png"  # 画像ファイルのパスを指定
    image = Image.open(image_path)
    image = np.array(image)  # NumPy配列に変換
    image_list_dis.append(image)

#目標画像を読み込む
targ_path = "train_data/train_images/%d.png" % targ_num1  # 画像ファイルのパスを指定
targ_img = Image.open(targ_path)
targ_img = np.array(targ_img)

# 画像データの数
num_images = len(image_list_normal)

# 100x100のMSE行列を初期化
mse_matrix_p2 = np.zeros((100, 100))
#mse_matrix_ideal = np.zeros((100, 100))
mse_matrix_p1 = np.zeros((100, 100))
mse_matrix_p1p2 = np.zeros((100, 100))


# 画像同士のMSEを計算して行列に格納
#p2
for i in range(100):
    for j in range(100):
        mse = ((image_list_normal[100*i + j] - targ_img)**2).mean()
        mse_matrix_p2[99-j, i] = mse
        print(100*i+j)
vmin = np.min(mse_matrix_p2)
vmax = np.max(mse_matrix_p2)

#ideal
# for i in range(100):
#     for j in range(100):
#         mse = ((image_list_dis[100*i + j] - image_list_normal[targ_num-1])**2).mean()
#         mse_matrix_ideal[99-j, i] = mse
#         print(100*i+j)

#p1
for i in range(100):
    for j in range(100):
        mse = ((image_list_dis[100*i + j] - image_list_normal[100*i + j])**2).mean()
        mse_matrix_p1[99-j, i] = mse
        print(100*i+j)

#p1+p2
mse_matrix_p1p2 = mse_matrix_p1 * mse_matrix_p2

# ヒートマップの軸の目盛りを設定
xtick_labels = np.linspace(-0.698, 0.698, num=5)  # x軸のラベルを5つに分割
ytick_labels = np.linspace(-0.698, 0.698, num=5)  # y軸のラベルも同様に

xtick_positions = np.linspace(0, 99, num=5)  # x軸の位置
ytick_positions = np.linspace(99, 0, num=5)  # y軸の位置


min_range = -0.698
max_range = 0.698

def normalize_to_0_100(value, min_value, max_value):
    # まず、[-0.698, 0.698]の範囲の値を[0, 1]の範囲に変換
    normalized_value = (value - min_value) / (max_value - min_value)
    
    # 次に、[0, 1]の範囲の値を[0, 100]の範囲にスケーリング
    scaled_value = normalized_value * 100
    
    return scaled_value
# ヒートマップをプロットする関数
def plot_heatmap(ax, mse_matrix):
    ax.imshow(mse_matrix, cmap='RdBu', vmin=vmin, vmax=vmax)
    ax.set_xlabel(r"$\theta_1 [rad]")
    ax.set_ylabel(r"$\theta_2 [rad]")
    ax.set_xticks(xtick_positions)
    ax.set_yticks(ytick_positions)
    ax.set_xticklabels([f"{label:.3f}" for label in xtick_labels])
    ax.set_yticklabels([f"{label:.3f}" for label in ytick_labels])

# アニメーションの更新関数
def update(frame, path_x, path_y, line):
    # 点の座標を更新
    line.set_data(path_x[frame], path_y[frame])
    return line,

log_q = np.array(log_q)
path_x = log_q[0]
path_y = -log_q[1]

# path_x = np.linspace(0,100,100)
# path_y = np.linspace(0,100,100)

path_x = normalize_to_0_100(path_x, min_range, max_range)
path_y = normalize_to_0_100(path_y, min_range, max_range)



# 目的のピクセルサイズ
width_in_pixels = 320
height_in_pixels = 240

# DPIを100と仮定する
dpi = 100

# インチ単位でのサイズを計算
width_in_inches = width_in_pixels / dpi
height_in_inches = height_in_pixels / dpi


# アニメーションを作成するための準備(p2)
fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plot_heatmap(ax, mse_matrix_p2)  # ヒートマップをプロット
line, = ax.plot([], [], 'ro')  # 初期位置に赤い点をプロット

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=len(path_x), fargs=(path_x, path_y, line), interval=100/6, blit=True)


# 動画として保存 (ffmpegが必要)
ani.save("result/fault_sim/arm_len/error_field_p2.mp4", writer='ffmpeg', dpi=dpi)
plt.show()



# アニメーションを作成するための準備(ideal)
# fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
# plot_heatmap(ax, mse_matrix_ideal)  # ヒートマップをプロット
# line, = ax.plot([], [], 'ro')  # 初期位置に赤い点をプロット

# # アニメーションの作成
# ani = animation.FuncAnimation(fig, update, frames=len(path_x), fargs=(path_x, path_y, line), interval=100/6, blit=True)


# # 動画として保存 (ffmpegが必要)
# ani.save("result/fault_sim/arm_len/error_field_ideal.mp4", writer='ffmpeg', dpi=dpi)
# plt.show()


# アニメーションを作成するための準備(p1)
fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plot_heatmap(ax, mse_matrix_p1)  # ヒートマップをプロット
line, = ax.plot([], [], 'ro')  # 初期位置に赤い点をプロット

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=len(path_x), fargs=(path_x, path_y, line), interval=100/6, blit=True)


# 動画として保存 (ffmpegが必要)
ani.save("result/fault_sim/arm_len/error_field_p1.mp4", writer='ffmpeg', dpi=dpi)
plt.show()


# アニメーションを作成するための準備(p1p2)
fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plot_heatmap(ax, mse_matrix_p1p2)  # ヒートマップをプロット
line, = ax.plot([], [], 'ro')  # 初期位置に赤い点をプロット

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=len(path_x), fargs=(path_x, path_y, line), interval=100/6, blit=True)


# 動画として保存 (ffmpegが必要)
ani.save("result/fault_sim/arm_len/error_field_p1p2.mp4", writer='ffmpeg', dpi=dpi)
plt.show()







# 勾配図
plt.rcParams["font.size"] = 10
# ヒートマップを作成(p2)
plt.imshow(mse_matrix_p2, cmap='RdBu', vmin=vmin, vmax=vmax)
plt.colorbar()  # カラーバーを追加
# ヒートマップの勾配を計算
grad_y, grad_x = np.gradient(mse_matrix_p2)

# ヒートマップ上に矢印を描画
# 'X' と 'Y' は矢印の位置、'U' と 'V' は矢印の方向と大きさ
Y, X = np.mgrid[0:100:5, 0:100:5]
plt.quiver(X, Y, -grad_x[::5, ::5], -grad_y[::5, ::5], color='white', width=0.005)

plt.xlabel(r"$\theta_{1} \,[rad]$")
plt.ylabel(r"$\theta_{2} \,[rad]$")
plt.plot(path_x, path_y, marker='.', markersize=5, color="black")
plt.plot(path_x[0], path_y[0], marker='.', markersize=20, color="red")
plt.plot(path_x[-1], path_y[-1], marker='D', markersize=10, color="red")
plt.xlim([0,100])
# ヒートマップの軸の目盛りを設定
xtick_labels = np.linspace(-0.698, 0.698, num=5)  # x軸のラベルを5つに分割
ytick_labels = np.linspace(-0.698, 0.698, num=5)  # y軸のラベルも同様に

xtick_positions = np.linspace(0, 99, num=5)  # x軸の位置
ytick_positions = np.linspace(99, 0, num=5)  # y軸の位置

plt.xticks(xtick_positions, labels=[f"{label:.3f}" for label in xtick_labels])
plt.yticks(ytick_positions, labels=[f"{label:.3f}" for label in ytick_labels])

# plt.colorbar()  # カラーバーを追加
plt.savefig("result/fault_sim/arm_len/error_field_p2.png", dpi=300)
plt.show()




# ヒートマップを作成(ideal)
# plt.imshow(mse_matrix_ideal, cmap='RdBu')
# plt.colorbar()  # カラーバーを追加
# # ヒートマップの勾配を計算
# grad_y, grad_x = np.gradient(mse_matrix_ideal)

# # ヒートマップ上に矢印を描画
# # 'X' と 'Y' は矢印の位置、'U' と 'V' は矢印の方向と大きさ
# Y, X = np.mgrid[0:100:2, 0:100:2]
# plt.quiver(X, Y, -grad_x[::2, ::2], -grad_y[::2, ::2], color='white')

# plt.xlabel("joint1 [rad]")
# plt.ylabel("joint2 [rad]")
# plt.plot(path_x, path_y, marker='.', markersize=5, color="black")
# plt.plot(path_x[0], path_y[0], marker='.', markersize=20, color="red")
# plt.plot(path_x[-1], path_y[-1], marker='D', markersize=10, color="red")
# plt.xlim([0,100])
# # ヒートマップの軸の目盛りを設定
# xtick_labels = np.linspace(-0.698, 0.698, num=5)  # x軸のラベルを5つに分割
# ytick_labels = np.linspace(-0.698, 0.698, num=5)  # y軸のラベルも同様に

# xtick_positions = np.linspace(0, 99, num=5)  # x軸の位置
# ytick_positions = np.linspace(99, 0, num=5)  # y軸の位置

# plt.xticks(xtick_positions, labels=[f"{label:.3f}" for label in xtick_labels])
# plt.yticks(ytick_positions, labels=[f"{label:.3f}" for label in ytick_labels])

# # plt.colorbar()  # カラーバーを追加
# plt.savefig("result/fault_sim/arm_len/error_field_ideal.png", dpi=300)
# plt.show()



# ヒートマップを作成(p1)
plt.imshow(mse_matrix_p1, cmap='RdBu', vmin=vmin, vmax=vmax)
plt.colorbar()  # カラーバーを追加
# ヒートマップの勾配を計算
grad_y, grad_x = np.gradient(mse_matrix_p1)

# ヒートマップ上に矢印を描画
# 'X' と 'Y' は矢印の位置、'U' と 'V' は矢印の方向と大きさ
Y, X = np.mgrid[0:100:5, 0:100:5]
plt.quiver(X, Y, -grad_x[::5, ::5], -grad_y[::5, ::5], color='white', width=0.005)

plt.xlabel(r"$\theta_{1} \,[rad]$")
plt.ylabel(r"$\theta_{2} \,[rad]$")
plt.plot(path_x, path_y, marker='.', markersize=5, color="black")
plt.plot(path_x[0], path_y[0], marker='.', markersize=20, color="blue")
plt.plot(path_x[-1], path_y[-1], marker='D', markersize=10, color="blue")
plt.xlim([0,100])
# ヒートマップの軸の目盛りを設定
xtick_labels = np.linspace(-0.698, 0.698, num=5)  # x軸のラベルを5つに分割
ytick_labels = np.linspace(-0.698, 0.698, num=5)  # y軸のラベルも同様に

xtick_positions = np.linspace(0, 99, num=5)  # x軸の位置
ytick_positions = np.linspace(99, 0, num=5)  # y軸の位置

plt.xticks(xtick_positions, labels=[f"{label:.3f}" for label in xtick_labels])
plt.yticks(ytick_positions, labels=[f"{label:.3f}" for label in ytick_labels])

# plt.colorbar()  # カラーバーを追加
plt.savefig("result/fault_sim/arm_len/error_field_p1.png", dpi=300)
plt.show()




# ヒートマップを作成(p1p2)
plt.imshow(mse_matrix_p1p2, cmap='RdBu')
plt.colorbar()  # カラーバーを追加
# ヒートマップの勾配を計算
grad_y, grad_x = np.gradient(mse_matrix_p1p2)

# ヒートマップ上に矢印を描画
# 'X' と 'Y' は矢印の位置、'U' と 'V' は矢印の方向と大きさ
Y, X = np.mgrid[0:100:5, 0:100:5]
plt.quiver(X, Y, -grad_x[::5, ::5], -grad_y[::5, ::5], color='white', width=0.005)

plt.xlabel(r"$\theta_{1} \,[rad]$")
plt.ylabel(r"$\theta_{2} \,[rad]$")
plt.plot(path_x, path_y, marker='.', markersize=5, color="black")
plt.plot(path_x[0], path_y[0], marker='.', markersize=20, color="red")
plt.plot(path_x[-1], path_y[-1], marker='D', markersize=10, color="red")
plt.xlim([0,100])
# ヒートマップの軸の目盛りを設定
xtick_labels = np.linspace(-0.698, 0.698, num=5)  # x軸のラベルを5つに分割
ytick_labels = np.linspace(-0.698, 0.698, num=5)  # y軸のラベルも同様に

xtick_positions = np.linspace(0, 99, num=5)  # x軸の位置
ytick_positions = np.linspace(99, 0, num=5)  # y軸の位置

plt.xticks(xtick_positions, labels=[f"{label:.3f}" for label in xtick_labels])
plt.yticks(ytick_positions, labels=[f"{label:.3f}" for label in ytick_labels])

# plt.colorbar()  # カラーバーを追加
plt.savefig("result/fault_sim/arm_len/error_field_p1p2.png", dpi=300)
plt.show()