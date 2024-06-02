import mujoco
import mediapy as media
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from typing import Callable, NamedTuple, Optional, Union, List
import mujoco.viewer
import csv
from PIL import Image


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
	<option integrator="RK4" timestep="0.02"/>
	<worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <camera name="track" mode="fixed" pos="0 1 5" xyaxes="1 0 0 0 1 0"/>
        <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="1 1 1 1" size="40 40 40" type="plane"/>
        <body name="foundation" pos="0 0 0.1">
            <geom name="fd" rgba="0.3 0.3 0.3 1" size=".2 .1 .1" type="box"/>
            <body name="pole" pos="0 0 0">
                <joint axis="0 0 1" name="joint1" pos="0 0 0" range="-40 40" type="hinge"/>
                <geom fromto="0 0 0 0 1.2 0" name="cpole" rgba="0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
                <body name="pole2" pos="0 1.2 0">
                    <camera name="end" mode="fixed" pos="0 .8 0" xyaxes="1 0 0 0 0 1"/>
                    <joint axis="0 0 1" name="joint2" pos="0 0 0" range="-40 40" type="hinge"/>
                    <geom fromto="0 0 0 0 0.2 0" name="cpole2" rgba="0.0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
                </body>
            </body>
        </body>
	</worldbody>
  <actuator>
      <velocity name="vel_servo1" joint="joint1"/>
      <velocity name="vel_servo2" joint="joint2"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0.1 -0.2"/>
  </keyframe>
</mujoco>
"""



m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m)


internal_data_train = np.zeros((10000, 2))
#internal_data_test = np.zeros((900, 2))

# num1 = 2
# num2 = 3
# t1 = np.linspace(0, 2*np.pi*num1, 9000)
# t2 = np.linspace(0, 2*np.pi*num2, 9000)
# a1 = 0.698 * np.sin(t1)
# a2 = 0.698 * np.sin(t2)

# set initial position
#d.qpos[0] = 35
#print(len(d.qpos))
print(len(d.ctrl))
print(len(d.qpos))


with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  i = 0
  k = 0
  q1 = np.linspace(-0.698, 0.698, 100)
  q2 = np.linspace(-0.698, 0.698, 100)
  while viewer.is_running() and time.time() - start < 3000000:
    j = int(i / 100)
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.

    # control joints
    while time.time() - step_start < 0.1:  
      d.qpos[0] = q1[j]
      d.qpos[1] = q2[k]
      mujoco.mj_step(m, d)


    #mujoco.mj_step(m, d)
    r.update_scene(d, camera="track")


    

    # save images and internal data
    # image
    #img_path_train = "train_data/train_images/%d.png" % i
    img_path_test = "arm_len_image/1.2_0.2/%d.png" % (i+1)
    media.write_image(img_path_test, r.render())
    # resize
    # img = Image.open(img_path)
    # img_resized = img.resize((80, 60))
    # img_resized.save(img_path)

    # internal data
    internal_data_train[i] = d.qpos

    k += 1
    if k == 100:
       k = 0

    i += 1
    if i == 10000:
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

# csv_path_train = r"train_data/train_internal/train.csv"
# with open(csv_path_train, "w", newline="") as file:
#     writer = csv.writer(file)

#     for row in internal_data_train:
#         writer.writerow(row)
