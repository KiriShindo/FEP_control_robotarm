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
		<motor ctrlrange="-1.396 1.396"/>
	</default>
	<option integrator="RK4" timestep="0.02"/>
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
        <position name="actuator1" joint="joint1" kp="1"/>
    </actuator>
</mujoco>
"""



m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m)


#internal_data_train = np.zeros((9000, 1))
internal_data_test = np.zeros((1000, 1))

# set initial position
#d.qpos[0] = 35
#print(len(d.qpos))


with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  i = 0
  n = 0
  a = np.linspace(-1.396, 1.396, 1000)
  while viewer.is_running() and time.time() - start < 3000000:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.

    # control joints
    d.qpos[0] = a[i]

    mujoco.mj_step(m, d)
    r.update_scene(d, camera="track")



    # save images and internal data
    # image
    #img_path_train = "train_data/train_images/%d.png" % i
    img_path_test = "test_data/test_images/%d.png" % (i+1)
    media.write_image(img_path_test, r.render())
    # resize
    # img = Image.open(img_path)
    # img_resized = img.resize((80, 60))
    # img_resized.save(img_path)

    # internal data
    internal_data_test[i] = d.qpos
    n += 1

    i += 1

    if i >= 1000:
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

csv_path_train = r"test_data/test_internal/test.csv"
with open(csv_path_train, "w", newline="") as file:
    writer = csv.writer(file)

    for row in internal_data_test:
        writer.writerow(row)
