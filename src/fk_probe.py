#!/usr/bin/env python
"""Check the robot model against reality: torque off, then continuously print where forward
kinematics (URDF + your joint calibration) thinks the gripper tip is, in the base frame.

Move the arm by hand and touch the closed fingertips to the table: z should read ~0 mm
(within ~1 cm) if the table and the robot base share a plane and the joint calibration matches
the URDF. Also try a point a measured distance in front of the base: x should match it.
The arm is limp the whole time - support it. Ctrl+C to stop.
"""

import argparse
import sys
import time

import numpy as np

from cube_pick_common import ALL_JOINTS, ARM_JOINTS, DEFAULT_URDF
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--port", required=True)
p.add_argument("--id", required=True)
args = p.parse_args()

kin = RobotKinematics(str(DEFAULT_URDF), "gripper_frame_link", joint_names=ARM_JOINTS)
robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
robot.connect()
robot.bus.disable_torque()
print("Torque off. Touch the fingertips to the table and read z.\n")
try:
    while True:
        pos = robot.bus.sync_read("Present_Position")
        q = np.array([pos[n] for n in ALL_JOINTS], dtype=float)
        tip = kin.forward_kinematics(q)[:3, 3] * 1000
        joints = " ".join(f"{v:6.1f}" for v in q[:5])
        sys.stdout.write(f"\rtip x {tip[0]:7.1f}  y {tip[1]:7.1f}  z {tip[2]:7.1f} mm   | joints {joints}   ")
        sys.stdout.flush()
        time.sleep(0.2)
except KeyboardInterrupt:
    print()
finally:
    robot.disconnect()
