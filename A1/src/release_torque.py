#!/usr/bin/env python
"""Turn off torque on all follower motors so the arm can be moved by hand.

The arm goes limp immediately: support it first, or it will drop.
"""

import argparse

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--port", required=True)
p.add_argument("--id", required=True)
args = p.parse_args()

robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
robot.bus.connect()
robot.bus.disconnect(disable_torque=True)
print("Torque off - the arm can now be moved by hand.")
