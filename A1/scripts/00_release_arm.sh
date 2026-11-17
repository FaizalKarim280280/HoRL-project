#!/usr/bin/env bash
# Turn the follower's motor torque off so it can be moved by hand. SUPPORT THE ARM FIRST - it goes limp.
cd "$(dirname "$0")" && source ./common.sh
python ../src/release_torque.py --port $FOLLOWER_PORT --id $FOLLOWER_ID
