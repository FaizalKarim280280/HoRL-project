# Shared settings for the SO-101 scripts. Edit here if ports or ids change.
#OLLOWER_PORT=/dev/tty.usbmodem5B7B0094501 
FOLLOWER_PORT=/dev/tty.usbmodem5B7B0094501
FOLLOWER_ID=so101_follower_arm
LEADER_PORT=/dev/tty.usbmodem5B7B0095441
LEADER_ID=so101_leader_arm
WRIST_CAMERA_INDEX="Innomaker"   # wrist camera, chosen by NAME so the MacBook webcam is never opened (a number also works)
BOARD_ARGS="--squares-x 7 --squares-y 5 --dictionary DICT_6X6_250 --square-mm 22 --marker-mm 15"   # your ChArUco board
TABLE_Z=-0.01   # table height relative to the robot base (m)
TARGET_OFFSET="-0.005 0.01"   # constant correction (m) added to the detected object x, y (base frame)
PLATE_OFFSET="0 0"   # constant correction (m) added to the detected blue plate centre x, y (base frame)
