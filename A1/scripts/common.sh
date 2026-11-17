# Settings for the A1 (classical pick and place) scripts. Ports, ids and the wrist camera come from common/robot.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../common/robot.sh"
BOARD_ARGS="--squares-x 7 --squares-y 5 --dictionary DICT_6X6_250 --square-mm 22 --marker-mm 15"   # your ChArUco board
TABLE_Z=-0.01   # table height relative to the robot base (m)
TARGET_OFFSET="-0.005 0.015"   # constant correction (m) added to the detected object x, y (base frame)
PLATE_OFFSET="-0.005 0.0"   # constant correction (m) added to the detected tray centre x, y (base frame); start = TARGET_OFFSET, fine-tune live in viser
