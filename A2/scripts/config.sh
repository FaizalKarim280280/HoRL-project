# Settings for the A2 demonstration scripts. Ports, arm ids and the wrist camera come from
# common/robot.sh (shared with A1); everything specific to the dataset is here.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # A2/scripts
A2_DIR="$(dirname "$SCRIPTS_DIR")"                           # A2
source "$A2_DIR/../common/robot.sh"

# Cameras, chosen by NAME (case-insensitive part of it) so they keep working when macOS renumbers
# them; an OpenCV index also works. ./00_find_cameras.sh lists the names.
WRIST_CAMERA="$WRIST_CAMERA_INDEX"   # from common/robot.sh ("1080p")
SCENE_CAMERA="720P"                  # scene camera (Innomaker U20CAM-720P, looking down at the table)
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
FPS=30

# Dataset. Recording writes raw demonstrations (JPEGs + joint data) to data/raw/<DATASET_NAME>/;
# ./03_convert.sh turns them into a LeRobot dataset in data/lerobot/<DATASET_NAME>/ for training.
# Recording again with the same name adds episodes. One-off name: DATASET_NAME=my_test ./02_record.sh
DATASET_NAME="${DATASET_NAME:-so101_pick_place}"
RAW_ROOT="$A2_DIR/data/raw/$DATASET_NAME"
DATASET_REPO_ID="local/$DATASET_NAME"
DATASET_ROOT="$A2_DIR/data/lerobot/$DATASET_NAME"
TASK="Pick up the red cube and place it in the blue tray."
JPEG_QUALITY=95

# OpenCV index of a camera from its name (see cameras.py).
camera_index() {
  python "$SCRIPTS_DIR/cameras.py" index "$1"
}
