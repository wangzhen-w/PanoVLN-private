# PanoVLN Real-world Deployment

This directory contains the real-world deployment path for the PanoVGGT-based
VLN checkpoint. It does not use vLLM because the checkpoint adds custom
PanoVGGT inputs and model code.

The server first checks whether the VLN checkpoint itself contains saved
`panovggt.*` encoder weights. If they are present, those weights are used.
`PANOVGGT_CHECKPOINT` in `run_server.sh` is only a fallback for checkpoints that
did not save the encoder inside the same directory.

## Server

Run on the model server, for example `10.14.114.132`:

```bash
cd /workspace/code/VLN
pip install -r realworld/requirements-server.txt
bash realworld/run_server.sh
```

All server parameters are assigned at the top of `run_server.sh`. Edit
`ATTN_IMPLEMENTATION="eager"` there if the server environment does not have
flash attention installed. Use `GPU_IDS="0"` to choose the visible GPU.
The model loads before the HTTP server starts, and startup prints stage-by-stage
loading logs.
Crop degrees, dtype, device, and generation length are not script parameters:
they follow the same defaults as Habitat eval, with crop read from the model
`config.json`.

The default endpoint is:

```text
POST http://10.14.114.132:8000/predict
```

Multipart fields:

- `instruction`: navigation instruction text.
- `images`: one or more JPEG/PNG files ordered from older to newer.

Response:

```json
{
  "actions": ["forward", "left", "stop", "stop"],
  "executable_actions": ["forward", "left", "stop"],
  "raw_text": "forward left stop stop",
  "prompt_images": 3,
  "latency_s": 1.23
}
```

## Unitree Go2 Client

Run on the robot. The Insta360 X5 must be visible as an OpenCV camera source
through USB webcam/UVC mode.

Dry run first:

```bash
cd /workspace/code/VLN
pip install -r realworld/requirements-robot.txt
bash realworld/run_go2_client.sh
```

The instruction is entered at the top of `run_go2_client.sh`:

```bash
INSTRUCTION="Walk forward to the hallway and stop by the door."
```

For real robot motion, edit the same script:

```bash
SERVER_BASE_URL="http://10.14.114.132:8000"
CONTROL_BACKEND="sdk2"
REAL_ROBOT_ACK="yes"
NETWORK_INTERFACE="eth0"
CAMERA="0"
```

`FRAME_WIDTH`, `FRAME_HEIGHT`, and `CAMERA_FPS` can be filled in, but they are
only requests sent to the USB/UVC camera driver. The camera may ignore them if
that mode is unsupported. Check the actual mode printed by the client, or
inspect supported modes with:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

The client does not collect an image queue at `CAMERA_FPS`. It captures one
initial observation, then captures one new observation after each executed
non-stop action. `CAPTURE_FLUSH_FRAMES` discards a few buffered frames before
each observation so the image is less likely to be stale.

To reduce Wi-Fi transfer time, the client resizes selected frames before upload:

```bash
FRAME_WIDTH="2880"
FRAME_HEIGHT="1440"
CAMERA_FPS="30"
CAPTURE_FLUSH_FRAMES="2"
UPLOAD_IMAGE_MODE="resize"
UPLOAD_WIDTH="1280"
UPLOAD_HEIGHT="640"
JPEG_QUALITY="90"
```

Set `UPLOAD_IMAGE_MODE="raw"` to upload the captured frames unchanged. The server
still performs the final eval-style resize and ERP crop using the model
`config.json`, so crop parameters do not need to be duplicated on the robot.

`SERVER_BASE_URL` is only the server root. The client appends `/predict`
internally to match the server endpoint. Before standing the robot, the client
captures one camera frame and checks the server `/ready` endpoint.

Use `CONTROL_BACKEND="ros2"` instead if the Go2 is controlled through the
Unitree ROS2 bridge.

The client executes the model's action words as:

- `forward`: move forward 0.25 m.
- `left`: turn left 15 degrees.
- `right`: turn right 15 degrees.
- `stop`: stop and exit.

Tune motion conservatively on the robot with `FORWARD_SPEED`, `YAW_SPEED`,
`FORWARD_DISTANCE`, and `TURN_DEGREES` in `run_go2_client.sh`.
By default, the real-world client executes up to four atomic actions per model
request (`ACTIONS_PER_REPLAN="4"`), matching Habitat eval. If a returned action
is `stop`, the current instruction task ends immediately. The client stops after
`MAX_REPLANS` replans unless you set `MAX_REPLANS="0"` for unlimited replanning.
