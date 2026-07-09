# Offline YOLO26 bag segmentation

This workflow copies an input ROS 2 (mcap) bag to a new output bag and injects
YOLO26 instance-segmentation topics for every image on a chosen source image
topic. It runs `bag_segmenter_yolo26.py` (the `segmenter_ros` package) so the
segmentation model does not need to run online.

For every frame on `--raw-image-topic` the writer adds:

- `--instance-mask-topic` (default `/camera/color/image_instance_masks`):
  `sensor_msgs/msg/Image`, `mono8`, a single binary mask covering all
  detected "movable" instances.
- `--instance-masks-array-topic` (default
  `<instance-mask-topic>_array`): `segmenter_ros/msg/InstanceMaskArray`, one
  `InstanceMask` per detection (class id, score, ROI, cropped mask).

All other topics from the input bag are copied through unchanged. If the
input bag already contains the output topics, they are replaced.

## Build

Build this thin wrapper on top of your existing `vsgraphs_ros2` image:

```bash
docker build \
  -f scene_segment_ros/docker/offline_bag_segmentation/Dockerfile \
  -t offline-bag-segmentation \
  --build-arg BASE_IMAGE=vsgraphs_ros2 \
  --build-arg USERNAME="$USERNAME" \
  .
```

Run the command from `/home/$USERNAME/workspaces/dv_sgraphs/src` so the
Docker build context contains `scene_segment_ros`.

## Run

Example using the same dataset mount style as `visual_sgraphs/docker`:

```bash
docker run --rm --gpus all --network host \
  -v "/home/$USERNAME/workspaces/dv_sgraphs/src/scene_segment_ros:/home/$USERNAME/workspace/src/scene_segment_ros:rw" \
  -v "/media/$USERNAME/muffin/datasets:/home/$USERNAME/datasets:rw" \
  offline-bag-segmentation \
  /home/$USERNAME/datasets/input_bag \
  /home/$USERNAME/datasets/input_bag_with_instance_masks \
  --raw-image-topic /camera/realsense/color/image_raw \
  --instance-mask-topic /camera/color/image_instance_masks \
  --model-path include/yolo26x-seg.pt \
  --image-size 640 \
  --confidence 0.4 \
  --overwrite
```

```bash
docker run --rm --gpus all --network host \
  -v "/home/$USERNAME/workspaces/dv_sgraphs/src/scene_segment_ros:/home/$USERNAME/workspace/src/scene_segment_ros:rw" \
  -v "/media/$USERNAME/muffin/datasets:/home/$USERNAME/datasets:rw" \
  offline-bag-segmentation \
  /home/$USERNAME/datasets/dynamic/uniLuCampus/OUTRO/OUTRO.mcap \
  /home/$USERNAME/datasets/dynamic/uniLuCampus/OUTRO_seg/OUTRO_seg.mcap \
  --raw-image-topic /camera/realsense/color/image_raw \
  --model-path include/yolo26x-seg.pt
```

The output bag argument is optional; if omitted it defaults to
`<input_bag>_augmented`. Do not point the output path at an existing bag
directory unless you also pass `--overwrite` — rosbag2 expects to create the
output storage itself.

Run with no arguments (or `--help`, the default `CMD`) to print the full
list of flags (`--raw-image-topic`, `--instance-mask-topic`,
`--instance-masks-array-topic`, `--model-path`, `--image-size`,
`--confidence`, `--movable-class-ids`, `--overwrite`).

## Notes

- Both the input and output bags are read/written with the `mcap` storage
  plugin (hardcoded in the script); `sqlite3` bags are not supported here.
- `--model-path` is resolved relative to the `segmenter_ros` package share
  directory when not absolute, so the default `include/yolo26x-seg.pt`
  picks up the checkpoint already bundled under
  `scene_segment_ros/include/`. Point it at any of the other bundled
  `include/yolo26*-seg.pt` checkpoints to trade off speed vs. accuracy.
- `--movable-class-ids` defaults to the COCO ids treated as movable objects
  by the online segmenter; override it to keep a different set of classes.
- Later, launch the normal pipeline with the online segmenter disabled and
  consume `--instance-mask-topic` / `--instance-masks-array-topic` from this
  augmented bag instead.
