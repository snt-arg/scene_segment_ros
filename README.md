# Scene Segmenter

A framework for real-time segmentation of given images (video frames) based on the given parameters and configurations. The main use case of this repository is to be used in [vS-Graphs](https://github.com/snt-arg/visual_sgraphs), where the camera output seen by the robot is sent to the current package to be segmented.

## 📝 Prerequisites

The required libraries ...

## 📚 Preparation

### I. Cloning

Create a new workspace and clone the repo in its `src` folder. In case the repository is going to be used for `vS-Graphs`, it is recommended to clone the repository in the same `src` folder where it exists, as `vS-Graphs` depends on it. Accordingly, you can use the command below:

```
git clone --recurse-submodules git@github.com:snt-arg/scene_segment_ros.git
```

After cloning the repository, you may add a command like `alias sourcecsr='source ~/workspace/[PATH]/devel/setup.bash'` in your `.bashrc` file.

### II. Installing Python Libraries

Install the required `Python` libraries for running this program using the command below:

```
pip install -r src/requirements.txt
```

## 🔨 Configurations

The parameters to be set and used ....

## 🚀 Running the Code

How to run the code ....

## 📊 Segmenter Models Benchmarking

Here you can see the benchmarking results of the work in the table below. Some of these libraries are already available in [this repository](https://github.com/snt-arg/scene_segmentation).

| Method / Image Resolution                                                      | 480x360     | 640x480     | 720x540     |
| ------------------------------------------------------------------------------ | ----------- | ----------- | ----------- |
| [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) | 6.8 seconds | 6.9 seconds | 6.9 seconds |
| Semantic SAM                                                                   | ~2 minutes  | ~2 minutes  | ~2 minutes  |
| [Segment Any RGBD](https://github.com/Jun-CEN/SegmentAnyRGBD)                  | ~2 minutes  | ~2 minutes  | ~2 minutes  |
| [FAST SAM](https://github.com/CASIA-IVA-Lab/FastSAM)                           | 2.4 seconds | 4.5 seconds | 6.5 seconds |
| Mobile SAM                                                                     | 4.1 seconds | 4.7 seconds | 4.3 seconds |

## 📅 TODO

- Finding a realtime Semantic Segmentation framework
  - Benchmarking and comparing
- Merging it with ROS-based ORB-SLAM for mapping
- Testing with a robot
  - Drone/Spot
