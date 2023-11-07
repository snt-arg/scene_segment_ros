# Scene Segmenter

![Scene Segmenter](demo.gif "Scene Segmenter")

A framework for real-time (semantic )segmentation of given images (video frames) based on the given parameters and configurations. The main use case of this repository is to be used in [vS-Graphs](https://github.com/snt-arg/visual_sgraphs), where the camera output seen by the robot is sent to the current package to be segmented.

The framework currently covers two main modules, including **[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)** for real-time scene segmentation and **[PanopticFCN](https://github.com/dvlab-research/PanopticFCN)** for real-time scene segmentation and semantic object detection.

### 📊 Models Benchmarking

[Here](https://github.com/snt-arg/scene_segmentation_benchmark) you can see the benchmarking results of the work using different libraries.

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

### III. Installing Models

Download one of the model checkpoints ([PanopticFCN](https://github.com/dvlab-research/PanopticFCN#results) or [Fast-SAM](https://github.com/CASIA-IVA-Lab/FastSAM#replicate-demo)) from the repository, and specify the path in the `config/cfg_[model].yaml` file.

The next step is to install the required frameworks:

- For **FastSAM**, you need to install required _OpenAI_ modules using `pip install git+https://github.com/openai/CLIP.git`.
- For **PanopticFCN**, you need to install _Detectron2_ using `git clone https://github.com/facebookresearch/detectron2.git` and then, `python -m pip install -e detectron2`.

## 🔨 Configurations

The system has different configurations for each of the segmentation libraries, accessible from `/config` folder. In the table below, you can see these configurations in details.

| Main Category  | Parameter               | Default        | Description                  |
| -------------- | ----------------------- | -------------- | ---------------------------- |
| `image_params` | `image_params`          | 640            | width of the input image     |
| `ros_topics`   | `raw_image_topic`       | `/img`         | raw image topic              |
|                | `segmented_image_topic` | `/seg`         | segmented image topic        |
| `model_params` | `model_name`            | -              | name of the model            |
|                | `model_path`            | -              | path of the model file       |
|                | `point_prompt`          | [[0, 0]]       | a point for segmentation     |
|                | `box_prompt`            | [[0, 0, 0, 0]] | boxes for segmentation       |
|                | `text_prompt`           | -              | text prompt (e.g., "a dog")  |
|                | `point_label`           | [0]            | 0: background, 1: foreground |
|                | `iou`                   | 0.9            | annots filtering threshold   |
|                | `conf`                  | 0.4            | object confidence threshold  |
|                | `contour`               | False          | draw contours                |

## 🚀 Running the Code

You can run the below launch files (accessible from `/launch` folder):

- **FastSAM**: `roslaunch segmenter_ros segmenter_fastSAM.launch`
- **PanopticFCN**: `roslaunch segmenter_ros segmenter_pFCN.launch`

## 📅 TODO

- Excluding unnecessary segments
- Removing logs
- Providing segment information
- Merging it with ROS-based ORB-SLAM for mapping
- Testing with a robot
  - Drone/Spot
