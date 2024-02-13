# Scene (Semantic) Segmenter

| Scene Segmentation using **Fast-SAM**    | Semantic Segmentation using **PanopticFCN**        |
| ---------------------------------------- | -------------------------------------------------- |
| ![Fast-SAM](demo_fastSAM.gif "Fast-SAM") | ![PanopticFCN](demo_panopticFCN.gif "PanopticFCN") |

A framework for real-time (semantic) segmentation of given images (video frames) based on the given parameters and configurations. The main use case of this repository is to be used in [vS-Graphs](https://github.com/snt-arg/visual_sgraphs), where the camera output seen by the robot is sent to the current package to be segmented.

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

Download one of the model checkpoints ([PanopticFCN](https://github.com/dvlab-research/PanopticFCN#results) or [Fast-SAM](https://github.com/CASIA-IVA-Lab/FastSAM#replicate-demo)) from the repository, and put it in the `/include` folder. Remember that you need to specify the path in the `config/cfg_[model].yaml` file (e.g., `include/[model].pth`).

⚠️ It is recommended to put the models in the `/include` directory. The main reason for better integration of different frameworks, we read the **absolute path** of this folder to target a model.

The next step is to install the required frameworks:

- For **FastSAM**, you need to install required _OpenAI_ modules using `pip install git+https://github.com/openai/CLIP.git`.
- For **PanopticFCN**, you need to install _Detectron2_ using `git clone https://github.com/facebookresearch/detectron2.git` and then, `python -m pip install -e detectron2`.

## 🔨 Configurations

The system has different configurations for each of the segmentation libraries, accessible from `/config` folder. In the table below, you can see these configurations in details.

| Main Category  | Parameter               | Default        | Description                           |
| -------------- | ----------------------- | -------------- | ------------------------------------- |
| `image_params` | `image_params`          | 640            | width of the input image              |
| `ros_topics`   | `raw_image_topic`       | `/img`         | raw image topic                       |
|                | `segmented_image_topic` | -              | segmented image topic (custom Msg)    |
|                | `segmented_image_vis`   | -              | segmented image topic (visualization) |
| `model_params` | `model_name`            | -              | name of the model                     |
|                | `model_path`            | -              | path of the model file                |
|                | `model_config`          | -              | path of the model's specific configs  |
|                | `point_prompt`          | [[0, 0]]       | a point for segmentation              |
|                | `box_prompt`            | [[0, 0, 0, 0]] | boxes for segmentation                |
|                | `text_prompt`           | -              | text prompt (e.g., "a dog")           |
|                | `point_label`           | [0]            | 0: background, 1: foreground          |
|                | `iou`                   | 0.9            | annots filtering threshold            |
|                | `conf`                  | 0.4            | object confidence threshold           |
|                | `contour`               | False          | draw contours                         |

### Results Filtration

In order get only the classes that you want (such as `wall` or `floor`), you need to know the identifier of the class in **Detectron2** ([link](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/builtin_meta.py)) and set the `category_id` in the [configuration file](config/cfg_pFCN.yaml). For instance, the array `[43, 52]` can detect `[floors, walls]` in the scene.

## 🚀 Running the Code

You can run the below launch files (accessible from `/launch` folder):

- **FastSAM**: `roslaunch segmenter_ros segmenter_fastSAM.launch`
- **PanopticFCN**: `roslaunch segmenter_ros segmenter_pFCN.launch`
- **SegFormer**: `roslaunch segmenter_ros segmenter_segFormer.launch`

## 🤖 ROS Topics, Params and Services

### Subscribed Topics

| Topic                        | Description                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------------- |
| `/orb_slam3/keyframe_image`  | for providing input KeyFrames to be segmented (for normal frames use `/camera/color/image_raw`) topic |
| ---- `keyFrameId (UInt64)`   | the ID of the KeyFrame sent to be processed                                                           |
| ---- `keyFrameImage (Image)` | the content of the KeyFrame image sent to be processed                                                |

### Published Topics

| Topic                                          | Description                             |
| ---------------------------------------------- | --------------------------------------- |
| `/camera/color/image_segment`                  | the output semantically segmented topic |
| ---- `keyFrameId (UInt64)`                     | the ID of the processed KeyFrame        |
| ---- `segmentedImage (Image)`                  | the segmented image                     |
| ---- `segmentedImageUncertainty (Image)`       | the segmented image uncertainty values  |
| ---- `segmentedImageProbability (PointCloud2)` | the segmented image probability values  |

## 📅 TODO

- Removing unnecessary logs
- A new repository to contain all ROS messages (similar to moveit_msgs)
