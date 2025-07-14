<div align="center">


# HiM2SAM 
**Enhancing SAM2 with Hierarchical Motion Estimation and Memory Optimization towards Long-term Tracking**

** **  
[Ruixiang Chen](https://scholar.google.com/citations?user=U_ew7O4AAAAJ&hl=en)¹, [Guolei Sun](https://guoleisun.github.io)²<sup>✉</sup>, [Yawei Li](https://yaweili.bitbucket.io)³, [Jie Qin](https://sites.google.com/site/firmamentqj/)⁴, [Luca Benini](https://ee.ethz.ch/the-department/people-a-z/person-detail.luca-benini.html)³


¹ KTH Royal Institute of Technology, Stockholm, Sweden  
² CVL, ETH Zurich, Zurich, Switzerland  
³ IIS, ETH Zurich, Zurich, Switzerland  
⁴ Nanjing University of Aeronautics and Astronautics, Nanjing, China
</div>

<p align="center">
  <a href="http://arxiv.org/abs/2507.07603">📃 Arxiv</a> &nbsp; | &nbsp;
  <a href="https://drive.google.com/file/d/1IwUqABs91H14rpSGne1pDON9Dbsvm-AK/view?usp=sharing">📊 Raw Results</a>
</p>

## 🌟 Highlights
Our method enhances the SAM2 framework for video object tracking with trainless, low-overhead improvements that significantly boost long-term tracking performance.

•**Hierarchical Motion Estimation**: Combines lightweight linear prediction with selective non-linear refinement for accurate tracking without extra training.

•**Optimized Memory Bank**: Distinguishes short-term and long-term memory with motion-aware filtering to improve robustness under occlusion and appearance changes.
<div style="text-align: center;">
<img src="assets/mainmethod.png" alt="Framework Overview" style="width: 100%; max-width: 540px; height: auto;" />
</div>
</div>


## ⚖️ Comparisions
We compare the visualization results of HiM2SAM with SAMURAI and DAM4SAM on long video sequences.

HiM2SAM produces more stable and accurate tracking in long-term, challenging scenarios, showing improved robustness over the baselines.

<table border="0">
  <tr>
    <td>
      <img src="assets/label2.png" alt="Legend Image" width="300"/>
    </td>
  </tr>
</table>

### Motion & Appearance Variation

<img src="assets/b_m3_clip.gif" alt="Motion & Appearance Variation" style="max-width: 100%; width: 480px;" />

### Occlusion

<img src="assets/c_m3_clip.gif" alt="Occlusion" style="max-width: 100%; width: 480px;" />

### Reappearance & Background Clutter

<img src="assets/ca_m3_clip.gif" alt="Reappearance & Background Clutter" style="max-width: 100%; width: 480px;" />


## 📚 Table of Contents



- [Installation](#️-installation)
- [Data Preparation](#-data-preparation)
- [Running Inference and Visualization](#-Running-Inference-and-Visualization)
- [Running Evaluation](#-running-evaluation)
- [Demo on Custom Video](#demo-on-custom-video)
- [Citation and Acknowledgment](#Citation-and-Acknowledgment)


## 🛠️ Installation 
**Requirements**

`python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`

Our environment is tested on both RTX 3090 and A100 GPUs.

1. **Install SAM 2**  
    It is recommended to follow the official SAM 2 project [here](https://github.com/facebookresearch/sam2?tab=readme-ov-file) to install both PyTorch and TorchVision dependencies. To install **the HiM2SAM version** of SAM 2 on a GPU machine, run:

```
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```

Download the SAM2.1 checkpoints:

```
cd checkpoints
./download_ckpts.sh
cd ..
```

2. **Install CoTracker 3**  
  HHiM2SAM uses the offline version of CoTracker 3 for pixel-level motion estimation. For more details about the model, please refer to [CoTracker 3](https://github.com/facebookresearch/co-tracker).

    The model can be easily loaded via torch.hub and will be automatically downloaded upon first use, requiring no additional setup.

3. **Other Packages**
```
pip install scipy jpeg4py lmdb
```

## 📁 Data Preparation

Prepare the dataset directories as shown below. LaSOT and LaSOT<sub>ext</sub> are supported. Download the official data from [here](http://vision.cs.stonybrook.edu/~lasot/):

```
data
  ├── LaSOT_extension_subset
  │   ├── atv/
  │   │   ├── atv-1/
  │   │   │   ├── full_occlusion.txt
  │   │   │   ├── groundtruth.txt
  │   │   │   ├── img
  │   │   │   ├── nlp.txt
  │   │   │   └── out_of_view.txt
  │   │   ├── atv-2/
  │   │   ├── atv-3/
  │   │   ├── ...
  │   ├── badminton
  │   ├── cosplay
  │   ...
  │   └── testing_set.txt
  └── LaSOT
      ├── airplane/
      │   ├── airplane-1/
      │   ├── airplane-2/
      │   ├── airplane-3/
      │   ├── ...
      ├── basketball
      ├── bear
      ...
      ├── training_set.txt  
      └── testing_set.txt  

```

## 🏃 Running Inference and Visualization

Run inference on LaSOT:

```
python scripts/main_inference.py 
```
Run inference on LaSOT<sub>ext</sub>:

```
python scripts/main_inference_ext.py 
```

By default, the code runs inference using the large model, it takes some time to evalate on the whole dataset, you can skip to next step and download our results for quick evaluation. 

Numerical results are saved under the `./result/` directory, and visualization outputs are stored in `./visualisation/`. The scripts can be easily adapted to other box-based VOT datasets with minimal modifications, just place the data under the `./dataset/` directory in the same format, and update the scripts accordingly.



## 📊 Running Evaluation

To reproduce the AUC, precision, and normalized precision metrics reported in the paper, our evaluation methodology aligns with those used in [SAMURAI](https://github.com/yangchris11/samurai) and the [VOT Toolkit](https://github.com/votchallenge/toolkit).

Please ensure that the tracking results are saved under the `./result/` directory. Our results can be downloaded from [here](https://drive.google.com/file/d/1IwUqABs91H14rpSGne1pDON9Dbsvm-AK/view?usp=sharing). You may add your own results and register your tracker in `scripts.py` for further comparison.

Run evaluation on LaSOT:
```
python lib/test/analysis/scripts.py > res_lasot.log
```
Run evaluation on LaSOT<sub>ext</sub>:
```
python lib/test/analysis/scripts_ext.py > res_lasot_ext.log
```
The evaluation results will be saved in the corresponding log files.

## 🎯 VOT Challenges

We provide wrapper scripts for evaluating **HiM2SAM** on the VOT challenges. For more information about the benchmarks, please refer to the official [VOT Toolkit](https://github.com/votchallenge/toolkit).  
Example configuration files are provided under the `./vot_utils/` directory for quick setup.

## 🧩 Demo on Custom Video

To run the demo with your custom video or frame directory, use the following examples:

**Note:** The `.txt` file contains a single line with the bounding box of the first frame in `x,y,w,h` format.

### Input is Video File

```
python scripts/demo.py --video_path <your_video.mp4> --txt_path <path_to_first_frame_bbox.txt>
```
 
### Input is Frame Folder
```
# Only JPG images are supported
python scripts/demo.py --video_path <your_frame_directory> --txt_path <path_to_first_frame_bbox.txt>
```

## 📝 Citation and Acknowledgment

We kindly ask you to cite our paper along with SAM 2 if you find this work valuable.


```
@misc{chen2025him2sam,
      title={HiM2SAM: Enhancing SAM2 with Hierarchical Motion Estimation and Memory Optimization towards Long-term Tracking}, 
      author={Ruixiang Chen and Guolei Sun and Yawei Li and Jie Qin and Luca Benini},
      year={2025},
      eprint={2507.07603},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.07603}, 
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

```

This repository is developed by Ruixiang Chen, and built on top of [SAM 2](https://github.com/facebookresearch/sam2?tab=readme-ov-file), [SAMURAI](https://github.com/yangchris11/samurai), [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM), and [CoTracker3](https://github.com/facebookresearch/co-tracker). The VOT evaluation code is modified from the [VOT Toolkit](https://github.com/votchallenge/toolkit).

Many thanks to the authors of these excellent projects for making their work publicly available.


