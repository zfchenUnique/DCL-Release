# NSCL-PyTorch
Pytorch implementation for the Dynamic Concept Learner (DCL) on CLEVRER.

## Framework
<div align="center">
  <img src="_assets/framework.png" width="100%">
</div>

## Prerequisites
- Python 3
- PyTorch 1.0 or higher, with NVIDIA CUDA Support
- Other required python packages specified by `requirements.txt`. See the Installation.

## Installation
Install [Jacinle](https://github.com/vacancy/Jacinle): Clone the package, and add the bin path to your global `PATH` environment variable:

```
git clone https://github.com/vacancy/Jacinle --recursive
export PATH=<path_to_jacinle>/bin:$PATH
```

Clone this repository:

```
git clone https://github.com/zfchenUnique/DCL-Release-Private.git --recursive
```

Create a conda environment for NS-CL, and install the requirements. This includes the required python packages
from both Jacinle NS-CL. Most of the required packages have been included in the built-in `anaconda` package:

## Dataset preparation
- Download videos, video annotation,  questions and answers, and object proposals accordingly from the [official website](http://clevrer.csail.mit.edu/#)
- Transform videos into ".png" frames with ffmpeg.
- Organize the data as shown below.
    ```
    clevrer
    ├── annotation_00000-01000
    │   ├── annotation_00000.json
    │   ├── annotation_00001.json
    │   └── ...
    ├── ...
    ├── image_00000-01000
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   └── ...
    ├── ...
    ├── questions
    │   ├── train.json
    │   ├── validation.json
    │   └── test.json
    ├── proposals
    │   ├── proposal_00000.json
    │   ├── proposal_00001.json
    │   └── ...
    ```
## Fast Evaluation
- Download the extracted object trajectories from [google drive]().
- Git clone the dynamic model, download [the pretrained propNet models]() and make dynamic prediction by 
```
    git clone https://github.com/zfchenUnique/clevrer_dynamic_propnet.git
    cd clevrer_dynamic_propnet
    sh ./scripts/eval_fast_release.sh
```
- Download [the pretrained DCL model]()
- Answering questions. 
```
   sh scripts/script_test_prp_clevrer_qa.sh 0
```
- Re-organize questions for submission on [evalAI](https://eval.ai/web/challenges/challenge-page/667/overview).
```
   sh scripts/script_transform_answer_order.sh    
```

## Step-by-step Training

## Generalization to CLEVRER-Grounding
## Generalization to CLEVRER-Retrieval
## Extension to Tower Blocks
## Acknowledge
