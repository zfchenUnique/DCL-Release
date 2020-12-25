# NSCL-PyTorch
Pytorch implementation for the Dynamic Concept Learner (DCL) on CLEVRER.

## Framework
<div align="center">
  <img src="_assets/framework.png" width="100%">
</div>

- Python 3
- PyTorch 1.0 or higher, with NVIDIA CUDA Support
- Other required python packages specified by `requirements.txt`. See the Installation.

## Installation
## Prerequisites

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
--Download videos, video annotation,  questions and answers, and object proposals accordingly from the [official website](http://clevrer.csail.mit.edu/#)
--Transform videos into ".png" frames with ffmpeg. 
