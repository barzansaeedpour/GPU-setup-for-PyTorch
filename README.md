# Pytorch on GPU

This repository is a tutorial to run Pytorch on GPU:

Python Version: 3.10

1) install visual studio community edition
- https://visualstudio.microsoft.com/vs/community/

2) install  CUDA 11.8
- https://developer.nvidia.com/cuda-11-8-0-download-archive

3) install compatible version of cudnn to CUDA 11.x
- https://developer.nvidia.com/rdp/cudnn-download

Extract the files in a folder like:

> C:\tools\

should be like this:

> C:\tools\cudnn-windows-x86_64-8.9.5.30_cuda11-archive

4) add the environment variables to system path
the following varables must exist in the environment variables:

- Varible name: CUDA_PATH 
- Variable value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

- Varible name: CUDA_PATH_V11_8
- Variable value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

- path:

    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\include
    - C:\Program Files\NVIDIA Corporation\Nsight Compute 2022.3.0\
    - C:\tools\cudnn-windows-x86_64-8.9.5.30_cuda11-archive\bin
    - C:\tools\cudnn-windows-x86_64-8.9.5.30_cuda11-archive\include

5) install pytorch compatible to cuda version:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

6) Run test.py to see the result