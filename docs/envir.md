## a. environment configuration instructions

we assume that an environment has been created using **venv** or **anaconda/miniconda** and has been activated:

```bash
conda create -n repvf python=3.8 pip -y
conda activate repvf
```

also, you're assumed to have created a workspace folder and navigated into it, and if you haven't:

```bash
mkdir repvf_workspace
cd repvf_workspace
```

we believe that this is one of the best ways to correctly install mmdetection3d without worrying about cuda/pytorch versions mismatch.

---

### option 1. best practice

simply run the script:

```bash
bash RepVF/docs/envir.sh
```

---

### option 2. step by step guide

#### pytorch

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

we strongly suggest you check if it's been correctly installed simply by:

```python
import torch
print(torch.cuda.is_available())
print(torch.zeros(1).cuda())
```

and if not, please refer to our [troubleshoot guide](tourbleshoot.md).

#### devkit

install dataset devkit:

```bash
# install toolkit for openlane dataset (our implementation)
cd RepVF/custom_modules/openlane_devkit
pip install -r requirements.txt
python setup.py develop
# go back to workspace
cd ../../../
```

#### mmdet3d

install prerequisites (mim install might take a long time), if you use torch 1.13.1, you need to change mmcv-full version to 1.6.0.

```bash
# install prerequisites for mmdetection3d
pip install ConfigParser
pip install openmim
mim install mmcv-full==1.5.2
mim install mmdet==2.26.0
mim install mmsegmentation==0.29.1
```

install mmdet3d from source:

```bash
# install mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git mmdetection3d
cd mmdetection3d
git checkout v1.0.0rc6
pip install -e .
cd ..
```

#### final steps

install some other packages

```bash
# fix some extra issues
pip uninstall setuptools
pip install setuptools==59.5.0
pip3 install waymo-open-dataset-tf-2-11-0==1.5.1
pip3 install Pillow==9.2.0
pip install tensorflow==2.11.0
pip install yapf==0.32.0

# extra
# we suggest using wandb for logging
pip install wandb 
# you can use flash attention for faster training
pip install flash_attn==2.3.3
```

fix ddp bug in mmdet3d

```bash
# fix a minor mmdet3d bug that causes issue in ddp training
sed -i 's/--local_rank/--local-rank/g' mmdetection3d/tools/train.py
```

project soft link

```bash
# create soft link
ln -s $(pwd)/RepVF $(pwd)/mmdetection3d/projects/repvf
```

---

## b. after installation

the directory structure after the environment configuration is expected to be:

```bash
repvf_workspace
├── mmdetection3d
│   ├── projects
│       ├── repvf (symbolic link)
│       └── ...
│   ├── tools
│       ├── train.py (modified)
│       └── ...
│   └── ...
└── RepVF (this repo)

```

if anything is wrong with the environment, please refer to our [troubleshoot guide](troubleshoot.md).

---

Let's head over to the [data preparation](data.md) part, which gonna be a bit tricky and may take at least a week.
