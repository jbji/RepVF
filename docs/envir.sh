git clone https://github.com/jbji/RepVF

# install pytorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# install toolkit for openlane dataset (our implementation)
cd RepVF/custom_modules/openlane_devkit
pip install -r requirements.txt
python setup.py develop
# go back to workspace
cd ../../../

# install prerequisites for mmdetection3d
pip install ConfigParser
pip install openmim
mim install mmcv-full==1.5.2
mim install mmdet==2.26.0
mim install mmsegmentation==0.29.1


# install mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git mmdetection3d
cd mmdetection3d
git checkout v1.0.0rc6
pip install -e .
cd ..

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

# fix a minor mmdet3d bug that causes issue in ddp training
sed -i 's/--local_rank/--local-rank/g' mmdetection3d/tools/train.py

# create soft link
ln -s $(pwd)/RepVF $(pwd)/mmdetection3d/projects/repvf
