## e. before running training/evaluation

download backbone checkpoints and create work dir

```bash
# create a work dir
mkdir work_dirs
mkdir ckpts
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json
cd ckpts
wget https://download.openmmlab.com/pretrain/third_party/resnet50_msra-5891d200.pth
```

expected proper directory structure:

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
├── RepVF (this repo)
├── ckpts
│   ├── resnet50_msra-5891d200.pth
└──  data
    └──  waymo
        ├── waymo_format
        ├── openlane_format
        ├── training_filtered.pkl
        ├── validation_filtered.pkl
        ├── cam_gt_filtered.bin
        ├── training_filtered_300.pkl
        ├── validation_filtered_300.pkl
        └── cam_gt_filtered_300.bin
```

For [SyncBN implementation](https://github.com/exiawsh/StreamPETR/blob/95f64702306ccdb7a78889578b2a55b5deb35b2a/tools/train.py#L222), modify mmdetection3d/tools/train.py to intergrate (this is only required for 30% data experiments for consistency): 

```python
...  
    model.init_weights()

    if cfg.get('SyncBN', False):
        import torch.nn as nn
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Using SyncBN")
  
    logger.info(f'Model:\n{model}')
....
```

we have also provided our modified version of [train.py](../tools/train.py) that can be used out of the box.

## f. training and evaluation command

assume you're under the workspace we have created, and here we use `mmdetection3d/projects/repvf/configs/rftr_r50_15p_1000.py` as example:

for debug or single-card:
```bash
python mmdetection3d/tools/train.py mmdetection3d/projects/repvf/configs/rftr_r50_15p_1000.py --work-dir work_dirs/rftr_r50_15p_1000/
```

for ddp training/evluation (4 is the gpu count):
```bash
# for training
bash mmdetection3d/tools/dist_train.sh mmdetection3d/projects/repvf/configs/rftr_r50_15p_1000.py 4 --work-dir work_dirs/rftr_r50_15p_1000/
# for evluation
bash mmdetection3d/tools/dist_test.sh mmdetection3d/projects/repvf/configs/rftr_r50_15p_1000.py work_dirs/rftr_r50_15p_1000/epoch_24.pth 4 --eval bbox
```
it would take about 3~4 days on 4*RTX4090 to train, we have also provided a 30% subset data version config `rftr_r50_20p_300_syncbn_flash_bs8.py` that would take about less than one day.

to resume training:
```bash
bash mmdetection3d/tools/dist_train.sh mmdetection3d/projects/repvf/configs/rftr_r50_15p_1000.py 4 --work-dir work_dirs/rftr_r50_15p_1000/ --resume-from work_dirs/rftr_r50_15p_1000/epoch_x.pth
```
environment variables:
| environment variable   | purpose                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| SAVE_FOR_VISUALIZATION | set to True to save predictions as numpy arrays                                                                           |
| SAVE_PLT_BBOX          | set to True to save visualizations                                                                                        |
| WANDB_API_KEY          | your [wandb](https://wandb.ai) api key; modify [default_runtime](../configs/_base_/default_runtime.py) to use tensorboard |