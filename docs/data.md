## c. data preparation

in this part, you will
1) download openlane and waymo data
2) compile waymo cli and unpack waymo tfrecords
3) put them together
4) generate pkl and bin for training and evaluation

### data downloads
You can download openlane-v1 dataset from [here](https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md) or with opendatalab cli at [opendatalab](https://opendatalab.com/OpenDriveLab/OpenLane); (you will need about 100~200GB of storage)

Waymo Open Dataset v1.4.2 can be downloaded [here](https://waymo.com/open/licensing/?continue=%2Fopen%2Fdownload%2F) and you can use gcloud cli to download it. (you will need about 1TB of storage)

you can put them wherever you like, 
### openlane dataset

just download the data and unzip them and there should be nothing you need to specifically take care of, their file format should look like:
```bash
openlane
├── images # openlane only contain front view so has no suffix
├── lane3d_1000
└── ...
```

### waymo open dataset

#### waymo metrics cli/toolkit
we have precompiled a waymo metrics computation cli in RepVF/tools/waymo_utils, you may want test it with:
```bash
RepVF/tools/waymo_utils/compute_detection_metrics_main
```
if it won't work, please refer to this [guide](waymo_toolkit.md) for compiling one.


#### unpack dataset
here we need to convert Waymo 1.4.2 dataset consisting of tfrecord files into individual files, make sure you have enough space left on your disk (about identical size to waymo_format).

Install some package first(According to https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb)
```bash
sudo apt-get install openexr
sudo apt-get install libopenexr-dev
```
Then Run the unpack script, and wait:
```
python RepVF/tools/waymo_adapt_tools.py --load_dir path-to-waymo/waymo_format --save_dir path-to-save/openlane_format --workers 16 --verbose True
```
in our practice, you can use up to 80 workers or more.

before unpack conversion:
```bash
waymo
└── waymo_format
    ├── training
    │   ├── segment-xxx.tfrecord
    │   └── ...
    └── validation
        ├── segment-xxx.tfrecord
        └── ...
```
after unpack conversion:
```bash
waymo
└── openlane_format
    ├── images_i # (i=0,1,2,3,4), Waymo has 5 views of camera instead of 6
    │   ├── training
    │       ├── segment-xxx
    │           ├── (timestamp).jpg
    │           └──  ...
    │       ├── segment-xxx
    │       └──  ...
    │   └──  validation
    ├── velodyne
    │   ├── training (timestamp.bin)
    │   └──  validation
    └── detection3d_1000
        ├── training
        └──  validation

```

### putting together

we suggest symbolic link all the data in the following place to ease specifying data root:

```bash
repvf_workspace
├── mmdetection3d
├── RepVF
└──  data
    └──  waymo
        ├── waymo_format
        ├── openlane_format # link
            ├── images_i
            ├── detection_1000
            ├── lane3d_1000 # link
            └── lane3d_300 # link
        └──  ... # other pkl or bin files that we will generate
```

to achieve this, you need to
1) copy or softlink openlane format waymo data as data/waymo/openlane_format
2) copy or softlink lane3d_1000 and lane3d_300 under data/waymo/openlane_format/


### generate pkl and bin for joint training
for the complete set, to generate .pkl and .bin while filtering empty samples, this by default combines **detection_1000** with **lane3d_1000**:
```bash
python RepVF/tools/waymo_adapt_tools/generate_pkl_waymo.py --workers 80 --filter_empty_gt --suffix filtered
python RepVF/tools/waymo_adapt_tools/create_waymo_gt_bin.py --pkl-path data/waymo/openlane_format/validation_filtered.pkl --bin-name cam_gt_filtered.bin
```
for the 30% subset, **lane3d_300** is used:
```bash
python RepVF/tools/waymo_adapt_tools/generate_pkl_waymo.py --workers 80 --filter_empty_gt --suffix filtered_300 --lane3d lane3d_300
python RepVF/tools/waymo_adapt_tools/create_waymo_gt_bin.py --pkl-path data/waymo/openlane_format/validation_filtered_300.pkl --bin-name cam_gt_filtered_300.bin 
```

if you need no filteration:
```bash
python RepVF/tools/waymo_adapt_tools/generate_pkl_waymo.py --workers 80
python RepVF/tools/waymo_adapt_tools/create_waymo_gt_bin.py
```

the .pkl file is for mmdetection3d dataset and the .bin file is required by waymo metrics computation.


### expected directory structure after this step

```bash
repvf_workspace
├── mmdetection3d
├── RepVF
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

## d. demo data
you can download our demo data [here](to-be-uploaded), which contains one segment for debug purpose.