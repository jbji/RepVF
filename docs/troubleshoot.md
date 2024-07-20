### environment troubleshooting

#### pytorch
```python
import torch
print(torch.cuda.is_available())
print(torch.zeros(1).cuda())
```

if anything goes wrong, it's probably that torch2 is incompatible with your cuda driver. (For example, On 2080Ti and CUDA11.7). You can try torch 1.13.1:

```bash
# Warning: you will PROBABLY get different results from ours.
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```


#### mmcv

If you encounter any of these problems below while training/evaluating:

- mmcv.ext_ does not exist
- DDP multi-card evaluation does not work, saying 'TypeError: object of type 'DataContainer' has no xxx' ([reference](https://github.com/open-mmlab/mmdetection/issues/1501))

it's because the mmcv is not correctly setup. mmcv heavily relies on your torch, just install another version.

> As we have tested,** mmcv 1.5.2 works with torch 2.0. And for torch 1.13.1, mmcv 1.6.0 works.** (1.5.2&1.6.2 not compatible with torch 1.13.1, 1.5.3 does not support ddp eval)

```bash
mim install mmcv-full==1.6.0
```
If you have unfortunately encountered this problem and can only use this workaround, you need to re-install ALL mm-series packages.

#### conda exported yml
we also provide an [exported conda environment yml](envir.yml) to help you determine if minor package version discrepancies exist.

