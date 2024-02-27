# EPAN
Code of ACM MM 2023 paper [Emotion-Prior Awareness Network for Emotional Video Captioning]. 

### Prepare Data
Download datasets from [EmVidCap](https://mic.tongji.edu.cn/ce/70/c9778a183920/page.htm).

The video feature of the MSVD, EmVidCap-S, and EmVidCap-L datasets can be downloaded from .....

### Training
   ```
   $ python main.py
   ```
   You can change some hyperparameters by modifying `config.py`.

## Reference
If you find this repository useful, please consider citing our work:
```
@inproceedings{song2023emotion,
    author = {Song, Peipei and Guo, Dan and Yang, Xun and Tang, Shengeng and Yang, Erkun and Wang, Meng},
    title = {Emotion-Prior Awareness Network for Emotional Video Captioning},
    year = {2023},
    booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
    pages = {589–600}
}
```
