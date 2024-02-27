# EPAN
Code of ACM MM 2023 paper [Emotion-Prior Awareness Network for Emotional Video Captioning](https://dl.acm.org/doi/abs/10.1145/3581783.3611726). 

### Prepare Data
Datasets can be downloaded from [here](https://mic.tongji.edu.cn/ce/70/c9778a183920/page.htm).

The CLIP feature of the MSVD, EmVidCap-S, and EmVidCap-L datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Pp0H2fEG3eZ4cOpK-Bz_JBts7qp9pwEN?usp=drive_link).
> I refer to [this repo](https://github.com/openai/CLIP) for extracting the CLIP features.

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
