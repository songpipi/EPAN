# EPAN
Code of ACM MM 2023 paper [Emotion-Prior Awareness Network for Emotional Video Captioning](https://dl.acm.org/doi/abs/10.1145/3581783.3611726). 

### Prepare Data
Datasets can be downloaded from [here](https://mic.tongji.edu.cn/ce/70/c9778a183920/page.htm). Please refer to [this repo](https://github.com/openai/CLIP) for extracting the CLIP features (ViT-B-32). I refer to [this repo](https://github.com/hobincar/pytorch-video-feature-extractor) for extracting the ResNet101 features, and [this repo](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) for extracting the 3D-ResNext101 features. You can also directly download the CLIP, ResNet101, and 3D-ResNext101 features of MSVD and EmVidCap videos from [here](https://drive.google.com/drive/folders/1Pp0H2fEG3eZ4cOpK-Bz_JBts7qp9pwEN?usp=drive_link), and put it into `data/<DATASET>/features<NETWORK>.hdf5`. Download the GloVe Embedding from [here](https://drive.google.com/file/d/12l40qeIi5zioX2WaVBLXD_oarc8zbaqG/view?usp=sharing) and locate it at `data/Embeddings/GloVe/GloVe_300.json`. 


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

## Acknowledgement
This code is built on [SGN](https://github.com/hobincar/SGN). Great thanks to them!
