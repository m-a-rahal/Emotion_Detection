## Requirements
```
tensoflow
mtcnn
pandas
open-cv
```

## MobileNet pretrained on AffectNet model
The default model included in this repository is a modified version of Savchenko et al. [MobileNet model](https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/mobilenet_7.h5) pretrained on [AffectNet](http://mohammadmahoor.com/affectnet/).

```BibTex
@inproceedings{savchenko2021facial,
  title={Facial expression and attributes recognition based on multi-task learning of lightweight neural networks},
  author={Savchenko, Andrey V.},
  booktitle={Proceedings of the 19th International Symposium on Intelligent Systems and Informatics (SISY)},
  pages={119--124},
  year={2021},
  organization={IEEE},
  url={https://arxiv.org/abs/2103.17107}
}
```

```BibTex
@inproceedings{Savchenko_2022_CVPRW,
  author    = {Savchenko, Andrey V.},
  title     = {Video-Based Frame-Level Facial Analysis of Affective Behavior on Mobile Devices Using EfficientNets},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2022},
  pages     = {2359-2366},
  url={https://arxiv.org/abs/2103.17107}
}
```

```BibTex
@article{savchenko2022classifying,
  title={Classifying emotions and engagement in online learning based on a single facial expression recognition neural network},
  author={Savchenko, Andrey V and Savchenko, Lyudmila V and Makarov, Ilya},
  journal={IEEE Transactions on Affective Computing},
  year={2022},
  publisher={IEEE},
  url={https://ieeexplore.ieee.org/document/9815154}
}
```


## Face localization methods
Thanks to :
- Centeno, Iván de Paz for the MTCNN face detection implementation at [MTCNN face detection implementation for TensorFlow, as a PIP package.](https://github.com/ipazc/mtcnn)
- Tiwari, Shantnu for the HARR Cascade implementation/code at [Python for engineers](https://github.com/shantnu/PyEng)