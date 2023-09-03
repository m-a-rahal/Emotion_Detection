## API global logic
![](attachements/API%20genereal%20structure%20—%20ENGLISH.drawio.png)

## See the models in action
To demonstrate the performance of the models we trained/fine tuned, see [This Collab notebook](https://colab.research.google.com/drive/1Rx6vD0kzd29Sp6wZON6-b3TzWp3-mJe4?usp=sharing)

## Requirements
```
tensoflow
mtcnn
pandas
open-cv
[jupyter-lab] # useful to display data — mostly pandas tables styling
```


# Credits
This work relies on other repositories :
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

## Face localization
- see [emotion_detection/face_localization/README.md](emotion_detection/face_localization/README.md)
