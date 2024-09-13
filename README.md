<h1 align="center">Microfluidic Flow Programming for Complex Semantic Flow Profiles </br>Via Efficient Receptive Field Augmentation of Convolutional Neural Networks</h1>
<h4 align="center"><a href="https://doi.org/10.5281/zenodo.13363709"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13363709.svg" alt="DOI"></a></h4>
</h4>
<div align="center">
Zhenyu Yang<sup>1,2,7</sup>, Zhongning Jiang<sup>3,7</sup>, Haisong Lin<sup>4,5</sup>, Edmund Y. Lam<sup>6,8</sup>, Hayden Kwok-Hay So<sup>6,8</sup>, Ho Cheung Shum<sup>1,2,8</sup>
</div>
<div align="center">
  <sup>1</sup>Advanced Biomedical Instrumentation Centre, Hong Kong, China. <br>
  <sup>2</sup>Department of Mechanical Engineering, The University of Hong Kong, Hong Kong, China.<br>
  <sup>3</sup>Department of Biomedical Engineering, City University of Hong Kong, Hong Kong, China.<br>
  <sup>4</sup>School of Engineering, Westlake University, Hangzhou, China.<br>
  <sup>5</sup>Research Center for Industries of the Future, Westlake <br>University, Hangzhou, China.
  <sup>6</sup>Department of Electrical and Electronic Engineering, The University of Hong Kong, Hong Kong, China.<br>
  <sup>7</sup>These authors contributed equally: Zhenyu Yang, Zhongning Jiang.<br>
  <sup>8</sup>These authors are the corresponding authors. e-mail: elam@eee.hku.hk, hso@eee.hku.hk, ashum@hku.hk.
</div>

## Introduction and setup
We introduce a framework for prediction of the flow transformation tensor induced by 'zigzag' obstacles in microchannels, heuristic design of microchannel for target output flow profiles, and  automatic searching of micochannel designs to output desired flow profiles. The framework is detailed in the manuscript *"Microfluidic Flow Programming for Complex Semantic Flow Profiles Via Efficient Receptive Field Augmentation of Convolutional Neural Networks"*.

To conduct similar studies as those presented in the manuscript, start by cloning this repository via
```
git clone https://github.com/ZhenyuYuYang/FlowProgrammer.git
```

The dataset for model training and model checkpoint are provided on [Zenodo](https://zenodo.org/records/13363709). Unzip the dataset `dataset.zip` into the `../dataset` folder and the pre-trained model checkpoint `checkpoint.zip` in the `../log` folder. To automatic search the microchannel design for a specific output flow profile, put the flow porfile image into `../auto_search/target_profile.png`.

The complete directory tree of the codes and data is shown below. 
```
..
├── dataset
│   ├── obs_img_train
│   │   └── ...
│   ├── obs_img_valid
│   │   └── ...
│   ├── tt_train
│   │   └── ...
│   └── tt_valid
│       └── ...
├── log
|   └── CEyeNet
|       ├── CEyeNet
|       └── ...
├── auto_search
|   └── target_profile.png
└── FlowProgrammer
    └── ...
```

To run the codes, first install the dependent packages listed in `./requirements.txt`

To train a CEyeNet model, run
```
python train.py --model CEyeNet --profile_size 200
```
To launch the GUI for heuristic design with the pretrained CEyeNet Checkpoint, run
```
python launch_gui.py
``` 
To automatically search the microchannel design, run
```
python search.py
```


To experiment with different training and GUI configurations, use option flags defined in `./config/config.py` 

For more details, please refer to the manuscript. 