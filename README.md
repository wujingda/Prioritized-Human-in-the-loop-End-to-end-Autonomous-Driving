# Prioritized Human-In-the-Loop (PHIL) RL for End-to-end Autonomous Driving

This repo is the implementation of the following paper:

**Prioritized Experience-Based Reinforcement Learning With Human Guidance for Autonomous Driving**
<br> [Jingda Wu](https://scholar.google.com/citations?user=icu-ZFAAAAAJ&hl=en), [Zhiyu Huang](https://scholar.google.com/citations?user=aLZEVCsAAAAJ&hl=en), [Wenhui Huang](https://scholar.google.co.kr/citations?user=Hpatee0AAAAJ&hl=zh-CN), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[arXiv]](https://arxiv.org/abs/2109.12516)**&nbsp;**[[Project Website]](https://wujingda.github.io/Prioritized-Human-in-the-loop-End-to-end-Autonomous-Driving/)**

## Getting started
1. Install the CARLA simulator, with referring to
https://carla.readthedocs.io/en/latest/start_quickstart/#a-debian-carla-installation

2. Install the dependent package
```shell
pip install -r requirements.txt
```
3. Run the training procedure of the left-turn task
```
python train_leftturn.py
```

## Using human guidance to improve RL performance (by joystick or keyboard)

![image](figures/human_guidance.gif)

The algorithms in this repo allow human subjects to provide guidance for improving RL training performance in the real-time (by joystick or keyboard)

## Training performance

<img src="figures/results.png" width = "500" height = "400" alt=" " align=center />

(a-b) Results in the left-turn scenario;
(c-d) Results in the traffic-jam scenario.

## Reference
If you find this repo to be useful in your research, please consider citing our work
```
@ARTICLE{9793564,
  author={Wu, Jingda and Huang, Zhiyu and Huang, Wenhui and Lv, Chen},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Prioritized Experience-Based Reinforcement Learning With Human Guidance for Autonomous Driving}, 
  year={2022},
  doi={10.1109/TNNLS.2022.3177685}}
```

## License
This repo is released under GNU GPLv3 License.


