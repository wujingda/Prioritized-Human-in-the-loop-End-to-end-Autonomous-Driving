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

## Provide human guidance to improve RL training performance (by joystick or keyboard)

![figures/human_guidance.gif]()

## Training performance of different algorithms

<img src="figures/results.png" width = "500" height = "400" alt=" " align=center />




