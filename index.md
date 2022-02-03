## Prioritized Experience-based Reinforcement Learning with Human Guidance: Methodology and Application to Autonomous Driving

[Jingda Wu](https://scholar.google.com/citations?user=icu-ZFAAAAAJ&hl=en), [Zhiyu Huang](https://scholar.google.com/citations?user=aLZEVCsAAAAJ&hl=en), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 

[AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)

## Abstract
Reinforcement learning requires skillful definition and remarkable computational efforts to solve optimization and control problems, which could impair its prospect. Introducing human guidance into reinforcement learning is a promising way to improve learning performance. In this paper, a comprehensive human guidance-based reinforcement learning framework is established. A novel prioritized experience replay mechanism that adapts to human guidance in the reinforcement learning process is proposed to boost the efficiency and performance of the reinforcement learning algorithm. To relieve the heavy workload on human participants, a behavior model is established based on an incremental online learning method to mimic human actions. We design two challenging autonomous driving tasks for evaluating the proposed algorithm. Experiments are conducted to access the training and testing performance and learning mechanism of the proposed algorithm. Comparative results against the state-of-the-arts suggest the advantages of our algorithm in terms of learning efficiency, performance, and robustness.

## Results

The visualization results of different driving policies are presented.

### Left-turn Scenario

| Scene 1 | Scene2 |
|:-------------------------------------:|:---------------------------------------:|
| <video muted controls width=380> <source src="./src/LeftTurn1.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/LeftTurn2.mp4"  type="video/mp4"> </video> |


### Congestion Road Scenario

| Expert Prior RL | Behavior Cloning |
|:---------------:|:----------------:|
| <video muted controls width=380> <source src="./src/CongestionRoad1.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/CongestionRoad2.mp4"  type="video/mp4"> </video> | 


## Contact

If you have any questions, feel free to contact us (jingda001@e.ntu.edu.sg).
