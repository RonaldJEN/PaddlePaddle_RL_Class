# 基于PARL的终极复现项目
## 复现项目 ##
- [BipedalWalkerHardcore-v2](https://github.com/RonaldJEN/PaddlePaddle_RL_Class/blob/master/Game_Reproduction/star2_train.py)
    + [环境-GYM Box2D](https://github.com/openai/gym/tree/07e0c98f8e8e18c5197fab7ff74635f5b0cb2662/gym/envs/box2d)
    + 难度:二星
    + 算法:SAC

- [Quadrotor <velocity_control>](https://github.com/RonaldJEN/PaddlePaddle_RL_Class/blob/master/Game_Reproduction/star3_train.py)
    + [环境-RLSchool](https://github.com/PaddlePaddle/RLSchool/tree/master/rlschool/quadrotor)
    + 难度:三星
    + 算法:DDPG
## 安装 ##
```
pip install paddlepaddle-gpu==1.8.2.post107
pip install parl==1.3.1
pip install rlschool==0.3.1 
pip install gym==0.15.4
```
## 训练 ##
```
#训练BipedalWalkerHardcore-v2
python star2_train.py

#训练Quadrotor <velocity_control>
python star3_train.py
```
