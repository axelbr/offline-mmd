# Scalable Offline Reinforcement Learning for Mean Field Games

Official code for [Offline Munchausen Mirror Descent](https://arxiv.org/abs/2410.17898).

Dependency installation: ```pip install -r requirements.txt```.

The main file for training is `src/train.py`.

Note that in order to train Off-MMD, you need to generate the datasets first.
We provide the policy checkpoints to generate the datasets via `src/collect_dataset.py`.

Configuration files are located in `config`.
