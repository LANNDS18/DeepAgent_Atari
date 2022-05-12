# DeepAgent
## Play atari game through DeepAgent

This repository implements Deep Q-Network (DQN) and several improvement algorithms (Double DQN, Dueling DQN, Prioritized Experience Replay) by DeepAgent framework for playing Atari game 'DemonAttack'.

![DemonAttackDQN](/assets/DemonAttackDDDQN.gif)
![PongDQN](/assets/PongDDDQN.gif)
![EnduroDDDQN](/assets/EnduroDDDQN.gif)

## Requirements

* Python 3.8+ (Native support for Apple Silicon) 

* tensorflow-macos 2.0+ (for macOS)

```
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
```

* OpenAI Gym 

`pip install 'gym[all]'`
* opencv-python 

`pip install opencv-python`

tensorflow 2.0+ is initially running on CPU, tensorflow-gpu and tensorflow-metal is also supported。

`python -m pip install tensorflow-metal`

## Architecture
```
├── DeepAgent
│   ├── __init__.py
│   ├── agents
│   │   ├── __init__.py
│   │   ├── d3nPER.py
│   │   ├── doubleDQN.py
│   │   └── dqn.py
│   ├── interfaces
│   │   ├── __init__.py
│   │   ├── ibaseAgent.py
│   │   ├── ibaseBuffer.py
│   │   ├── ibaseConfig.py
│   │   └── ibaseNetwork.py
│   ├── networks
│   │   ├── __init__.py
│   │   ├── dqnNet.py
│   │   ├── duelingNet.py
│   │   └── duelingResNet50.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── buffer.py
│   │   ├── common.py
│   │   ├── game.py
│   │   └── offPolicyWrapper.py
│   ├── client.py
│   └── visualization.py
...
```

## Installation

Clone the repository:

`git clone https://github.com/LANNDS18/DeepAgent_DemonAttack.git`

Install dependencies:

`python setup.py install` or `pip install -e .`


## BibTeX

```
@article{mnih2013playing,
  title={Playing atari with deep reinforcement learning},
  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin},
  journal={arXiv preprint arXiv:1312.5602},
  year={2013}
}

@article{journals/nature/MnihKSRVBGRFOPB15,
  added-at = {2021-01-06T14:52:16.000+0100},
  author = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin A. and Fidjeland, Andreas and Ostrovski, Georg and Petersen, Stig and Beattie, Charles and Sadik, Amir and Antonoglou, Ioannis and King, Helen and Kumaran, Dharshan and Wierstra, Daan and Legg, Shane and Hassabis, Demis},
  biburl = {https://www.bibsonomy.org/bibtex/27c6f54424f0d4672eae5411091b5bd95/frankyanpan},
  ee = {https://www.wikidata.org/entity/Q27907579},
  interhash = {eac59980357d99db87b341b61ef6645f},
  intrahash = {7c6f54424f0d4672eae5411091b5bd95},
  journal = {Nature},
  keywords = {Reinforcementlearning deeplearning},
  number = 7540,
  pages = {529-533},
  timestamp = {2021-01-06T14:52:16.000+0100},
  title = {Human-level control through deep reinforcement learning.},
  url = {http://dblp.uni-trier.de/db/journals/nature/nature518.html#MnihKSRVBGRFOPB15},
  volume = 518,
  year = 2015
}

@article{schaul2015prioritized,
  title={Prioritized experience replay},
  author={Schaul, Tom and Quan, John and Antonoglou, Ioannis and Silver, David},
  journal={arXiv preprint arXiv:1511.05952},
  year={2015}
}


@article{ReplayBuffer,
  title={Self-improving reactive agents based on reinforcement learning, planning and teaching},
  author={Longxin Lin},
  journal={Machine Learning},
  year={2004},
  volume={8},
  pages={293-321}
}

@misc{doubleDQN,
  doi = {10.48550/ARXIV.1509.06461},
  url = {https://arxiv.org/abs/1509.06461},
  author = {van Hasselt, Hado and Guez, Arthur and Silver, David},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Deep Reinforcement Learning with Double Q-learning},
  publisher = {arXiv},
  year = {2015},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@misc{dueling,
  doi = {10.48550/ARXIV.1511.06581},
  url = {https://arxiv.org/abs/1511.06581},
  author = {Wang, Ziyu and Schaul, Tom and Hessel, Matteo and van Hasselt, Hado and Lanctot, Marc and de Freitas, Nando},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Dueling Network Architectures for Deep Reinforcement Learning},
  publisher = {arXiv},
  year = {2015},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@misc{multistep-learning,
  doi = {10.48550/ARXIV.1901.07510},
  url = {https://arxiv.org/abs/1901.07510},
  author = {Hernandez-Garcia, J. Fernando and Sutton, Richard S.},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target},
  publisher = {arXiv},
  year = {2019},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@misc{baselines,
  author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai and Zhokhov, Peter},
  title = {OpenAI Baselines},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/openai/baselines}}
}
```

## Author
Wuwei Zhang ([@LANNDS18](https://github.com/LANNDS18))
