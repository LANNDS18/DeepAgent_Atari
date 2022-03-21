# DeepAgent
## Play atari game through DeepAgent

This repository implements Deep Q-Network (DQN) and several improvement algorithms (Double DQN, Dueling DQN, Prioritized Experience Replay) by DeepAgent framework for playing Atari game 'DemonAttack'.

![DemonAttackDQN](/assets/DemonAttackDQN.gif)
![PongDQN](/assets/PongDQN.gif)
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
@article{mnih2015humanlevel,
  added-at = {2015-08-26T14:46:40.000+0200},
  author = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski, Georg and Petersen, Stig and Beattie, Charles and Sadik, Amir and Antonoglou, Ioannis and King, Helen and Kumaran, Dharshan and Wierstra, Daan and Legg, Shane and Hassabis, Demis},
  biburl = {https://www.bibsonomy.org/bibtex/2fb15f4471c81dc2b9edf2304cb2f7083/hotho},
  description = {Human-level control through deep reinforcement learning - nature14236.pdf},
  interhash = {eac59980357d99db87b341b61ef6645f},
  intrahash = {fb15f4471c81dc2b9edf2304cb2f7083},
  issn = {00280836},
  journal = {Nature},
  keywords = {deep learning toread},
  month = feb,
  number = 7540,
  pages = {529--533},
  publisher = {Nature Publishing Group, a division of Macmillan Publishers Limited. All Rights Reserved.},
  timestamp = {2015-08-26T14:46:40.000+0200},
  title = {Human-level control through deep reinforcement learning},
  url = {http://dx.doi.org/10.1038/nature14236},
  volume = 518,
  year = 2015
}
```

## Author
Wuwei Zhang ([@LANNDS18](https://github.com/LANNDS18))
