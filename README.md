# HyperComm Implementation
This is the codebase for "HyperComm: Hypergraph-based Communication in Multi-Agent Reinforcement
Learning"

## Requirements
* OpenAI Gym
* PyTorch (GPU version recommended)
* [SMAC](https://github.com/oxwhirl/smac)
* Predator-Prey and Traffic Junction [Environments](https://github.com/apsdehal/ic3net-envs)



## Installation instructions

Set up StarCraft II and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.6.2.69232 into the 3rdparty folder and copy the maps necessary to run over. You may also need to set the environment variable for SC2:

```bash
export SC2PATH=[Your SC2 folder like /abc/xyz/3rdparty/StarCraftII]
```

Predator-Prey and Traffic Junction (from IC3Net)
  ```
  cd envs/ic3net-envs
  python setup.py develop
  ```


## Training HyperComm
-Run `python main.py --help` to check all the options.  
-Use `--comm_mask_zero` to block the communication.
* Predator-Prey easy (5-agent, 10 * 10 grid) scenario:
  `sh run_pp_easy.sh`
* Predator-Prey medium  (10-agent, 10 * 10 grid) scenario:
  `sh run_pp_medium.sh`
* Predator-Prey hard  (25-agent, 20 * 20 grid) scenario:
  `sh run_pp_hard.sh`
* Traffic-Junction easy (5-agent) scenario:
  `sh run_tj_easy.sh`
* Traffic-Junction medium (10-agent) scenario:
  `sh run_tj_medium.sh`
* Traffic-Junction hard (20-agent) scenario:
  `sh run_tj_hard.sh`