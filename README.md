# Deterministic-Policy-Gradient-Cell-Migration
This code accompanies "[A deep reinforcement learning model based on deterministic policy gradient for collective neural crest cell migration](https://arxiv.org/abs/2007.03190)", which appeared on *arXiv.org* in 2020.

## About
This project uses a deep reinforcement learning algorithm in association with a particle dynamics model to train agents to model cell interactions, such as co-attraction and contact-inhibition of locomotion, which is essential for understanding collective cell migration. We apply the deep deterministic policy gradient algorithm in association with a particle dynamics simulation environment to train agents to determine the migration path. Because of the different migration mechanisms of leader and follower neural crest cells, we train two types of agents (leaders and followers) to learn the collective cell migration behavior. For a leader agent, we consider a linear combination of a global task, resulting in the shortest path to the target source, and a local task, resulting in a coordinated motion along the local chemoattractant gradient. For a follower agent, we consider only the local task. First, we show that the self-driven forces learned by the leader cell point approximately to the placode, which means that the agent is able to learn to follow the shortest path to the target. We show that the overall leader cell migration for the case with co-attraction is slower because the co-attraction mitigates the source-driven effect. In addition, we find that the leader and follower agents learn to follow a similar migration behavior as in experimental observations. Overall, our proposed method provides useful insight on how to apply reinforcement learning techniques to simulate collective cell migration.

# *How to use this code*
## Setup
*Note:* This code was designed to be used with Python 3.6.13, and is not compatible with later versions.

To install this project's package dependencies, please run the following command:

    pip install -r requirements.txt

## Train
This project provides framework to train an agent for cell interactions:

To train this model, run the following command:
```
python Moving_Source_DDPG_local.py
```

## Test
A built-in testing script can be used to assess generalization capabilities, illustrating the optimal policy learned by an agent during training. A pre-trained policy has been included in the model folder, which can be tested for reference. To run the testing framework, you can use the following command:

 python Moving_Source_DDPG_test.py

## Configure
This code was developed with many customizable parameters to facilitate its application to different evacuation environments. You can modify the following parameters in the source code to appropriately configure your training:

- In file `Continuum_Cellspace_Moving_Source.py`:

    | Argument                 | Type     | Default    | Description                                                               |
    | ------------------------ | -------- | ---------- | ------------------------------------------------------------------------- |
    | door_size                | float    | 0.5        | Size of door                                                              |
    | agent_size               | float    | 0.5        | Size of agent (particle)                                                  |
    | reward                   | float    | -0.1       | Reward                                                                    |
    | end_reward               | float    | 0          | End_reward                                                                |
    | dis_lim                  | float    | 0.5        | Direct distance from the center of the agent to the center of the exit.   |
    | action_force             | float    | 1.0        | Unit action force                                                         |
    | desire_velocity          | float    | 2.0        | Desire velocity                                                           |
    | relaxation_time          | float    | 0.5        | Relaxation_time                                                           |
    | delta_t                  | float    | 0.01       | Time step                                                                 |
    | xmax                     | float    | 50.0       | X-direction size of the cell space                                        |
    | ymax                     | float    | 50.0       | Y-direction size of the cell space                                        |
    | cfg_save_step            | int      | 5          | Time interval for saving Cfg file                                         |

- In file `Moving_Source_DDPG_local.py`:

    | Argument                 | Type     | Default    | Description                                                  |
    | ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
    | num_episodes             | int      | 100000     | Max number of episodes to learn from                         |
    | max_steps                | int      | 1000       | Max steps in an episode                                      |
    | gamma                    | float    | 0.         | Future reward discount                                       |
    | memory_size              | int      | 10000      | Memory capacity                                              |
    | batch_size               | int      | 50         | Batch size                                                   |
    | explore_start            | float    | 1.0        | Exploration probability at start                             |
    | explore_stop             | float    | 0.1        | Minimum exploration probability                              |
    | num_agent                | int      | 1          | How many agents for the training                             |
    | update_target_every      | int      | 1          | Target update frequency                                      |
    | tau                      | float    | 0.01       | Target update factor                                         |
    | save_step                | int      | 1000       | Steps to save the model                                      |
    | train_step               | int      | 50         | Steps to train the model                                     |
    | learning_rate            | float    | 1e-04      | Learning rate to use                                         |
    | Cfg_save_freq            | int      | 1000       | Cfg save frequency (episode)                                 |

- In file `Moving_Source_DDPG_test.py`:

    | Argument                 | Type     | Default    | Description                                                  |
    | ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
    | test_episodes            | int      | 1          | Max number of episodes to test                               |
    | Number_Agent             | int      | 40         | How many agents to evacuate from the cell space during test  |
    | max_steps                | int      | 1000       | Max steps in an episode                                      |
    | Cfg_save_freq            | int      | 1          | Cfg save frequency (episode)                                 |
    | cfg_save_step            | int      | 2          | Cfg save frequency (step)                                    |
    | arrow_len                | float    | 0.07       | The arrow length in optimal distribution figure              |

## Cite

To reference this work, please use the following:
```
@misc{zhang2020deep,
      title={A deep reinforcement learning model based on deterministic policy gradient for collective neural crest cell migration}, 
      author={Yihao Zhang and Zhaojie Chai and Yubing Sun and George Lykotrafitis},
      year={2020},
      eprint={2007.03190},
      archivePrefix={arXiv},
      primaryClass={q-bio.CB}
}
```

