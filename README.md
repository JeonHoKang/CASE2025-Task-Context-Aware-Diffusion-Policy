# Task-Context-Aware Diffusion Policy with Language Guidance for Multi-task Disassembly


Jeon Ho Kang, Sagar Joshi, Ruopeng Huang, and Satyandra K. Gupta

University of Southern California

![System Architecture](imgs/overview_system.png)

Baseline code for diffusion policy was derived from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)


## Dependencies

Create Conda Environment (Recommended) and run:


```bash
$ pip install requirements.txt
```

## Real Robot 

For all demonstrations, we used [KUKA IIWA 14 Robot](https://www.kuka.com/en-de/products/robot-systems/industrial-robots/lbr-iiwa)


## Training Your Own Policy


After loading your own zarr file or ours in [real_robot_network.py](real_robot_network.py)

```bash
$ python train_real.py
```

You can select or create your own [Config](config) file for training configuration


## Acknowledgement

+ Diffusion policy was adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)



For **Demonstration Purposes** use the regular folders that do not contain the **Real** tag

Those are for PushTEnv demonstration.

All  **Real** tages are for real robot implementation

However, [data_util.py](data_util.py) is shared amongst them.

Also checkout [modality](https://github.com/JeonHoKang/CASE2025-Task-Context-Aware-Diffusion-Policy/tree/modality) for main implementation for the time being.