# RRAAA-Sim
### Robust and Resilient Autonomy for Advanced Air Mobility

This repository is an extension to AirTaxiSim which is based on CARLA UE4. In this work the simulator is updated to CARLA UE5.

<table>
  <tr>
    <td>
      <img src="doc/images/landing_simu.png" alt="Landing in Carla UE5" width="800"/>
      <p align="center">Landing in Carla UE5</p>
    </td>
  </tr>
</table>




### Prereqsites: 

- Ubuntu 20.04.6 LTS or 22.04.4 LTS (Other versions untested, but should work.)
- CUDA GPU for Pytorch and Unreal Engine, e.g., NVIDIA GeForce RTX series.
- Install Docker and Nvidia Docker Toolkit, see [doc/tools_installation.md](doc/tools_installation.md) for detailed instructions.
- Python packages
```bash
python3 -m pip install loguru
```

### Quick Start
Clone this repository with submodules.

```bash
git clone -b ue5_simulator --single-branch --recurse-submodules https://github.com/ashikrasul/TTU_vtol.git
cd ttu_vtol
#If you already cloned without submodules: 
git submodule update --init --recursive
#docker sudo access:
sudo chmod 666 /var/run/docker.sock
#docker access to host display
xhost +local:docker
#Run the simulator: It will build the containers and initiate perception based landing. 
python3 rraaa.py configs/single-static.yml
```


### Contact
  - [Ashik E Rasuk](mailto:ashik.rasul@outlook.edu)
  - [Hyung-Jin Yoon](mailto:stargaze221@gmail.com)


