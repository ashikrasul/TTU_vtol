# RRAAA-Sim
### Robust and Resilient Autonomy for Advanced Air Mobility

This repository is a work-in-progress SITL simulator for autonomous air taxis.

<table>
  <tr>
    <td>
      <img src="doc/images/functional_diagram.jpeg" alt="Functional Diagram" width="400"/>
      <p align="center">Functional Diagram</p>
    </td>
    <td>
      <img src="doc/images/current_state.jpeg" alt="Current State" width="400"/>
      <p align="center">Current State</p>
    </td>
  </tr>
</table>


### Prereqs

- Ubuntu 20.04.6 LTS (Other versions untested, but should work.)
- CUDA GPU for Pytorch and Unreal Engine, e.g., NVIDIA GeForce RTX series.
- Install Docker, see [doc/tools_installation.md](doc/tools_installation.md) for detailed instructions.


### Usage
Clone this repository with submodules ([jax_guam](https://github.com/oswinso/jax_guam)). Access to [jax_guam](https://github.com/oswinso/jax_guam) is separate from this repository and is required to use this simulator.

```bash
git clone https://github.com/CPS-IL/rraaa-sim.git --recurse-submodules
cd rraaa-sim
```

Build images and launch containers. On the first run, container image build can take a long time, 1 hour +.

```bash
cd docker
docker compose up
```
Wait till the docker compose finishes, then in a new terminal
```bash
cd docker
./enter_console.sh
```

The above will take us to simulator container console.

```bash
catkin_make
source devel/setup.bash
roslaunch rraaa run.launch
```

### Contact
  - [Hyung-Jin Yoon](mailto:stargaze221@gmail.com)
  - [Ayoosh Bansal](mailto:ayooshb2@illinois.edu)
  - [Oswin So](mailto:oswinso@mit.edu) : JAX GUAM
  - [Yang Zhao](mailto:yz107@illinois.edu) : GUAM Carla Interface, Adaptive Control
