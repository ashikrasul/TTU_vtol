#!/usr/bin/env python3

import carla
import functools as ft
import ipdb
import numpy as np
import os
import rospy

from loguru import logger

from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Twist

import jax
import jax.random as jr
import jax.numpy as jnp
import jax.tree_util as jtu
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.guam_types import RefInputs
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs_1, lift_cruise_reference_inputs_2
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format

from plot.guam_plot_batch_with_ref import plot_batch_with_ref
from node_static_landing_planner import landing_reference_inputs

# reference_type:
# 1 = lift_cruise_reference_inputs_1
# 2 = lift_cruise_reference_inputs_2
# 3 = landing_reference_inputs
# 4 = subscriber

# Functions
def get_vehicle_type() -> str:
    """
    Get the vehicle type specified in the launch file. Check the vehicle type is valid.
    """
    # TODO: export the list of supported vehicle to some config file.
    supported_vehicles = ["guam", "minihawk"]
    vehicle = rospy.get_param("vehicle").lower()
    assert vehicle in supported_vehicles, f"Vehicle type {vehicle} is not supported."
    return vehicle

# Classes
class GUAM_Node:
    def __init__(self, pos, vel):
        jax_use_cpu()
        jax_use_double()
        set_logger_format()

        # sim params
        self.reference_type = rospy.get_param('/guam/reference_type', default=4)
        self.skip_sleep =  rospy.get_param('/guam/skip_rospy.sleep', default=False)
        self.plot_switch = rospy.get_param('/guam/plot', default=False)
        self.save_video = rospy.get_param('/guam/save_video', default=False)

        # jax-guam related params
        self.guam = None
        self.guam_reference = None
        self.guam_reference_sub = None

        self.initial_position = [pos.x, pos.y, pos.z]
        self.initial_velocity = [vel.x, vel.y, vel.z]

        logger.info("Constructing guam node...")
        rospy.init_node('guam')
        self.guam_pose_pub = rospy.Publisher('/guam/pose', PoseStamped, queue_size=1)
        self.guam_pose_msg = PoseStamped()
        self.guam_vel_pub = rospy.Publisher('/guam/velocity', Twist, queue_size=1)
        self.guam_vel_msg = Twist()
        match self.reference_type:
            case 1:
                logger.info("Reference type is lift_cruise_reference_inputs_1")
                self.guam_reference = lift_cruise_reference_inputs_1(0)
            case 2:
                logger.info("Reference type is lift_cruise_reference_inputs_2")
                self.guam_reference = lift_cruise_reference_inputs_2(0)
            case 3:
                logger.info("Reference type is landing_reference_inputs")
                self.guam_reference = landing_reference_inputs(0)
            case 4:
                logger.info("Reference type is subscriber")
                logger.info("Subscribing to planner for trajectory reference...")
                self.guam_reference_sub = rospy.Subscriber('/planner/reference',
                                                            JointTrajectoryPoint,
                                                            self.guam_reference_callback)
                self.guam_reference_init()
                while self.guam_reference is None:
                    pass

    def guam_reference_init(self):
        # adjust for guam units and frame
        pos_des = jnp.array(self.initial_position) * jnp.array([3.28084, 3.28084, -3.28084])
        vel_bIc_des = jnp.array(self.initial_velocity) * jnp.array([3.28084, -3.28084, -3.28084])
        chi_des = 0
        chi_dot_des = 0
        self.guam_reference = RefInputs(
            Vel_bIc_des=vel_bIc_des,
            Pos_des=pos_des,
            Chi_des=chi_des,
            Chi_dot_des=chi_dot_des,
        )

    def guam_reference_callback(self, msg):
        # convert meter to feet
        pos_des = jnp.array(msg.positions) * jnp.array([3.28084, 3.28084, -3.28084])
        vel_bIc_des = jnp.array(msg.velocities) * jnp.array([3.28084, -3.28084, -3.28084])
        chi_des = 0
        chi_dot_des = 0
        self.guam_reference = RefInputs(
            Vel_bIc_des=vel_bIc_des,
            Pos_des=pos_des,
            Chi_des=chi_des,
            Chi_dot_des=chi_dot_des,
        )

    def main(self):
        logger.info("Constructing GUAM...")
        self.guam = FuncGUAM()
        batch_size = 1 # Simulate for a single vehicle test
        state = GuamState.create()

        # Assign first value of traj reference as initial guam state
        # [   0:3  ,   3:6    ,   6:9  ,  9:13 ]
        # [ vel_bEb, Omega_BIb, pos_bii, Q_i2b ]
        # state.aircraft[:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, -100*3.28084, 1, 0, 0, 0]) # Initial pos is z = -100 # Q1 must be 1
        # state.aircraft[:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, -0*3.28084, 1, 0, 0, 0]) # Initial pos is z = -100 # Q1 must be 1
        state.aircraft[:] = np.array([self.guam_reference.Vel_bIc_des[0],
                                      self.guam_reference.Vel_bIc_des[1],
                                      self.guam_reference.Vel_bIc_des[2],
                                      0, 0, 0,
                                      self.guam_reference.Pos_des[0],
                                      self.guam_reference.Pos_des[1],
                                      self.guam_reference.Pos_des[2],
                                      1, 0, 0, 0]) # Q1 must be 1

        logger.info("Calling GUAM...")
        b_state: GuamState = jtu.tree_map(lambda x: np.broadcast_to(x, (batch_size,) + x.shape).copy(), state)

        # Perturb the initial state in the x and y directions.
        # key0, key1 = jr.split(jr.PRNGKey(0))
        # b_state.aircraft[:, 6] = jr.uniform(key0, (batch_size,), minval=-20.0, maxval=20.0)
        # b_state.aircraft[:, 7] = jr.uniform(key1, (batch_size,), minval=-20.0, maxval=20.0)

        vmap_step = jax.jit(jax.vmap(ft.partial(self.guam.step, self.guam.dt), in_axes=(0, None)))
        loop_rate = rospy.Rate(1/self.guam.dt) #in Hz guam.dt = 0.005, so rate = 200Hz

        def simulate_batch(self, b_state0) -> GuamState:
            b_state = b_state0

            if self.plot_switch:
                Tb_state = [b_state0]
                time_list = [0]
                vel_des0 = b_state0.aircraft[0][0:3].tolist()
                pos_des0 = b_state0.aircraft[0][6:9].tolist()
                Ref_list = [vel_des0 + pos_des0]

            kk = 0
            while not rospy.is_shutdown():
                t = kk * self.guam.dt
                kk = kk + 1

                match self.reference_type:
                    case 1:
                        self.guam_reference = lift_cruise_reference_inputs_1(t)
                    case 2:
                        self.guam_reference = lift_cruise_reference_inputs_2(t)
                    case 3:
                        self.guam_reference = landing_reference_inputs(t)
                    # case 4:
                    #     pass

                ref_inputs = self.guam_reference
                b_state = vmap_step(b_state, ref_inputs)

                if self.plot_switch:
                    time_list.append(t)
                    Ref_list.append(ref_inputs.Vel_bIc_des.tolist() + ref_inputs.Pos_des.tolist()) # Store vel and pos reference
                    Tb_state.append(jax2np(b_state))

                # Assign guam state to publisher
                # [   0:3  ,   3:6    ,   6:9  ,  9:13 ]
                # [ vel_bEb, Omega_BIb, pos_bii, Q_i2b ]
                # Convert from feet to meter
                # North East Down in Guam ==> North East Up
                self.guam_pose_msg.pose.position.x = b_state.aircraft[0][6] / 3.28084            # North
                self.guam_pose_msg.pose.position.y = b_state.aircraft[0][7] / 3.28084
                self.guam_pose_msg.pose.position.z = b_state.aircraft[0][8] / 3.28084 * -1
                self.guam_pose_msg.pose.orientation.x = b_state.aircraft[0][9]
                self.guam_pose_msg.pose.orientation.y = b_state.aircraft[0][10]
                self.guam_pose_msg.pose.orientation.z = b_state.aircraft[0][11]
                self.guam_pose_msg.pose.orientation.w = b_state.aircraft[0][12]
                self.guam_pose_pub.publish(self.guam_pose_msg)

                self.guam_vel_msg.linear.x = b_state.aircraft[0][0] / 3.28084            # North
                self.guam_vel_msg.linear.y = b_state.aircraft[0][1] / 3.28084
                self.guam_vel_msg.linear.z = b_state.aircraft[0][2] / 3.28084 * -1
                self.guam_vel_pub.publish(self.guam_vel_msg)

                if self.skip_sleep == False:
                    loop_rate.sleep()

            if self.plot_switch:
                bT_state = jtu.tree_map(lambda *args: np.stack(list(args), axis=1), *Tb_state)
                return bT_state, Ref_list, time_list

        if self.plot_switch:
            if not os.path.exists("results"):
                os.makedirs("results")
            bT_state, Ref_list, time_list = simulate_batch(self, b_state)
            np.savez("results/bT_state.npz", aircraft=bT_state.aircraft)
            Ref_list = np.array(Ref_list)
            np.savez("results/Ref_list.npz", Vel_des=Ref_list[:, 0:3], Pos_des=Ref_list[:, 3:6])
            np.savez("results/time_list.npz", np.array(time_list))
        else:
            simulate_batch(self, b_state)


if __name__ == "__main__":
    pos = carla.Vector3D(0, 0, 100)
    vel = carla.Vector3D(0, 0, 0)

    # Get the vehicle type. Currently supported vehicles: GUAM, MiniHawk (in progress).
    vehicle = get_vehicle_type()

    if rospy.get_param(f'/{vehicle}/debug', default=False):
        with ipdb.launch_ipdb_on_exception():
            if vehicle == 'guam':
                guam_node = GUAM_Node(pos, vel)
                guam_node.main()
            elif vehicle == 'minihawk':
                raise NotImplementedError
            else:
                raise ValueError(f"Vehicle type {vehicle} is not supported.")
    else:
        if vehicle == 'guam':
            guam_node = GUAM_Node(pos, vel)
            guam_node.main()
        elif vehicle == 'minihawk':
            raise NotImplementedError
        else:
            raise ValueError(f"Vehicle type {vehicle} is not supported.")

    if guam_node.plot_switch:
        logger.info("Ploting...")
        plot_batch_with_ref(save_video=guam_node.save_video)
