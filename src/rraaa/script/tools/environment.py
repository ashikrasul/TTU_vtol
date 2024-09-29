import pygame
import carla
import random
import logging
import numpy as np

import rospy
import tf
from std_msgs.msg import Float32, Bool, Header, String, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pcl2  #https://answers.ros.org/question/207071/how-to-fill-up-a-pointcloud-message-with-data-in-python

from tools.sensors import CollisionSensor, SensorManager
from tools.utils import FPSTimer, pack_multiarray_ros_msg, pack_df_from_multiarray_msg, pack_image_ros_msg, ROSMsgMatrix
from tools.utils import add_carla_rotations, rad2deg, deg2rad, carla_transform_to_ros_xyz_quaternion

from scipy.spatial.transform import Rotation
QUAT_from_XYZ_to_NED = Rotation.from_euler('ZYX', np.array([90, 0, 180]), degrees=True).as_quat()  #x, y, z, w format

from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class Environment():

    def __init__(self, args, client, config):

        self.args = args
        self.client = client
        self.config = config
        self.world = client.load_world(config['map'])

        ### Setting the world ###
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager(args.tm_port)
        if args.asynch:
            self.traffic_manager.set_synchronous_mode(False)
            settings.synchronous_mode = False
        else:
            self.traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = .1
        settings.actor_active_distance  = 500
        self.world.apply_settings(settings)

        ### Initiate states and blank messages ###
        self._autopilot_on = False
        self.collision_n_count = 0
        self.df_msg_input_display    = None
        self.df_msg_tracking_control = None

        ### ROS msg publisher init. ###
        self.pub_vehicles_state = rospy.Publisher('/carla_node/vehicles_state', Float32MultiArray, queue_size=1)
        self.pub_world_state    = rospy.Publisher('/carla_node/world_state', Float32MultiArray, queue_size=1)

        self.pub_camera_frame_left  = rospy.Publisher('/carla_node/cam_left/image_raw', Image, queue_size=1)
        self.pub_camera_frame_front = rospy.Publisher('/carla_node/cam_front/image_raw',Image, queue_size=1)
        self.pub_camera_frame_right = rospy.Publisher('/carla_node/cam_right/image_raw',Image, queue_size=1)
        self.pub_camera_frame_back  = rospy.Publisher('/carla_node/cam_back/image_raw', Image, queue_size=1)
        self.pub_camera_frame_up    = rospy.Publisher('/carla_node/cam_up/image_raw',   Image, queue_size=1)
        self.pub_camera_frame_down  = rospy.Publisher('/carla_node/cam_down/image_raw', Image, queue_size=1)

        self.pub_camera_frame_overview = rospy.Publisher('/carla_node/cam_overview/image_raw', Image, queue_size=1)
        self.pub_lidar_point_cloud = rospy.Publisher('/carla_node/lidar_point_cloud', PointCloud2, queue_size=1)
        self.pub_initial_transform = rospy.Publisher('/carla_node/initial_transform',  Twist, queue_size=1)
        # tf broadcaster init.
        self.tf_broadcaster = tf.TransformBroadcaster()

        ### ROS msg Subscriber init. ###
        self.vehicle_type = rospy.get_param("vehicle")
        self.sub_jax_guam_pose = rospy.Subscriber(f'/{self.vehicle_type}/pose', PoseStamped, self.callback_jax_guam_pose)

        ### Timer for frames per second (FPS) ###
        self.fps_timer = FPSTimer()
        self.client_clock = pygame.time.Clock()
        self.world.on_tick(self.fps_timer.on_world_tick)

        self.msg_mat = ROSMsgMatrix()

        self.spectator = self.world.get_spectator()

        ### Start the environment ###
        self.start()

    def tick(self):

        ### Publish ROS messages ####
        self.broadcast_tf()
        self.publish_camera_image()
        self.publish_lidar()
        self.publish_states()

        ### Tick the Carla ###
        if self.args.asynch:
            self.world.wait_for_tick()
        else:
            self.world.tick()

        return 0

    def update_spectator(self):
        transform = carla.Transform(self.ego_vehicle.get_transform().transform(
                                        carla.Location(x=-10, y=0, z=4)),
                                    carla.Rotation(pitch=-20))
        self.spectator.set_transform(transform)

    def spawn_ego_vehicle(self):

        # Instanciating the Ego vehicle to which we attached the sensors
        ego_bp = self.world.get_blueprint_library().filter('model3')[0]
        ego_bp.set_attribute('role_name','ego')
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        
        # If the vehicle is MiniHawk, ensure that there is no rotation. We need it to make sure the pose processing happens correctly.
        if self.vehicle_type == 'minihawk':
            spawn_point = carla.Transform( 
                location=spawn_point.location,
                rotation=carla.Rotation(
                    0,
                    0,
                    0
                )
            )

        self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
        self.ego_vehicle.set_autopilot(False)
        self.ego_vehicle.set_simulate_physics(False)
        self.ego_vehicle.set_enable_gravity(False)
        self.control_variable = carla.VehicleControl()
        self.ego_vehicle.apply_control(self.control_variable)
        self.world.tick()
        self.initial_transform = self.ego_vehicle.get_transform()

        # Manually move the ego vehicle to 100m elevation. Match to node_guam.py:guam_reference_init
        location = self.ego_vehicle.get_location()
        new_location = location + carla.Location(x=0.0, y=0.0, z=100.0)
        self.ego_vehicle.set_location(new_location)
        self.world.tick()
        self.update_spectator()


    def start(self):

        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []

        ### spawn vehicles ###
        self.spawn_ego_vehicle()  # ego vehicle
        self.generate_traffic()

        ### sensor initialization ###
        self.camera_front= SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=2,  z=1.5), carla.Rotation(yaw=+00, pitch=-10)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_left = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_right= SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_back = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=-2,  z=1.5), carla.Rotation(yaw=+180, pitch=-10)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_down = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=0,  z=-1.5), carla.Rotation(pitch=-90)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_up   = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=0,  z=2.4), carla.Rotation(pitch= 90)),
                                        self.ego_vehicle, {'fov':'90.0', 'image_size_x': '400', 'image_size_y': '400'})
        self.camera_overview  = SensorManager(self.world, 'RGBCamera', carla.Transform(carla.Location(x=-1, z=7.0), carla.Rotation(pitch=-60)),
                                        self.ego_vehicle, {'fov':'60.0', 'image_size_x': '600', 'image_size_y': '600'})

        self.transform_lidar_from_vehicle = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(yaw=+00, roll=00))
        self.lidar_sensor   = SensorManager(self.world, 'LiDAR', self.transform_lidar_from_vehicle, self.ego_vehicle, {
                                            'channels' : '64',
                                            'range' : '100',
                                            'points_per_second': '230400',
                                            'upper_fov': '-27',
                                            'lower_fov': '-90',
                                            'rotation_frequency': '10',
                                            'horizontal_fov': '360',
                                            'sensor_tick':'0.1',
                                            })
        self.collision_sensor_ego = CollisionSensor(self.ego_vehicle, panic=True)
        return 0


    def reset(self):
        print('RESET CALLED!')
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.ego_vehicle.set_transform(spawn_point)
        return 0


    def destroy(self):
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()
        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.spectator.destroy()
        self.ego_vehicle.destroy()
        print('\nego vehicle destroyed!')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
        self.world.apply_settings(self.original_settings)

        return 0


    def callback_jax_guam_pose(self, msg):

        ### Unpack PoseStamped msg ###
        X, Y, Z = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z) # meters

        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w)
        (yaw, pitch, roll) = euler_from_quaternion(quaternion) # unit rad

        ### Overrisde the pose, i.e., transform ###
        transform = self.ego_vehicle.get_transform()
        matlab_location = carla.Location(X, -Y, Z) # CARLA uses the Unreal Engine coordinates system. This is a Z-up left-handed system.
        transform.location = matlab_location + self.initial_transform.location
        matlab_rotation = carla.Rotation(rad2deg(-pitch), rad2deg(-yaw), rad2deg(roll))  # The constructor method follows a specific order of declaration: (pitch, yaw, roll), which corresponds to (Y-rotation,Z-rotation,X-rotation).
        transform.rotation = add_carla_rotations(matlab_rotation, self.initial_transform.rotation)
        self.ego_vehicle.set_transform(transform)
        self.update_spectator()


    def broadcast_tf(self):

        ### Broadcast TF-vehicle from map ###
        veh_transform = self.ego_vehicle.get_transform()
        xyz, quaternion = carla_transform_to_ros_xyz_quaternion(veh_transform)
        self.tf_broadcaster.sendTransform(xyz, quaternion, rospy.Time.now(), 'vehicle', 'map')

        ### Broadcast TF-sensor from vehicle ###
        xyz, quaternion = carla_transform_to_ros_xyz_quaternion(self.transform_lidar_from_vehicle)
        self.tf_broadcaster.sendTransform(xyz, quaternion, rospy.Time.now(), 'sensor', 'vehicle')


    def publish_lidar(self):
        header = Header()
        header.frame_id = 'sensor'
        if self.lidar_sensor.data is not None:
            points = np.array(self.lidar_sensor.data[:,:3])
            points[:, 1] = -points[:, 1]
            self.pub_lidar_point_cloud.publish(pcl2.create_cloud_xyz32(header,points))

    def publish_camera_image(self):
        header = Header()
        header.stamp = rospy.Time.now()
        if self.camera_left and self.camera_left.data is not None:
            self.pub_camera_frame_left.publish(pack_image_ros_msg(self.camera_left.data, header, 'left_camera'))
        if self.camera_right and self.camera_right.data is not None:
            self.pub_camera_frame_right.publish(pack_image_ros_msg(self.camera_right.data, header, 'right_camera'))
        if self.camera_overview and self.camera_overview.data is not None:
            self.pub_camera_frame_overview.publish(pack_image_ros_msg(self.camera_overview.data, header, 'overview_camera'))
        if self.camera_front and self.camera_front.data is not None:
            self.pub_camera_frame_front.publish(pack_image_ros_msg(self.camera_front.data, header, 'front_camera'))
        if self.camera_back and self.camera_back.data is not None:
            self.pub_camera_frame_back.publish(pack_image_ros_msg(self.camera_back.data, header, 'back_camera'))
        if self.camera_up and self.camera_up.data is not None:
            self.pub_camera_frame_up.publish(pack_image_ros_msg(self.camera_up.data, header,  'up_camera'))
        if self.camera_down and self.camera_down.data is not None:
            self.pub_camera_frame_down.publish(pack_image_ros_msg(self.camera_down.data, header,  'down_camera'))

    def publish_states(self):

        # World State
        world_state = np.array([[self.fps_timer.server_fps, self.client_clock.get_fps()]])
        label_row = 'world'
        label_col = 'server_fps,client_fps'
        self.pub_world_state.publish(pack_multiarray_ros_msg(self.msg_mat.mat, world_state, label_row, label_col))

        # Vehicle State
        state_key, values_1 = self.get_vehicle_state(self.ego_vehicle)
        _,         values_2 = self.get_vehicle_state(self.ego_vehicle)
        veh_list_str = ['ego_vehicle','forward_vehicle']
        state_val_np = np.array([values_1, values_2])
        multiarray_ros_msg = pack_multiarray_ros_msg(self.msg_mat.mat, state_val_np, ','.join(veh_list_str), ','.join(state_key))
        self.pub_vehicles_state.publish(multiarray_ros_msg)

        return 0


    @staticmethod
    def get_vehicle_state(vehicle):
        transform = vehicle.get_transform()
        values = [transform.location.x, transform.location.y, transform.location.z]
        keys   = ['x', 'y', 'z']
        values += [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw]
        keys   += ['roll', 'pitch', 'yaw']
        return keys, values

    @staticmethod
    def get_actor_blueprints(world, filter, generation):
        bps = world.get_blueprint_library().filter(filter)
        if generation.lower() == "all":
            return bps
        # If the filter returns only one bp, we assume that this one needed
        # and therefore, we ignore the generation
        if len(bps) == 1:
            return bps

        try:
            int_generation = int(generation)
            # Check if generation is in available generations
            if int_generation in [1, 2]:
                bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                return bps
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []

    def generate_traffic(self):
        synchronous_master = False
        blueprints = self.get_actor_blueprints(self.world, self.args.filterv, self.args.generationv)
        blueprintsWalkers = self.get_actor_blueprints(self.world, self.args.filterw, self.args.generationw)

        if self.args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if self.args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self.args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.args.number_of_vehicles, number_of_spawn_points)
            self.args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        hero = self.args.hero
        for n, transform in enumerate(spawn_points):
            if n >= self.args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))

        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if self.args.car_lights_on:
            all_vehicle_actors = self.world.get_actors(self.vehicles_list)
            for actor in all_vehicle_actors:
                self.traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road

        if self.args.seedw:
            self.world.set_pedestrians_seed(self.args.seedw)
            random.seed(self.args.seedw)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(self.args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, synchronous_master)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, synchronous_master)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)


        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))
