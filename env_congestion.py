
import pygame
import pygame.freetype
from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
from pygame.locals import K_c
from pygame.locals import K_TAB
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
from pygame.locals import K_a
from pygame.locals import K_d

import weakref
import random
import collections
import numpy as np
import math
import cv2
import re
import sys

sys.path.append('C:\\Users\RRC4\Downloads\CARLA_0.9.9.4\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.9-py3.7-win-amd64.egg')
sys.path.append('C:\\Users\RRC4\Downloads\CARLA_0.9.9.4\WindowsNoEditor\PythonAPI\carla')

import carla
from carla import ColorConverter as cc
from agents.navigation.basic_agent import BasicAgent

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser


WIDTH, HEIGHT = 80, 45
screen_width, screen_height = 1280, 720
FULLGREEN = (0, 255, 0)
FULLRED = (255, 0, 0)
FULLBLUE = (0, 0, 255)
FULLBLACK = (255, 255, 255)

class Congestion(object):
    def __init__(self, enabled_obs_number=7,  vehicle_type = 'random',
                 joystick_enabled = False, control_interval = 1,
                 frame=12, port=2000):

             
        self.observation_size_width = WIDTH
        self.observation_size_height = HEIGHT
        self.observation_size = WIDTH * HEIGHT
        self.action_size = 1

        ## set the carla World parameters
        self.vehicle_type = vehicle_type
        self.joystick_enabled = joystick_enabled
        self.intervention_type = 'joystick' if joystick_enabled else 'keyboard'
        self.control_interval = control_interval
        self.index = None

        ## set the vehicle actors
        self.ego_vehicle = None
        self.obs_list, self.bp_obs_list, self.spawn_point_obs_list = [], [] ,[]
        self.enabled_obs = enabled_obs_number

        ## set the sensory actors
        self.collision_sensor = None
        self.seman_camera = None
        self.viz_camera = None
        self.surface = None
        self.camera_output = np.zeros([screen_height,screen_width,3])
        self.recording = False
        self.Attachment = carla.AttachmentType

        ## connect to the CARLA client
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.frame = frame
        
        ## build the CARLA world
        self.world = self.client.load_world('Town04')
        self.map = self.world.get_map()
        
        self._weather_presets = find_weather_presets()
        self._weather_index = 1
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])
        
        ## initialize the pygame settings
        pygame.init()
        pygame.font.init()
        
        self.display = pygame.display.set_mode((screen_width, screen_height),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.infoObject = pygame.display.Info()
        pygame.display.set_caption('Congestion Scenario')
        
        # initialize the hint settings
        font = pygame.font.Font('freesansbold.ttf', 32)
        self.text_humanguidance = font.render('Human Guidance Mode', True, FULLBLACK, FULLGREEN)
        self.text_humanguidance_rect = self.text_humanguidance.get_rect()
        self.text_humanguidance_rect.center = (1000, 60)
        self.text_RLinference = font.render('RL Inference Mode', True, FULLBLACK, FULLRED)
        self.text_RLinference_rect = self.text_humanguidance.get_rect()
        self.text_RLinference_rect.center = (1000, 60)
        self.text_humanmodelguidance = font.render('Human Model Guidance Mode', True, FULLBLACK, FULLBLUE)
        self.text_humanmodelguidance_rect = self.text_humanguidance.get_rect()
        self.text_humanmodelguidance_rect.center = (1000, 60)
        
        
        if self.joystick_enabled:
            pygame.joystick.init()
            self._parser = ConfigParser()
            self._parser.read('./wheel_config.ini')
            self._steer_idx = int(self._parser.get('G29 Racing Wheel', 'steering_wheel'))
            self._throttle_idx = int(self._parser.get('G29 Racing Wheel', 'throttle'))
            self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
            self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
            self._handbrake_idx = int(self._parser.get('G29 Racing Wheel', 'handbrake'))

        # self.reset()

    def reset(self):
        
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / self.frame
        self.world.apply_settings(settings)
        
        
        ## reset the recording lists
        self.intervene_history = []
        self.previous_action_list = []

        ## reset the human intervention state
        self.intervention = False
        self.keyboard_intervention = False
        self.joystick_intervention =  False
        
        self.world.tick()
        ## spawn the ego vehicle (fixed)
        bp_ego = self.world.get_blueprint_library().filter('vehicle.audi.etron')[0]
        bp_ego.set_attribute('color', '0, 0, 0')
        
        spawn_point_ego = self.world.get_map().get_spawn_points()[1]
        spawn_point_ego.location.x = -13
        spawn_point_ego.location.y = -160
        spawn_point_ego.location.z = 0.1
        spawn_point_ego.rotation.yaw = 90
        
        if self.ego_vehicle is not None:
            self.destroy()
        self.ego_vehicle = self.world.spawn_actor(bp_ego, spawn_point_ego)
        
        self.agent = BasicAgent(self.ego_vehicle, target_speed = 5.5 * 3.6)
        self.agent.set_destination((-13, -80, spawn_point_ego.location.z+1))
        
        self.ego_vehicle.set_velocity(carla.Vector3D(0, 5, 0))

        self.control = carla.VehicleControl()
        self.heuristic = -80

        ## spawn the surrounding vehicles
        self.obs_list, self.bp_obs_list, self.spawn_point_obs_list = [], [] ,[]
        self.obs_agent_list = []
        lat_list = [-16.5, -13, -9.5, -9.5, -11.5, -15, -16.5]
        long_list = [-170, -175, -170, -160, -150, -148, -158]
                
        for index in range(self.enabled_obs):
            bp, sp = self._produce_vehicle_blueprint(lat_list[index], long_list[index], 90)
            self.bp_obs_list.append(bp)
            self.spawn_point_obs_list.append(sp)
        for index in range(self.enabled_obs):
            try:
                self.obs_list.append(self.world.spawn_actor(self.bp_obs_list[index],
                                                        self.spawn_point_obs_list[index]))
                self.obs_list[index].set_velocity(carla.Vector3D(0, 5, 0))
            except:
                self.enabled_obs -= 1
                pass
        

        ## configurate and spawn the collision sensor
        # clear the collision history list
        self.collision_history = []
        bp_collision = self.world.get_blueprint_library().find('sensor.other.collision')
        # spawn the collision sensor actor
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        self.collision_sensor = self.world.spawn_actor(
                bp_collision, carla.Transform(), attach_to=self.ego_vehicle)
        # obtain the collision signal and append to the history list
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: Congestion._on_collision(weak_self, event))

        
        ## configurate and spawn the camera sensors
        # the candidated transform of camera's position: frontal
        self.camera_transforms = [
            (carla.Transform(carla.Location(x=-1, z=15), carla.Rotation(pitch=10)), self.Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8)), self.Attachment.SpringArm)]
        self.camera_transform_index = 1
        # the candidated camera type: rgb (viz_camera) and semantic (seman_camera)
        self.cameras = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}]
            ]
                
        bp_viz_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_viz_camera.set_attribute('image_size_x', '1280')
        bp_viz_camera.set_attribute('image_size_y', '720')
        bp_viz_camera.set_attribute('sensor_tick', '0.02')
        self.cameras[0].append(bp_viz_camera)

        bp_seman_camera = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp_seman_camera.set_attribute('image_size_x', '320')
        bp_seman_camera.set_attribute('image_size_y', '180')
        bp_seman_camera.set_attribute('sensor_tick', '0.04')
        self.cameras[1].append(bp_seman_camera)
        

        # spawn the camera actors
        if self.seman_camera is not None:
            self.seman_camera.destroy()
            self.viz_camera.destroy()
            self.surface = None

        self.viz_camera = self.world.spawn_actor(
            self.cameras[0][-1],
            self.camera_transforms[self.camera_transform_index][0],
            attach_to=self.ego_vehicle,
            attachment_type=self.Attachment.SpringArm)

        self.seman_camera = self.world.spawn_actor(
            self.cameras[1][-1],
            self.camera_transforms[self.camera_transform_index - 1][0],
            attach_to=self.ego_vehicle,
            attachment_type=self.camera_transforms[self.camera_transform_index - 1][1])

        # obtain the camera image
        weak_self = weakref.ref(self)
        self.seman_camera.listen(lambda image: Congestion._parse_seman_image(weak_self, image))
        self.viz_camera.listen(lambda image: Congestion._parse_image(weak_self, image))

        
        ## reset the step counter
        self.count = 0
        
        state, _ = self.get_observation()
        return state
    
    
    def render(self, display):
        if self.surface is not None:
            m = pygame.transform.smoothscale(self.surface, 
                                 [int(self.infoObject.current_w), 
                                  int(self.infoObject.current_h)])
            display.blit(m, (0, 0))


    def _parse_seman_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.cameras[1][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.array(image.raw_data)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        self.camera_output = array
    

    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.cameras[0][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.array(image.raw_data)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
        if self.intervention:
            # pygame.draw.rect(self.surface, FULLGREEN, self.rectangle)
            self.surface.blit(self.text_humanguidance, self.text_humanguidance_rect)
        else:
            # pygame.draw.rect(self.surface, FULLRED, self.rectangle)
            self.surface.blit(self.text_RLinference, self.text_RLinference_rect)
    
    def show_human_model_mode(self):
        self.surface.blit(self.text_humanmodelguidance, self.text_humanmodelguidance_rect)
        
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) > 4000:
            self.collision_history.pop(0)


    def get_collision_history(self):
        collision_history = collections.defaultdict(int)
        flag = 0
        for frame, intensity in self.collision_history:
            collision_history[frame] += intensity
            if intensity != 0:
                flag = 1
        return collision_history, flag
    
    
    def step(self,action):
        
        self.world.tick()
        self.render(self.display)
        
        pygame.display.flip()
        human_control = None
        
        # act once per control interval
        action = action[0] if action.shape != () else float(action)
        action = action if self.count % self.control_interval == 0 else self.previous_action_list[-1]
        
        # execute longitudinal planning & control by a IDM
        self.control = self.agent.run_step()
        self.control = self.IDM(self.control)
        
        
        # detect if guidance from human exists (jostick or keyboard)
        if self.intervention_type == 'joystick':
            human_control = self._parse_wheel()
        else:
            human_control = self._parse_key()
        
        self.intervention = self.joystick_intervention or self.keyboard_intervention
        
        
        # lateral control by RL's action if no human guidance else by human action
        if self.intervention:
            self.control.steer = human_control 
        else:
            self.control.steer = action

        
        ## achieve the control to the ego vehicle
        self.ego_vehicle.apply_control(self.control)
        
        
        # record the human intervention histroy
        self.intervene_history.append(human_control)
        
        
        # record the adopted action
        adopted_action = action if not self.intervention else human_control
        self.previous_action_list.append(adopted_action)
        
        
        # achieve the control to the surrounding vehicles
        for index in range(self.enabled_obs):
            obs_command = carla.VehicleControl()
            obs_command.steer = 0
            # obs_velocity_diff = 5 - self.obs_list[index].get_velocity().y
            # obs_command.throttle = min(1, 0.4*obs_velocity_diff) if obs_velocity_diff>0 else 0
            # obs_command.brake = min(1, -0.4*obs_velocity_diff) if obs_velocity_diff<0 else 0
            obs_command = self.IDM(obs_command, self.obs_list[index])
            self.obs_list[index].apply_control(obs_command)
        
        
        # obtain the state transition and other variables after taking the action (control command)
        next_states, other_indicators = self.get_observation()


        # detect if the step is the terminated step, by considering: collision and episode fininsh
        collision = self.get_collision_history()[1]
        finish = self.ego_vehicle.get_location().y > -80
        done = collision or finish
        
        
        # calculate the reward signal of the step
        r_smooth= -3 * abs(self.control.steer)
        reward = finish*10 - collision*10 + r_smooth * 1
        reward = np.clip(reward,-10,10)
        
        # update the epsodic step
        self.count += 1
        
        # record the physical variables
        yaw_rate = np.arctan(self.ego_vehicle.get_velocity().x/self.ego_vehicle.get_velocity().y) if self.ego_vehicle.get_velocity().y > 0 else 0
        physical_variables = {'velocity_y':self.ego_vehicle.get_velocity().y,
                 'velocity_x':self.ego_vehicle.get_velocity().x,
                 'position_y':self.ego_vehicle.get_location().y,
                 'position_x':self.ego_vehicle.get_location().x,
                 'yaw_rate':yaw_rate,
                 'yaw':self.ego_vehicle.get_transform().rotation.yaw,
                 'pitch':self.ego_vehicle.get_transform().rotation.pitch,
                 'roll':self.ego_vehicle.get_transform().rotation.roll,
                 'angular_velocity_y':self.ego_vehicle.get_angular_velocity().y,
                 'angular_velocity_x':self.ego_vehicle.get_angular_velocity().x,
                 'acceleration_x':self.ego_vehicle.get_acceleration().x,
                 'acceleration_y':self.ego_vehicle.get_acceleration().y
                 }
        
        if done or self.parse_events():
            self.post_process()
            
        return next_states, human_control, reward, self.intervention, done, physical_variables
    

    def destroy(self):
        self.seman_camera.stop()
        self.viz_camera.stop()
        self.collision_sensor.stop()
        actors = [
            self.ego_vehicle,
            self.viz_camera,
            self.seman_camera,
            self.collision_sensor,
            ]
        actors.extend(self.obs_list)

        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in actors])
        self.seman_camera = None
        self.viz_camera = None
        self.collision_sensor = None
        self.ego_vehicle = None


    def post_process(self):
        if self.original_settings:
            self.world.apply_settings(self.original_settings)

        if self.world is not None:
            self.destroy()


    def close(self):
        pygame.display.quit()
        pygame.quit()

        
    def signal_handler(self, sig, frame):
        print('Procedure terminated!')
        self.close()
        sys.exit(0)
        
        
    def get_observation(self):
        ## obtain image-based state space
        # state variable sets
        state_space = self.camera_output[:,:,0]
        state_space = cv2.resize(state_space,(WIDTH, HEIGHT))
        state_space = np.float16(np.squeeze(state_space)/255)
        
        # other indicators facilitating producing reward function signal
        other_indicators = None
        
        return state_space, other_indicators
    

    def obtain_real_observation(self):
        state_space = self.camera_output[:,:,0]
        return state_space
    
    
    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_TAB:
                    self._toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    self._next_weather(reverse=True)
                elif event.key == K_c:
                    self._next_weather()
                    
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    self.intervention = False
                elif event.button == self._reverse_idx:
                    self.control.gear = 1 if self.control.reverse else -1
                elif event.button == 1:
                    self._toggle_camera()
                elif event.button == 2:
                    self._next_sensor()
    
    def _parse_key(self):
        # Detect if there is a human action from the keyboard
        keys = pygame.key.get_pressed()

        # Process human action from the keyboard
        if (keys[K_LEFT] or keys[K_a]) and (not self.a_pressed):
            self.human_default_steer = -0.02
            self.a_pressed = 1
        elif (keys[K_LEFT] or keys[K_a]) and (self.a_pressed):
            self.human_default_steer -= 0.02

        elif (keys[K_RIGHT] or keys[K_d]) and (not self.d_pressed):
            self.human_default_steer = 0.02
            self.d_pressed = 1
        elif (keys[K_RIGHT] or keys[K_d]) and (self.d_pressed):
            self.human_default_steer += 0.02
        else:
            self.human_default_steer = 0
            self.d_pressed = 0
            self.a_pressed = 0
        
        if (keys[K_LEFT] or keys[K_a]) or (keys[K_RIGHT] or keys[K_d]):

            human_steer = np.clip(np.float(self.human_default_steer),-1,1)
        
            human_action = human_steer
            self.keyboard_intervention = True
        
        else:
            human_action = None
            self.keyboard_intervention = False
        
        return human_action
    
    
    def _parse_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        
        # detect joystick intervention signal
        if len(self.intervene_history) > 2:
            # intervention is activated if human participants move the joystick
            if abs(self.intervene_history[-2] - (jsInputs[self._throttle_idx]-jsInputs[self._brake_idx])) > 0.02:
                self.joystick_intervention = True
        if len(self.intervene_history) > 5:
            # the intervention is deactivated if the joystick continue to be stable for 0.2 seconds
            if abs(self.intervene_history[-5] - self.intervene_history[-1]) < 0.01:
                self.joystick_intervention = False

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        if self.joystick_intervention:
            human_steer = math.tan(1.1 * jsInputs[self._steer_idx])
            human_action = human_steer
        else:
            human_action = None
        
        return human_action
    
    
    def _produce_vehicle_blueprint(self, x, y, yaw=0, color=0):

        blueprint_library = self.world.get_blueprint_library()
        
        if self.vehicle_type != 'single':
            ran=np.random.rand()
            if ran<0.33:
                bp = blueprint_library.filter('vehicle.volkswagen.t2')[0]
            elif ran<0.67:
                bp = blueprint_library.filter('vehicle.jeep.wrangler_rubicon')[0]
            else:
                bp = blueprint_library.filter('vehicle.dodge_charger.police')[0]
        else:
            bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))

        spawn_point = self.world.get_map().get_spawn_points()[0]
        spawn_point.location.x = x
        spawn_point.location.y = y
        spawn_point.location.z = 0.1
        spawn_point.rotation.yaw = yaw

        return bp, spawn_point
    
    def _produce_walker_blueprint(self, x, y):
        
        bp = self.world.get_blueprint_library().filter('walker.*')[np.random.randint(2)]
        
        spawn_point = self.world.get_map().get_spawn_points()[0]
        spawn_point.location.x = x
        spawn_point.location.y = y
        spawn_point.location.z += 0.1      
        spawn_point.rotation.yaw = 0
        
        return bp, spawn_point
    
    
    def IDM(self, action, vehicle='Default'):
           
        delta = 4
        a, b, T, s0 = 0.73, 1.67, 0.8, 2
        rear = False
        
        # if no surrounding vehicles, do not use this IDM driver
        if len(self.obs_list)==0:
            return action
        
        if vehicle == 'Default':
            vehicle = self.ego_vehicle
            v0 = 5.1 * 3.6
        else:
            v0 = 5 * 3.6
        
        if (abs(vehicle.get_location().x - self.ego_vehicle.get_location().x))<0.5 and vehicle.id!=self.ego_vehicle.id:
            s0 = 8
            rear = True
            
        waypoint = self.map.get_waypoint(vehicle.get_location())
        
        waypoint_list, y_list, v_list = [], [] ,[]
        for index1 in range(self.enabled_obs):
            waypoint_list.append(self.map.get_waypoint(self.obs_list[index1].get_location()))
            y_list.append(self.obs_list[index1].get_location().y)
            v_list.append(self.obs_list[index1].get_velocity().y)
        if rear:
            y_list.append(self.ego_vehicle.get_location().y)
            v_list.append(self.ego_vehicle.get_velocity().y)
            
        y_list = np.array(y_list)
        v_list = np.array(v_list)
        
        y_diff = y_list - vehicle.get_location().y
        
        # leading vehicle index
        front_index = [i for i in range(len(y_diff)) if y_diff[i]>0]
        
        # same-lane vehicle index
        same_lane_index = [i for i in range(len(waypoint_list)) if (waypoint_list[i].lane_id == waypoint.lane_id)]
        
        # same-lane leading vehicle index
        same_lane_front_index = list(set(front_index).intersection(same_lane_index))
        
        # same_lane_front_index = list([5,6])
        # if there exists leading vehicle in the lane
        if len(same_lane_front_index) != 0:
    
            # find the nearest leading vehicle
            index = y_diff.tolist().index(min(y_diff[same_lane_front_index]))
            y_target = y_list[index]
            v_target = v_list[index]
            self.index = index
            s_delta = y_target - vehicle.get_location().y - 4
            v_delta = vehicle.get_velocity().y - v_target
            s_prime = s0 + vehicle.get_velocity().y * T + vehicle.get_velocity().y*v_delta/2/np.sqrt(a*b)
            
            acc = a * (1 - (vehicle.get_velocity().y/v0)**delta - (s_prime/s_delta)**2)
            
            # if no steering behavior (<0.03 for noise filter), then execute car-following constraints
            if abs(action.steer) < 0.03:
                if acc>0:
                    action.throttle = acc
                else:
                    action.brake = -acc
                action.manual_gear_shift = False
                action.hand_brake = False
            
        return action
    
    
    def _toggle_camera(self):
        self.camera_transform_index = (self.camera_transform_index + 1) % len(self.camera_transforms)
    
    def _next_sensor(self):
        self.camera_index += 1
        
    def _next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])
    
    def _calculate_velocity(self, actor):
        return ((actor.get_velocity().x)**2 + (actor.get_velocity().y)**2
                        + (actor.get_velocity().z)**2)**(1/2)
    
    def _dis_p_to_l(self,k,b,x,y):
        dis = abs((k*x-y+b)/math.sqrt(k*k+1))
        return self._sigmoid(dis,2)
    
    def _calculate_k_b(self,x1,y1,x2,y2):
        k = (y1-y2)/(x1-x2)
        b = (x1*y2-x2*y1)/(x1-x2)
        return k,b
    
    def _dis_p_to_p(self,x1,y1,x2,y2):
        return math.sqrt((x1-x2)**2+(y1-y2)**2)
    
    def _to_corner_coordinate(self,x,y,yaw):
        xa = x+2.64*math.cos(yaw*math.pi/180-0.43)
        ya = y+2.64*math.sin(yaw*math.pi/180-0.43)
        xb = x+2.64*math.cos(yaw*math.pi/180+0.43)
        yb = y+2.64*math.cos(yaw*math.pi/180+0.43)
        xc = x+2.64*math.cos(yaw*math.pi/180-0.43+math.pi)
        yc = y+2.64*math.cos(yaw*math.pi/180-0.43+math.pi)
        xd = x+2.64*math.cos(yaw*math.pi/180+0.43+math.pi)
        yd = y+2.64*math.cos(yaw*math.pi/180+0.43+math.pi)
        return xa,ya,xb,yb,xc,yc,xd,yd
    
    def _sigmoid(self,x,theta):
        return 2./(1+math.exp(-theta*x))-1

        
    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

