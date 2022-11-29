
import pygame
import pygame.freetype
from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
from pygame.locals import K_c
from pygame.locals import K_TAB
from pygame.locals import K_DOWN
from pygame.locals import K_UP
from pygame.locals import K_w
from pygame.locals import K_s
from pygame import gfxdraw

import weakref
import random
import collections
import numpy as np
import math
import cv2
import re
import sys
'''
Add your path of the CARLA simulator below.
'''
sys.path.append('xxx/carla-0.9.X-py3.X-linux-x86_64.egg')

import carla
from carla import ColorConverter as cc
from agents.navigation.basic_agent import BasicAgent

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser


screen_width, screen_height = 1280, 720
WIDTH, HEIGHT = 80, 45
FULLGREEN = (0, 255, 0)
FULLRED = (255, 0, 0)
FULLBLUE = (0, 0, 255)
FULLBLACK = (255, 255, 255)

class LeftTurn(object):
    def __init__(self, enabled_obs_number=3, vehicle_type = 'random',
                 joystick_enabled = False, control_interval = 1,
                 conservative_surrounding = False, frame=12, port=2000):

        self.observation_size_width = WIDTH
        self.observation_size_height = HEIGHT
        self.observation_size = WIDTH * HEIGHT
        self.action_size = 1

        ## set the carla World parameters
        self.vehicle_type = vehicle_type
        self.joystick_enabled = joystick_enabled
        self.intervention_type = 'joystick' if joystick_enabled else 'keyboard'
        self.control_interval = control_interval
        self.conservative_surrounding = conservative_surrounding

        ## set the vehicle actors
        self.ego_vehicle = None
        self.obs_list, self.bp_obs_list, self.spawn_point_obs_list = [], [] ,[]
        self.enabled_obs = enabled_obs_number

        ## set the sensory actors
        self.collision_sensor = None
        self.seman_camera = None
        self.viz_camera = None
        self.surface = None
        self.camera_output = np.zeros([720,1280,3])
        self.recording = False
        self.Attachment = carla.AttachmentType

        ## connect to the CARLA client
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.frame = frame
        
        ## build the CARLA world
        self.world = self.client.load_world('Town01')
        self.map = self.world.get_map()
        
        self._weather_presets = find_weather_presets()
        self._weather_index = 2
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])
        
        ## initialize the pygame settings
        pygame.init()
        pygame.font.init()
        
        self.display = pygame.display.set_mode((screen_width, screen_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.infoObject = pygame.display.Info()
        pygame.display.set_caption('LeftTurn Scenario')
        
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
        self.risk = None
        self.v_upp = 19.5/7
        self.v_low = 13.5/7
        self.terminate_position = 0
        self.index_obs_concerned = None
        self.risk = None
        
        ## spawn the ego vehicle (fixed)
        bp_ego = self.world.get_blueprint_library().filter('vehicle.audi.tt')[0]
        # bp_ego = self.world.get_blueprint_library().filter('vehicle.volkswagen.t2')[0]
        bp_ego.set_attribute('color', '0, 0, 0')

        spawn_point_ego = self.world.get_map().get_spawn_points()[0]
        spawn_point_ego.location.x = 88
        spawn_point_ego.location.y = 310
        spawn_point_ego.location.z  = 0.1
        spawn_point_ego.rotation.yaw = 90

        if self.ego_vehicle is not None:
            self.destroy()
        self.ego_vehicle = self.world.spawn_actor(bp_ego, spawn_point_ego)

        self.world.tick()
        self.agent = BasicAgent(self.ego_vehicle, target_speed = 5 * 3.6)
        self.agent.set_destination((120, 330, spawn_point_ego.location.z))

        initial_velocity = carla.Vector3D(0, 0, 0)
        self.ego_vehicle.set_velocity(initial_velocity)

        self.control = carla.VehicleControl()
        self.h_control = self.control # human controller
        self.heuristic = 330.5

        ## spawn the surrounding vehicles
        self.obs_list, self.bp_obs_list, self.spawn_point_obs_list = [], [] ,[]
        self.obs_velo_list = []
        lat_list = [110, 125, 145]
        long_list = [326.5, 326.5, 326.5]

                
        for index in range(self.enabled_obs):
            bp, sp = self._produce_vehicle_blueprint(lat_list[index], long_list[index], 180)
            self.bp_obs_list.append(bp)
            self.spawn_point_obs_list.append(sp)
        for index in range(self.enabled_obs):
            self.obs_list.append(self.world.spawn_actor(self.bp_obs_list[index],
                                                    self.spawn_point_obs_list[index]))
            self.obs_list[index].set_velocity(carla.Vector3D(-5,0,0))

            # self.world.tick()
            self.obs_velo_list.append(np.random.randint(3,5))
            # self.obs_agent_list.append(BasicAgent(self.obs_list[index], 
            #                                       target_speed=np.random.randint(3,5)*3.6))
            
        # for index in range(self.enabled_obs):
            # target_location = self.obs_list[index].get_transform().location
            # self.obs_agent_list[index].set_destination((target_location.x-100, target_location.y, target_location.z))
            

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
        self.collision_sensor.listen(lambda event: LeftTurn._on_collision(weak_self, event))

        
        ## configurate and spawn the camera sensors
        self.camera_transforms = [
            (carla.Transform(carla.Location(x=-1, z=20), carla.Rotation(pitch=10)), self.Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-5, z=5), carla.Rotation(pitch=10)), self.Attachment.SpringArm)]
        self.camera_transform_index = 1
        # the candidated camera type: rgb (viz_camera) and semantic (seman_camera)
        self.cameras = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}]]
                
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
        self.seman_camera.listen(lambda image: LeftTurn._parse_seman_image(weak_self, image))
        self.viz_camera.listen(lambda image: LeftTurn._parse_image(weak_self, image))


        # visualize the goal point
        xxx = carla.Location()
        xxx.x = 103
        xxx.y = 331
        xxx.z = 1
        self.world.debug.draw_point(xxx, size=0.1, color=carla.Color(r=255, g=0, b=0), life_time=1000)
        
        
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
    
    
    def step(self, action):
        
        self.world.tick()
        self.render(self.display)
        
        pygame.display.flip()
        human_control = None
        
        
        # act once per control interval
        action = action[0] if action.shape != () else action
        action = action if self.count % self.control_interval == 0 else self.previous_action_list[-1]
        
        
        # execute lateral planning & control by a third-part planner
        self.control = self.agent.run_step()
        
        
        # detect if guidance from human exists (jostick or keyboard)
        if self.intervention_type == 'joystick':
            human_control = self._parse_wheel()
        else:
            human_control = self._parse_key()
        
        self.intervention = self.joystick_intervention or self.keyboard_intervention
        

        # longitudinal control by RL's action if no human guidance else by human action
        if self.intervention:
            self.control.throttle = np.float(abs(human_control))*0.5 if human_control>=0 else 0
            self.control.brake = np.float(abs(human_control))*0.5 if human_control<0 else 0

        else:
            self.control.throttle = np.float(abs(action))*0.5 if action>=0 else 0
            self.control.brake = np.float(abs(action))*0.5 if action<0 else 0
            
        
        # calculate the ego agent's information
        v_ego = self._calculate_velocity(self.ego_vehicle)
        y_ego = self.ego_vehicle.get_location().y
        x_ego = self.ego_vehicle.get_location().x
        
        
        # achieve the control for the ego vehicle
        self.control.brake = (v_ego - 5) if v_ego>6 else self.control.brake   # limit the ego speed 
        self.ego_vehicle.apply_control(self.control)
        
        
        # record the human intervention histroy
        self.intervene_history.append(human_control)
        
        
        # record the adopted action
        adopted_action = action if not self.intervention else human_control
        self.previous_action_list.append(adopted_action)

        
        # achieve the control to the surrounding vehicles
        for index in range(len(self.obs_list)):
            obs_command = carla.VehicleControl()
            obs_command.steer = 0
            obs_velocity_diff = self.obs_velo_list[index] - abs(self.obs_list[index].get_velocity().x)
            obs_command.throttle = min(1, 0.4*obs_velocity_diff) if obs_velocity_diff>0 else 0
            obs_command.brake = min(1, -0.4*obs_velocity_diff) if obs_velocity_diff<0 else 0
            if index>0:
                dis_to_front = abs(self.obs_list[index-1].get_location().x - self.obs_list[index].get_location().x)
                v_front = abs(self.obs_list[index-1].get_velocity().x)
                v_current = abs(self.obs_list[index].get_velocity().x)
                if dis_to_front < 8 and v_current > v_front:
                    obs_command.brake = (8-dis_to_front)/4
                    
            if self.conservative_surrounding:
                if 323 < y_ego < 328:
                    obs_command.throttle = 0
                    obs_command.brake = 0.2
            self.obs_list[index].apply_control(obs_command)
        
        
        # obtain the state transition and other variables after taking the action (control command)
        next_observation, other_indicators = self.get_observation()


        # detect if the step is the terminated step, by considering: collision and episode fininsh
        collision = self.get_collision_history()[1]
        finish = (y_ego>330) and (x_ego>103)
        done = collision or finish or self.count>400/12*self.frame
        
        
        # calculate the reward signal of the step: r1-r3 distance reward, r4 terminal reward, r5-r6 smooth reward
        reward = finish*10 - collision*100 -(self.count>400)*100 - 0.5*(v_ego<0.2)
        
        
        # calculate associated reward (penalty)
        penalty_list = []
        
        if x_ego-self.obs_list[0].get_location().x+2 < 0:
            index_obs_concerned = 0
        elif x_ego-self.obs_list[1].get_location().x-1 < 0:
            index_obs_concerned = 1
        elif -5<x_ego-self.obs_list[2].get_location().x-2 < 0:
            index_obs_concerned = 2
        else:
            index_obs_concerned = None
        self.index_obs_concerned = index_obs_concerned
        
        if index_obs_concerned is not None:
            t_colli = (self.obs_list[index_obs_concerned].get_location().x+2 - 90) / abs(self.obs_list[index_obs_concerned].get_velocity().x+0.001)
            self.v_upp = (329.5 - y_ego)/t_colli
            self.v_low = (322.5 - y_ego)/t_colli
        else:
            self.v_upp, self.v_low = 0, 0
        
        if y_ego<323.5:  # when the ego agent has yet reach the target lane
            for index in range(len(self.obs_list)):
                if x_ego < (self.obs_list[index].get_location().x - 1.5):
                    dis = ((x_ego - self.obs_list[index].get_location().x)**2+(y_ego-self.obs_list[index].get_location().y)**2)**(1/2)
                    if v_ego > 0.1: # calculate ttc to the surrounding vehicle 
                        ttc = np.clip(dis/v_ego, 0,5)*0.1
                        penalty_list.append(ttc)
        # risk is the ttc to the nearest surrounding vehicle
        risk = min(penalty_list) if len(penalty_list) != 0 else None
        risk = risk if y_ego<323.5 else None
        self.risk = risk
        # penalty is added to the reward term
        reward = reward + risk if risk is not None else reward + 0.5
        
        
        # clip the reward value into [-10,10]
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
            self.terminate_position = y_ego
            self.post_process()

        
        return next_observation, human_control, reward, self.intervention, done, physical_variables
    

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
        self.destroy()
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
                elif event.button == 1:
                    self._toggle_camera()
                elif event.button == 2:
                    self._next_sensor()
    
    def _parse_key(self):
        # Detect if there is a human action from the keyboard
        keys = pygame.key.get_pressed()
        
        # Process human action from the keyboard
        if (keys[K_UP] or keys[K_w]) and (not self.w_pressed):
            self.human_default_throttle = 0.01
            self.w_pressed = 1
        elif (keys[K_UP] or keys[K_w]) and (self.w_pressed):
            self.human_default_throttle += 0.05
        elif (not keys[K_UP] and not keys[K_w]):
            self.human_default_throttle = 0.0
            self.w_pressed = 0
        
        if (keys[K_DOWN] or keys[K_s]) and (not self.s_pressed):
            self.human_default_brake = 0.01
            self.s_pressed = 1
        elif (keys[K_DOWN] or keys[K_s]) and (self.s_pressed):
            self.human_default_brake += 0.05
        elif (not keys[K_DOWN] and not keys[K_s]):
            self.human_default_brake = 0.0
            self.s_pressed = 0
        
        
        if (keys[K_DOWN] or keys[K_s]) or (keys[K_UP] or keys[K_w]):

            human_throttle = np.clip(np.float(self.human_default_throttle),0,1)
            human_brake = np.clip(np.float(self.human_default_brake),0,1)
        
            human_action = human_throttle - human_brake
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
        
        if self.joystick_intervention:
            # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
            # For the steering, it seems fine as it is
            throttleCmd = 1.6 + (2.05 * math.log10(
                -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
            if throttleCmd <= 0:
                throttleCmd = 0
            elif throttleCmd > 1:
                throttleCmd = 1
            elif 0.62 < throttleCmd < 0.623:
                throttleCmd = 0
    
            brakeCmd = 1.6 + (2.05 * math.log10(
                -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
            if brakeCmd <= 0:
                brakeCmd = 0
            elif brakeCmd > 1:
                brakeCmd = 1
            elif 0.62 < brakeCmd < 0.623:
                brakeCmd = 0
                
            # self.control.steer = steerCmd
            human_action = throttleCmd - brakeCmd
        else:
            human_action = None
            
        return human_action
        
    def _produce_vehicle_blueprint(self, x, y, yaw=0, color=0):

        blueprint_library = self.world.get_blueprint_library()
        
        if self.vehicle_type != 'single':
            # ran = np.random.rand()
            # if ran<0.25:
            #     bp = blueprint_library.filter('vehicle.audi.tt')[0]
            # elif ran<0.5:
            #     bp = blueprint_library.filter('vehicle.audi.tt')[0]
            # elif ran<0.75:
            #     bp = blueprint_library.filter('vehicle.audi.tt')[0]
            # else:
            #     bp = blueprint_library.filter('vehicle.audi.tt')[0]
            bp = blueprint_library.filter('vehicle.*')[np.random.randint(0,8)]
            while (int(bp.get_attribute('number_of_wheels')) != 4) or (bp.id == 'vehicle.bmw.grandtourer') or (bp.id == 'vehicle.lincoln.mkz2017'):
                bp = blueprint_library.filter('vehicle.*')[np.random.randint(0,20)]
        else:
            bp = blueprint_library.filter('vehicle.audi.etron')[0]
        
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))

        spawn_point = self.world.get_map().get_spawn_points()[0]
        spawn_point.location.x = x
        spawn_point.location.y = y
        spawn_point.location.z = 0.5
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

