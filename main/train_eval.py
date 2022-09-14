# Copyright (c) 2020: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

from absl import app
from absl import flags
from absl import logging

import functools
import gin
import numpy as np
import os
import tensorflow as tf
import time
import collections

import gym
import gym_carla

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import q_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from interp_e2e_driving.agents.ddpg import ddpg_agent
from interp_e2e_driving.environments import filter_observation_wrapper
from interp_e2e_driving.networks import multi_inputs_actor_rnn_network
from interp_e2e_driving.networks import multi_inputs_critic_rnn_network
from interp_e2e_driving.utils import gif_utils


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')

FLAGS = flags.FLAGS


@gin.configurable
def load_carla_env(
  env_name='carla-v0',
  discount=1.0,
  number_of_vehicles=60,
  number_of_walkers=60,
  display_size=128,
  max_past_step=1,
  dt=0.1,
  discrete=False,
  discrete_acc=[-3.0, 0.0, 3.0],
  discrete_steer=[-0.2, 0.0, 0.2],
  continuous_accel_range=[-3.0, 3.0],
  continuous_steer_range=[-0.3, 0.3],
  ego_vehicle_filter='vehicle.lincoln*',
  spectator_vehicle_filter = 'spectator',
  port=2000,
  town='Town01',
  task_mode='random',
  max_time_episode=500,
  max_waypt=12,
  obs_range=32,
  lidar_bin=0.5,
  d_behind=12,
  out_lane_thres=2.0,
  desired_speed_suprimum=4,
  desired_speed_infimum=2,
  max_ego_spawn_times=200,
  display_route=True,
  pixor_size=4,
  pixor=False,
  obs_channels=None,
  action_repeat=1):
  """Loads train and eval environments."""
  env_params = {
    'number_of_vehicles': number_of_vehicles,
    'number_of_walkers': number_of_walkers,
    'display_size': display_size,  # screen size of bird-eye render
    'max_past_step': max_past_step,  # the number of past steps to draw
    'dt': dt,  # time interval between two frames
    'discrete': discrete,  # whether to use discrete control space
    'discrete_acc': discrete_acc,  # discrete value of accelerations
    'discrete_steer': discrete_steer,  # discrete value of steering angles
    'continuous_accel_range': continuous_accel_range,  # continuous acceleration range
    'continuous_steer_range': continuous_steer_range,  # continuous steering angle range
    'ego_vehicle_filter': ego_vehicle_filter,  # filter for defining ego vehicle
    'spectator_vehicle_filter':spectator_vehicle_filter,
    'port': port,  # connection port
    'town': town,  # which town to simulate
    'task_mode': task_mode,  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': max_time_episode,  # maximum timesteps per episode
    'max_waypt': max_waypt,  # maximum number of waypoints
    'obs_range': obs_range,  # observation range (meter)
    'lidar_bin': lidar_bin,  # bin size of lidar sensor (meter)
    'd_behind': d_behind,  # distance behind the ego vehicle (meter)
    'out_lane_thres': out_lane_thres,  # threshold for out of lane
    'desired_speed_suprimum': desired_speed_suprimum,  # desired speed (m/s)
    'desired_speed_infimum':desired_speed_infimum,
    'max_ego_spawn_times': max_ego_spawn_times,  # maximum times to spawn ego vehicle
    'display_route': display_route,  # whether to render the desired route
    'pixor_size': pixor_size,  # size of the pixor labels
    'pixor': pixor,  # whether to output PIXOR observation
  }

  gym_spec = gym.spec(env_name)
  gym_env = gym_spec.make(params=env_params)

  if obs_channels:
    gym_env = filter_observation_wrapper.FilterObservationWrapper(gym_env, obs_channels)

  py_env = gym_wrapper.GymWrapper(
    gym_env,
    discount=discount,
    auto_reset=True,
  )

  eval_py_env = py_env

  if action_repeat > 1:
    py_env = wrappers.ActionRepeat(py_env, action_repeat)

  return py_env, eval_py_env


def compute_summaries(metrics,
                      environment,
                      policy,
                      train_step=None,
                      summary_writer=None,
                      num_episodes=1,
                      num_episodes_to_render=1,
                      fps=10,
                      image_keys=None):
  for metric in metrics:
    metric.reset()

  time_step = environment.reset()
  policy_state = policy.get_initial_state(environment.batch_size)

  if num_episodes_to_render:
    images = [[time_step.observation]]  # now images contain dictionary of images
  else:
    images = []

  # Get input images and latent states
  episode = 0
  counter = 0
  while episode < num_episodes:
    counter += 1
    print('----------------------------------------- test iteration ' + str(counter) + '  episode ' + str(episode) + ' -----------------------------------------')
    action_step = policy.action(time_step, policy_state)
    next_time_step = environment.step(action_step.action)
    policy_state = action_step.state

    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    for observer in metrics:
      observer(traj)

    if episode < num_episodes_to_render:
      if traj.is_boundary():
        images.append([])
      images[-1].append(next_time_step.observation)

    if counter >= 200 or traj.is_last():
      print('next test episode!')
      counter = 0
      episode += 1
      policy_state = policy.get_initial_state(environment.batch_size)

    time_step = next_time_step

  # Summarize scalars to tensorboard
  if train_step and summary_writer:
    with summary_writer.as_default():
      for m in metrics:
        tag = m.name
        tf.compat.v2.summary.scalar(name=tag, data=m.result(), step=train_step)

  # Concat input images of different episodes and generate reconstructed images.
  # Shape of images is [[images in episode as timesteps]].

  print('concating videos')
  images_is_dict = type(images[0][0]) is collections.OrderedDict
  images = pad_and_concatenate_videos(images, image_keys=image_keys, is_dict=images_is_dict)
  images = tf.image.convert_image_dtype([images], tf.uint8, saturate=True)
  images = tf.squeeze(images, axis=2)
  
  # Need to avoid eager here to avoid rasing error
  gif_summary = common.function(gif_utils.gif_summary_v2)

  # Summarize to tensorboard
  print('summarize videos for tensorboard')
  gif_summary('ObservationVideoEvalPolicy', images, 1, fps)


def pad_and_concatenate_videos(videos, image_keys, is_dict=False):
  max_episode_length = max([len(video) for video in videos])
  if is_dict:
    # videos = [[tf.concat(list(dict_obs.values()), axis=2) for dict_obs in video] for video in videos]
    videos = [[tf.concat([dict_obs[key] for key in image_keys], axis=2) for dict_obs in video] for video in videos]
  for video in videos:
    #　video contains [dict_obs of timesteps]
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  #　frames is [(each episodes obs at timestep t)]
  videos = [tf.concat(frames, axis=1) for frames in zip(*videos)]
  return videos


def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
  del init_action_stddev
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True)


class Preprocessing_Layer(tf.keras.layers.Layer):
  """Preprocessing layers for multiple source inputs."""

  def __init__(self, base_depth, feature_size, name=None):
    super(Preprocessing_Layer, self).__init__(name=name)
    self.base_depth = base_depth
    self.feature_size = feature_size
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv1 = conv(base_depth, 5, 2)
    self.conv2 = conv(2 * base_depth, 3, 2)
    self.conv3 = conv(4 * base_depth, 3, 2)
    self.conv4 = conv(8 * base_depth, 3, 2)
    self.conv5 = conv(8 * base_depth, 4, padding="VALID")

  def __call__(self, image, training=None):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)

  def get_config(self):
    config = {
      'base_depth':self.base_depth, 
      'feature_size':self.feature_size}
    return config


@gin.configurable
def train_eval(
    root_dir,
    agent_name='dqn',  # agent's name
    num_iterations=int(1e7),
    actor_fc_layers=(256, 256),
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    input_names=['camera', 'lidar'],  # names for inputs
    mask_names=['birdeye'],  # names for masks
    observations=['mask','input'],  # which observations to choose
    preprocessing_combiner=tf.keras.layers.Add(),  # takes a flat list of tensors and combines them
    actor_lstm_size=(40,),  # lstm size for actor
    critic_lstm_size=(40,),  # lstm size for critic
    actor_output_fc_layers=(100,),  # lstm output
    critic_output_fc_layers=(100,),  # lstm output
    epsilon_greedy=0.1,  # exploration parameter for DQN
    q_learning_rate=1e-3,  # q learning rate for DQN
    ou_stddev=0.2,  # exploration paprameter for DDPG
    ou_damping=0.15,  # exploration parameter for DDPG
    dqda_clipping=None,  # for DDPG
    # Params for collect
    initial_collect_steps=1000,
    collect_steps_per_iteration=1,
    replay_buffer_capacity=int(1e5),
    # Params for target update
    target_update_tau=0.005,
    target_update_period=1,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=256,
    sequence_length=4,  # number of timesteps to train model
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=100,
    # Params for summaries and logging
    train_checkpoint_interval=5000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=5000,
    log_interval=100,
    summary_interval=100,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    gpu_allow_growth=True,  # GPU memory growth
    gpu_memory_limit=None,  # GPU memory limit
    action_repeat=1):  
  # Name of observation channels for building processing layers ['camera', 'lidar', 'birdeye']
  observation_channels = []
  if('mask' in observations):
    print('adding mask in observation channels')
    observation_channels += mask_names
  if('input' in observations):
    print('adding sensors in observation channels')
    observation_channels += input_names

  # Setup GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpu_allow_growth:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpu_memory_limit:
    for gpu in gpus:
      tf.config.experimental.set_virtual_device_configuration(
          gpu,
          [tf.config.experimental.VirtualDeviceConfiguration(
              memory_limit=gpu_memory_limit)])

  # Get train and eval direction
  root_dir = os.path.expanduser(root_dir)
  now = datetime.now()
  root_dir = os.path.join(root_dir, agent_name, str(observations), str(now))

  # Get summary writers
  summary_writer = tf.summary.create_file_writer(
      root_dir, flush_millis=summaries_flush_secs * 1000)
  summary_writer.set_as_default()

  global_step = tf.compat.v1.train.get_or_create_global_step()

  # Whether to record for summary
  with tf.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    # Create Carla environment
    if agent_name == 'dqn':
      py_env, eval_py_env = load_carla_env(env_name='carla-v0', discrete=True, obs_channels=observation_channels, action_repeat=action_repeat)
    else:
      py_env, eval_py_env = load_carla_env(env_name='carla-v0', obs_channels=observation_channels, action_repeat=action_repeat)

    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Specs
    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    ## Make tf agent
    # Set up preprosessing layers for dictionary observation inputs
    preprocessing_layers = collections.OrderedDict()
    for name in observation_channels:
      preprocessing_layers[name] = Preprocessing_Layer(32,256)
    if len(observation_channels) < 2:
      preprocessing_combiner = None

    if agent_name == 'dqn':
      q_rnn_net = q_rnn_network.QRnnNetwork(
          observation_spec,
          action_spec,
          preprocessing_layers=preprocessing_layers,
          preprocessing_combiner=preprocessing_combiner,
          input_fc_layer_params=critic_joint_fc_layers,
          lstm_size=critic_lstm_size,
          output_fc_layer_params=critic_output_fc_layers)

      tf_agent = dqn_agent.DqnAgent(
          time_step_spec,
          action_spec,
          q_network=q_rnn_net,
          epsilon_greedy=epsilon_greedy,
          n_step_update=1,
          target_update_tau=target_update_tau,
          target_update_period=target_update_period,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=q_learning_rate),
          td_errors_loss_fn=common.element_wise_squared_loss,
          gamma=gamma,
          reward_scale_factor=reward_scale_factor,
          gradient_clipping=gradient_clipping,
          debug_summaries=debug_summaries,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=global_step)

    elif agent_name == 'ddpg':
      actor_rnn_net = multi_inputs_actor_rnn_network.MultiInputsActorRnnNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        input_fc_layer_params=actor_fc_layers,
        lstm_size=actor_lstm_size,
        output_fc_layer_params=actor_output_fc_layers)

      critic_rnn_net = multi_inputs_critic_rnn_network.MultiInputsCriticRnnNetwork(
        (observation_spec, action_spec),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        lstm_size=critic_lstm_size,
        output_fc_layer_params=critic_output_fc_layers)

      tf_agent = ddpg_agent.DdpgAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_rnn_net,
            critic_network=critic_rnn_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            ou_stddev=ou_stddev,
            ou_damping=ou_damping,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            dqda_clipping=dqda_clipping,
            td_errors_loss_fn=None,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars)

    elif agent_name == 'sac':
      actor_distribution_rnn_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
          observation_spec,
          action_spec,
          preprocessing_layers=preprocessing_layers,
          preprocessing_combiner=preprocessing_combiner,
          input_fc_layer_params=actor_fc_layers,
          lstm_size=actor_lstm_size,
          output_fc_layer_params=actor_output_fc_layers,
          continuous_projection_net=normal_projection_net)

      critic_rnn_net = multi_inputs_critic_rnn_network.MultiInputsCriticRnnNetwork(
        (observation_spec, action_spec),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        lstm_size=critic_lstm_size,
        output_fc_layer_params=critic_output_fc_layers)

      tf_agent = sac_agent.SacAgent(
          time_step_spec,
          action_spec,
          actor_network=actor_distribution_rnn_net,
          critic_network=critic_rnn_net,
          actor_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=actor_learning_rate),
          critic_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=critic_learning_rate),
          alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=alpha_learning_rate),
          target_update_tau=target_update_tau,
          target_update_period=target_update_period,
          td_errors_loss_fn=tf.math.squared_difference,  # make critic loss dimension compatible
          gamma=gamma,
          reward_scale_factor=reward_scale_factor,
          gradient_clipping=gradient_clipping,
          debug_summaries=debug_summaries,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=global_step)

    else:
      raise NotImplementedError

    print('initializing ' + agent_name + ' agent')
    tf_agent.initialize()
    print('initializing tf_agent finished')

    # Get replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,  # No parallel environments
        max_length=replay_buffer_capacity)
    replay_observer = [replay_buffer.add_batch]

    # Eval metrics
    eval_metrics = [
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(
        buffer_size=num_eval_episodes,
        batch_size=eval_tf_env.batch_size),
        tf_metrics.AverageReturnMetric(
          name='AverageReturnEvalPolicy', buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(
          name='AverageEpisodeLengthEvalPolicy',
          buffer_size=num_eval_episodes),
    ]
    
    # Train metrics
    env_steps = tf_metrics.EnvironmentSteps()
    average_return = tf_metrics.AverageReturnMetric(
        buffer_size=num_eval_episodes,
        batch_size=tf_env.batch_size)
    train_metrics = [
        # tf_metrics.NumberOfEpisodes(),
        env_steps,
        average_return,
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size),
    ]

    # Get policies
    # eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        time_step_spec, action_spec)
    collect_policy = tf_agent.collect_policy
    eval_policy = tf_agent.policy

    # Checkpointers
    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'train'),
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
        max_to_keep=2)
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step,
        max_to_keep=2)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)
    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    # Collect driver
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=initial_collect_steps)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=collect_steps_per_iteration)
    

    # Optimize the performance by using tf functions
    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay data.
    if (env_steps.result() == 0 or replay_buffer.num_frames() == 0):
      logging.info(
          'Initializing replay buffer by collecting experience for %d steps'
          'with a random policy.', initial_collect_steps)
      initial_collect_driver.run()

    print('initial steps finished ...!')
    compute_summaries(
      eval_metrics,
      eval_tf_env,
      eval_policy,
      train_step=global_step.numpy(),
      summary_writer=summary_writer,
      num_episodes=1,
      num_episodes_to_render=1,
      fps=10,
      image_keys=observation_channels)
    
    print('defining buffers for training')
    # Dataset generates trajectories with shape [Bxslx...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=sequence_length + 1).prefetch(3)
    iterator = iter(dataset)

    # Get train step
    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)
    train_step = common.function(train_step)

    # Training initializations
    time_step = None
    time_acc = 0
    env_steps_before = env_steps.result().numpy()

    print('start training')
    # Start training
    for iteration in range(num_iterations):
      print('----------------------------------------- iteration ' + str(iteration) + ' -----------------------------------------')
      start_time = time.time()

      # Run collect
      time_step, _ = collect_driver.run(time_step=time_step)

      # Train an iteration
      for _ in range(train_steps_per_iteration):
        train_step()

      time_acc += time.time() - start_time

      # Get training metrics
      for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=env_steps.result())

      global_step_val = global_step.numpy()
      
      # Log training information
      if global_step_val % log_interval == 0:
        logging.info('env steps = %d, average return = %f', env_steps.result(),
                     average_return.result())
        env_steps_per_sec = (env_steps.result().numpy() -
                             env_steps_before) / time_acc
        logging.info('%.3f env steps/sec', env_steps_per_sec)
        tf.summary.scalar(
            name='env_steps_per_sec',
            data=env_steps_per_sec,
            step=env_steps.result())
        time_acc = 0
        env_steps_before = env_steps.result().numpy()

      # Evaluation
      if global_step_val % eval_interval == 0:
        # Log evaluation metrics
        compute_summaries(
          eval_metrics,
          eval_tf_env,
          eval_policy,
          train_step=global_step_val,
          summary_writer=summary_writer,
          num_episodes=1,
          num_episodes_to_render=1,
          fps=10,
          image_keys=observation_channels)

      # Save checkpoints
      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)

      if global_step_val % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=global_step_val)

def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file,None)
  train_eval('logs')


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)