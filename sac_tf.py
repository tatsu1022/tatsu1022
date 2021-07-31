from re import I
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
import numpy as np
import os

from SAC.DatasetHandler.MultiStepReplayMemory import MultiStepMemory
from SAC.SoftActorCritic.TotalRewardService import TotalRewardService
from SAC.UserInterface.UserDefinedSettings import UserDefinedSettings
from SAC.Environment.EnvironmentFactory import EnvironmentFactory


userDefinedSettings = UserDefinedSettings()

tf.keras.backend.set_floatx('float64')


class Actor(Model):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    epsilon = 1e-6

    def __init__(self, input_dim, action_dim, userDefinedSettings=userDefinedSettings):
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_units = userDefinedSettings.hidden_units
        self.optimizer = tf.keras.optimizers.Adam(lr=userDefinedSettings.learning_rate)
        self.squash_action_scale, self.squash_action_shift = self.calc_squash_parameters(userDefinedSettings.ACTION_MIN, userDefinedSettings.ACTION_MAX)

        self.dense1 = layers.Dense(self.hidden_units[0], activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense1')
        self.dense2 = layers.Dense(self.hidden_units[1], activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense2')
        self.means = layers.Dense(self.action_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='means')
        self.log_stds = layers.Dense(self.action_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='log_stdss')

    def call(self, state):
        a1 = self.dense1(state)
        a2 = self.dense2(a1)
        means = self.means(a2)
        log_stds = self.log_stds(a2)
        stds = tf.math.exp(log_stds)
        
        normals = tfp.distributions.Normal(means, stds)
        # Apply the tanh squashing to keep the gaussian bounded in (-1,1)
        actions_not_squashed = normals.sample()
        actions_squashed = self.squash_action(actions_not_squashed)
        tanh_2 = actions_squashed
        # calculate the log probability
        log_probs = normals.log_prob(actions_not_squashed) - tf.math.log(1 - tanh_2**2 + self.epsilon)
        entropies = -tf.reduce_sum(log_probs, axis=1, keepdims=True)

        return actions_squashed, entropies, tf.tanh(means)

    def update(self, batch, critic, alpha): # calc_loss()
        states, actions, rewards, next_states, dones = batch
        with tf.GradientTape() as tape:
            sampled_action, entropy, _ = self.call(states)
            sampled_state_action = tf.concat([states, sampled_action], axis=1)
            q1, q2 = critic(sampled_state_action)
            q = tf.minimum(q1, q2)
            soft_q = -q - alpha * entropy
            loss = tf.reduce_mean(soft_q)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss, entropy
    
    def calc_squash_parameters(self, actions_min, action_max):
        squash_action_scale = tf.constant((action_max - actions_min) / 2.0, dtype=tf.float64)
        squash_action_shift = tf.constant(actions_min + squash_action_scale, dtype=tf.float64)
        return squash_action_scale, squash_action_shift

    def squash_action(self, actions_not_squashed):
        return self.squash_action_scale * tf.tanh(actions_not_squashed) + self.squash_action_shift
      
    @property
    def trainable_variables(self):
        return self.dense1.trainable_variables + \
                self.dense1.trainable_variables + \
                self.means.trainable_variables + \
                self.log_stds.trainable_variables


class Critic:
    
    def __init__(self, input_dim, action_dim, userDefinedSettings=userDefinedSettings):
        super().__init__()

        self.optimizer = tf.keras.optimizers.Adam(lr=userDefinedSettings.learning_rate)
        self.soft_update_rate = userDefinedSettings.soft_update_rate
        self.hidden_units = userDefinedSettings.hidden_units

        self.q_network = TwinnedQNetwork(input_dim, action_dim, userDefinedSettings)
        self.target_network = TwinnedQNetwork(input_dim, action_dim, userDefinedSettings)

        # / tf.stop_gradient(self.target_network)
        self.hard_update()
        
    def hard_update(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def soft_update(self):
        weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        for idx, w in enumerate(weights):
            target_weights[idx] *= (1 - self.soft_update_rate)
            target_weights[idx] += self.soft_update_rate * w
        self.target_network.set_weights(target_weights)

    def update(self, batch, policy, alpha, gamma_n): # calc_loss()
        current_q1, current_q2 = self.q_network.calc_current_q(*batch)
        q1_loss, q2_loss = self.calc_loss_and_update_params(*batch, policy, alpha, gamma_n)

        mean_q1 = tf.reduce_mean(tf.stop_gradient(current_q1)).cpu().numpy()
        mean_q2 = tf.reduce_mean(tf.stop_gradient(current_q2)).cpu().numpy()
        
        return q1_loss, q2_loss, mean_q1, mean_q2

    def calc_loss_and_update_params(self, states, actions, rewards, next_states, dones, policy, alpha, gamma_n): #
        with tf.GradientTape() as tape1:
            state_actions = tf.concat([states, actions], axis=1)
            current_q1, _ = self.q_network(state_actions)   # self.q_network.calc_current_q
            
            next_actions, next_entropies, _ = policy(next_states)   # self.target_network.calc_target_q
            next_state_actions = tf.concat([next_states, next_actions], axis=1)
            next_q1, next_q2 = self.target_network(next_state_actions)
            next_q = tf.minimum(next_q1, next_q2) + alpha * next_entropies   # Q(s',a') - alpha * log pi(a'|s')
            target_q = tf.stop_gradient(rewards + (1.0 - dones) * gamma_n * next_q)   # r(s,a) + gamma * (Q(s',a') - alpha * log pi(a'|s'))
            # tf.stop_gradient(op) is equivalent to with torch.no_grad():op   (?)

            q1_loss = self.q_network.Q1.loss_function(current_q1, target_q)

        with tf.GradientTape() as tape2:
            state_actions = tf.concat([states, actions], axis=1)
            _, current_q2 = self.q_network(state_actions)   # self.q_network.calc_current_q
            
            next_actions, next_entropies, _ = policy(next_states)   # self.target_network.calc_target_q
            next_state_actions = tf.concat([next_states, next_actions], axis=1)
            next_q1, next_q2 = self.target_network(next_state_actions)
            next_q = tf.minimum(next_q1, next_q2) + alpha * next_entropies   # Q(s',a') - alpha * log pi(a'|s')
            target_q = tf.stop_gradient(rewards + (1.0 - dones) * gamma_n * next_q)   # r(s,a) + gamma * (Q(s',a') - alpha * log pi(a'|s'))

            q2_loss = self.q_network.Q2.loss_function(current_q2, target_q)

        grads1 = tape1.gradient(q1_loss, self.q_network.Q1.trainable_variables)
        self.optimizer.apply_gradients(zip(grads1, self.q_network.Q1.trainable_variables))

        grads2 = tape2.gradient(q2_loss, self.q_network.Q2.trainable_variables)
        self.optimizer.apply_gradients(zip(grads2, self.q_network.Q2.trainable_variables))

        return q1_loss.cpu().numpy(), q2_loss.cpu().numpy()
        

class TwinnedQNetwork(Model):
    
    def __init__(self, input_dim, action_dim, userDefinedSettings=userDefinedSettings):
        super().__init__()

        self.Q1 = QNetwork(input_dim, action_dim, userDefinedSettings)
        self.Q2 = QNetwork(input_dim, action_dim, userDefinedSettings)

        self.optimizer = tf.keras.optimizers.Adam(lr=userDefinedSettings.learning_rate)

    def call(self, x):   # state_actions: x = tf.concat([states, actions], axis=1)
                        # it was not valid to implement (state, action) as inputs for call() of Model
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2
    
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        state_actions = tf.concat([states, actions], axis=1)
        current_q1, current_q2 = self.call(state_actions)
        return current_q1, current_q2

    def calc_loss_and_update_params(self, states, actions, rewards, next_states, dones, policy, alpha, gamma_n): #
        with tf.GradientTape() as tape1:
            state_actions = tf.concat([states, actions], axis=1)
            current_q1, _ = self.call(state_actions)
            next_actions, next_entropies, _ = policy(next_states)
            next_state_actions = tf.concat([next_states, next_actions], axis=1)
            next_q1, next_q2 = self.call(next_state_actions)

            next_q = tf.minimum(next_q1, next_q2) + alpha * next_entropies   # Q(s',a') - alpha * log pi(a'|s')
            target_q = tf.stop_gradient(rewards + (1.0 - dones) * gamma_n * next_q)   # r(s,a) + gamma * (Q(s',a') - alpha * log pi(a'|s'))
            # tf.stop_gradient(op) is equivalent to with torch.no_grad():op   (?)
            q1_loss = self.Q1.loss_function(current_q1, target_q)

        with tf.GradientTape() as tape2:
            state_actions = tf.concat([states, actions], axis=1)
            _, current_q2 = self.call(state_actions)
            next_actions, next_entropies, _ = policy(next_states)
            next_state_actions = tf.concat([next_states, next_actions], axis=1)
            next_q1, next_q2 = self.call(next_state_actions)

            next_q = tf.minimum(next_q1, next_q2) + alpha * next_entropies   # Q(s',a') - alpha * log pi(a'|s')
            target_q = tf.stop_gradient(rewards + (1.0 - dones) * gamma_n * next_q)   # r(s,a) + gamma * (Q(s',a') - alpha * log pi(a'|s'))
        
            q2_loss = self.Q2.loss_function(current_q2, target_q)

        grads1 = tape1.gradient(q1_loss, self.Q1.trainable_variables)
        self.optimizer.apply_gradients(zip(grads1, self.Q1.trainable_variables))

        grads2 = tape2.gradient(q2_loss, self.Q2.trainable_variables)
        self.optimizer.apply_gradients(zip(grads2, self.Q2.trainable_variables))

        return q1_loss.cpu().numpy(), q2_loss.cpu().numpy()


class QNetwork(Model):

    def __init__(self, input_dim, action_dim, userDefinedSettings=userDefinedSettings):
        super().__init__()

        self.hidden_units = userDefinedSettings.hidden_units

        self.dense1 = layers.Dense(self.hidden_units[0], activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense1')
        self.dense2 = layers.Dense(self.hidden_units[1], activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense2')
        self.out = layers.Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='output')
       
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def call(self, x):
        a1 = self.dense1(x)
        a2 = self.dense2(a1)
        q = self.out(a2)
        return q

    def calc_loss(self, current_q, target_q):
        return self.loss_function(current_q, target_q).cpu().numpy()

    @property
    def trainable_variables(self):
        return self.dense1.trainable_variables + \
                self.out.trainable_variables + \
                self.dense2.trainable_variables


class EntropyTerm:
    
    def __init__(self, ACTION_DIM, userDefinedSettings=userDefinedSettings):
        self.entropy_tuning = userDefinedSettings.entropy_tuning

        if self.entropy_tuning:
            target_entropy = ACTION_DIM[0] * userDefinedSettings.entropy_tuning_scale
            target_entropy = (target_entropy, )
            self.target_entropy = -tf.reduce_prod(tf.constant(target_entropy, dtype=tf.float64)).cpu().numpy()

            self.log_alpha = tf.Variable(0.0, dtype=tf.float64)
            self.alpha = tf.math.exp(self.log_alpha)
            self.optimizer = tf.keras.optimizers.Adam(lr=userDefinedSettings.learning_rate)
        else:
            self.alpha = tf.constant(userDefinedSettings.entropy_coefficient, dtype=tf.float64)
    
    def update(self, entropies):
        if self.entropy_tuning:
            with tf.GradientTape() as tape:
                entropy_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(self.target_entropy - entropies))
            
            variables = [self.log_alpha]
            grads = tape.gradient(entropy_loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))

            self.alpha = tf.math.exp(self.log_alpha)
            return entropy_loss
        else:
            return None
           

class SoftActorCritic:

    def __init__(self, env, userDefinedSettings=userDefinedSettings):
        self.num_steps = userDefinedSettings.num_steps
        self.soft_update_rate = userDefinedSettings.soft_update_rate
        self.batch_size = userDefinedSettings.batch_size
        self.start_steps = userDefinedSettings.start_steps
        self.gamma_n = userDefinedSettings.gamma ** userDefinedSettings.multi_step_reward_num
        self.entropy_tuning = userDefinedSettings.entropy_tuning
        self.grad_clip = userDefinedSettings.grad_clip
        self.updates_per_step = userDefinedSettings.updates_per_step
        # self.log_interval = userDefinedSettings.log_interval
        self.target_update_interval = userDefinedSettings.target_update_interval
        self.evaluate_interval = userDefinedSettings.evaluate_interval
        self.run_num_per_evaluate = userDefinedSettings.run_num_per_evaluate
        self.env = env
        self.learning_episode_num = userDefinedSettings.learning_episode_num
        self.log_dir = userDefinedSettings.LOG_DIRECTORY

        self.STATE_DIM, self.ACTION_DIM = self.env.get_state_action_space()

        self.actor = Actor(self.STATE_DIM[0], self.ACTION_DIM[0], userDefinedSettings)
        self.critic = Critic(self.STATE_DIM[0], self.ACTION_DIM[0], userDefinedSettings)
        self.initial_build()
        """
        self.actor.build(input_shape=(None, self.STATE_DIM[0]))    # (batch, input)
        self.critic.q_network.build(input_shape=(self.batch_size, self.STATE_DIM[0]+self.ACTION_DIM[0]))
        self.critic.target_network.build(input_shape=(self.batch_size, self.STATE_DIM[0]+self.ACTION_DIM[0]))

        self.actor.summary()
        print()
        self.critic.q_network.summary()
        print()
        self.critic.target_network.summary()
        print()
        """

        self.entropyTerm = EntropyTerm(self.ACTION_DIM, userDefinedSettings)    # alpha

        self.memory = MultiStepMemory(self.STATE_DIM, self.ACTION_DIM, userDefinedSettings) # replay pool

        self.log_dir = userDefinedSettings.LOG_DIRECTORY
        self.model_dir = os.path.join(self.log_dir, 'model')
        self.summary_dir = os.path.join(self.log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = tf.summary.create_file_writer(self.summary_dir)
        self.totalRewardService = TotalRewardService(userDefinedSettings)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.max_episodes_steps = self.env.get_max_episode_steps()

    def initial_build(self): # initialize model input sizezs
        state = tf.expand_dims(tf.constant(self.env.reset(), dtype=tf.float64), axis=0)
        action_, _, _ = self.actor(state)
        action = tf.expand_dims(action_.cpu().numpy().reshape(-1), axis=0)

        state_action = tf.concat([state, action], axis=1)

        self.critic.q_network(state_action)
        self.critic.target_network(state_action)

        self.env.reset()

    def run(self):
        for _ in range(self.learning_episode_num):
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and self.steps >= self.start_steps

    def choose_action(self, state):
        if self.start_steps > self.steps:
            action = self.env.random_action_sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        state = tf.expand_dims(tf.constant(state, dtype=tf.float64), axis=0)
        action, _, _ = self.actor(tf.stop_gradient(state))    # return actions_squashed, entropies, tf.tanh(mean)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        state = tf.expand_dims(tf.constant(state, dtype=tf.float64), axis=0)
        _, _, action = self.actor(tf.stop_gradient(state))
        return action.cpu().numpy().reshape(-1)

    def train_episode(self):
        self.episodes += 1
        total_reward = 0.
        episode_steps = 0
        state = self.env.reset()
        done = False

        while not done and episode_steps < self.max_episodes_steps:
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            total_reward += reward         

            self.memory.append(state, action, reward, next_state, done)

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            state = next_state

            # print('Episodes:{}, Steps:{}, Reward:{}'.format(self.episodes,episode_steps,reward))
        
        print('Total Reward:{}'.format(total_reward))
        self.totalRewardService.append_train(total_reward)

        if self.episodes % self.evaluate_interval == 0:
            self.evaluate()

        if self.decide_model_save():
            print('\n### model updated!!')
            self.save_models()
        else:
            print('\n### model keep')

        train_total_reward = self.totalRewardService.get_train_latest()
        with self.writer.as_default():
            tf.summary.scalar('reward/train', train_total_reward, self.steps)
            self.writer.flush()

    def decide_model_save(self):
        return self.totalRewardService.check_reward_peak()

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            self.critic.soft_update()
        
        batch = self.memory.sample(self.batch_size)

        q1_loss, q2_loss, mean_q1, mean_q2 = self.critic.update(batch, self.actor, self.entropyTerm.alpha, self.gamma_n)
        policy_loss, entropies = self.actor.update(batch, self.critic.q_network, self.entropyTerm.alpha)
        entropy_loss = self.entropyTerm.update(entropies)

        with self.writer.as_default():            
            if entropy_loss is not None:
                tf.summary.scalar('loss/alpha', tf.stop_gradient(entropy_loss).numpy(), self.steps)
            tf.summary.scalar('loss/Q1', tf.stop_gradient(q1_loss).numpy(), self.learning_steps)
            tf.summary.scalar('loss/Q2', tf.stop_gradient(q2_loss).numpy(), self.learning_steps)
            tf.summary.scalar('loss/policy', tf.stop_gradient(policy_loss).numpy(), self.learning_steps)
            tf.summary.scalar('stats/alpha', tf.stop_gradient(self.entropyTerm.alpha).numpy(), self.learning_steps)
            tf.summary.scalar('stats/mean_Q1', mean_q1, self.learning_steps)
            tf.summary.scalar('stats/mean_Q2', mean_q2, self.learning_steps)
            tf.summary.scalar('stats/entropy', tf.reduce_mean(tf.stop_gradient(entropies)).numpy(), self.learning_steps)
            self.writer.flush()


    def evaluate(self):
        episodes = self.run_num_per_evaluate
        total_rewards = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.env.reset()
            total_reward = 0.
            done = False
            steps = 0
            while not done and steps < self.max_episodes_steps:
                steps += 1
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                print('Test, Episodes:{}, Steps:{}, Reward:{}'.format(self.episodes ,steps, reward))
            
            total_rewards[i] = total_reward

        mean_total_reward = np.mean(total_rewards)

        with self.writer.as_default():
            tf.summary.scalar('reward/test', mean_total_reward, self.steps)
            self.writer.flush()
            
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'reward: {mean_total_reward:<5.1f}')
        print('-' * 60)

        self.totalRewardService.append_test(mean_total_reward)
   
    def save_models(self):
        tf.keras.models.save_model(self.actor, os.path.join(self.model_dir, 'policy'))
        tf.keras.models.save_model(self.critic.q_network, os.path.join(self.model_dir, 'critic'))
        tf.keras.models.save_model(self.critic.target_network, os.path.join(self.model_dir, 'critic_target'))
        
        # tf.keras.models.save_model()
        # tf.keras.models.load_model(path)

        # my_model = SubclassedModel()
        # my_model.save(keras_model_path)  # ERROR! 
        #tf.saved_model.save(my_model, saved_model_path)
        # https://www.tensorflow.org/tutorials/distribute/save_and_load?hl=ja

    def __del__(self):
        self.writer.close()
        self.env.close()


class LearningService(object):

    def run(self):
        userDefinedSettings = UserDefinedSettings()
        environmentFactory = EnvironmentFactory(userDefinedSettings)
        env = environmentFactory.generate()

        agent = SoftActorCritic(env=env, userDefinedSettings=userDefinedSettings)

        agent.run()

if __name__ == '__main__':
    learningService = LearningService()
    learningService.run()