import tensorflow as tf
from DeepAgent.agents.double_dqn import DoubleDQNAgent
from DeepAgent.utils.buffer import PrioritizedExperienceReplay


class D3NPERAgent(DoubleDQNAgent):

    def __init__(
            self,
            env,
            model,
            buffer,
            agent_id='PER_D3N',
            **kwargs,
    ):
        """
        Initialize networks agent.
        Args:
            env: A gym environment.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            buffer: A prioritized experience replay buffer objects
            **kwargs: kwargs passed to super classes.
        """
        super(D3NPERAgent, self).__init__(env, model, buffer, agent_id, **kwargs)
        assert (isinstance(self.buffer, PrioritizedExperienceReplay)), \
            'The buffer should be a PrioritizedExperienceReplay buffer.'

    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.env_name, batching and gradient updates.
        """
        action = self.get_action(tf.constant(self.state), tf.constant(self.epsilon, tf.float32))

        next_state, reward, done, info = self.env.step(action)
        self.buffer.append(self.state, action, reward, done, next_state)

        self.episode_reward += reward
        self.state = next_state
        self.done = done

        if self.total_step % self.model_update_freq == 0:
            indices = self.buffer.get_sample_indices()
            states, actions, rewards, dones, next_states = self.buffer.get_sample(indices)

            main_q, target_q = self.update_main_model(states, actions, rewards, dones, next_states)
            abs_error = abs(main_q - target_q)
            self.buffer.update_priorities(indices, abs_error)
