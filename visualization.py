import io

import cv2
import numpy as np
import pyglet
import tensorflow as tf
import keras.backend as K
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pyglet.gl import *

from DeepAgent.policy import DuelingPolicy, CNNPolicy
from DeepAgent.agents import DQNAgent, DoubleDQNAgent, D3NPERAgent
from DeepAgent.utils import testEnvWrapper, GameEnv, ExperienceReplay, testPolicyWrapper
from atari_config import PongConfig, DemonAttackConfig, EnduroConfig


def get_intermediate_representation(policy, state, layer_names=None, stack_state=True):
    """
    Get the output of a hidden layer inside the model.  This will be/is used for visualizing model
    Arguments:
        state: The input to the model to get outputs for hidden layers from
        layer_names: Names of the layers to get outputs from.  This can be a list of multiple names, or a single name
        stack_state: Stack `state` four times so the model can take input on a single (84, 84, 1) frame
    Returns:
        Outputs to the hidden layers specified, in the order they were specified.
    """
    # Prepare list of layers
    if isinstance(layer_names, list) or isinstance(layer_names, tuple):
        layers = [policy.model.get_layer(name=layer_name).output for layer_name in layer_names]
    else:
        layers = policy.model.get_layer(name=layer_names).output

    # Model for getting intermediate output
    temp_model = tf.keras.Model(policy.model.inputs, layers)

    # Stack state 4 times
    if stack_state:
        if len(state.shape) == 2:
            state = state[:, :, np.newaxis]
        state = np.repeat(state, 4, axis=2)

    # Put it all together
    return temp_model.predict(state.reshape((-1, policy.input_shape[0], policy.input_shape[1], 4)))


def display_nparray(arr, maxwidth=500):
    assert len(arr.shape) == 3

    height, width, _channels = arr.shape

    if width > maxwidth:
        scale = maxwidth / width
        width = int(scale * width)
        height = int(scale * height)

    image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    texture = image.get_texture()
    texture.width = width
    texture.height = height

    return texture


def generate_heatmap(frame, model):
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv_3')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(frame[np.newaxis, :, :, :])
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((7, 7))
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) / 255

    return heatmap


class VisWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        self.frame_rate = 1 / 60
        self.max_q_val = 0.1
        self.min_q_val = -0.1
        self.fps_display = pyglet.window.FPSDisplay(self)
        self.fps_display.label.x = self.width - 100
        self.fps_display.label.y = self.height - 50

        # For drawing screens
        self.game_image = np.ones((210, 160, 3))

        # For keeping simulating the game
        self.done = True
        self.state = np.ones((84, 84, 4))
        self.eval_rewards = []
        self.evaluate_frame_number = 0
        self.episode_reward_sum = 0
        self.life_lost = True

        self.q_vals = [0] * game_wrapper.env.action_space.n
        self.values = []

        # Text
        self.human_title = pyglet.text.Label('Human-Rendered Game Screen', font_size=20, color=(0, 0, 0, 255),
                                             x=10, y=self.height - 20, anchor_y='center')
        self.q_val_title = pyglet.text.Label('Q-Values', font_size=20, color=(0, 0, 0, 255),
                                             x=500, y=self.height - 20, anchor_y='center')
        self.agent_title = pyglet.text.Label('Agent-Rendered Game Screen', font_size=20, color=(0, 0, 0, 255),
                                             x=10, y=235, anchor_y='center')
        self.heatmap_title = pyglet.text.Label('Attention Heatmap', font_size=20, color=(0, 0, 0, 255),
                                               x=1000, y=self.height - 140, anchor_y='center')

        self.action_titles = []

        for i, action in enumerate(game_wrapper.env.unwrapped.get_action_meanings()):
            self.action_titles.append(
                pyglet.text.Label(action, font_size=20, color=(0, 0, 0, 255), x=0, y=0, anchor_x='center'))

    def on_draw(self):
        self.clear()
        glClearColor(1., 1., 1., 1.)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        self.switch_to()
        self.dispatch_events()

        # Draw FPS counter
        if True:
            self.fps_display.draw()

        # Display RGB "human" version of the game state
        if True:
            self.human_title.draw()
            base_dimensions = (210, 160)
            scale = 2
            display_nparray(cv2.resize(
                self.game_image,
                dsize=(int(base_dimensions[1] * scale), int(base_dimensions[0] * scale)),
                interpolation=cv2.INTER_CUBIC)) \
                .blit(50, self.height - base_dimensions[0] * scale - 50)

        # Display grayscale "machine" version of the game state (this is what the agent really sees)
        if True:
            self.agent_title.draw()
            base_dimensions = (84, 84)
            scale = 2.5
            # For some strange reason, we must render this in RGB mode (not 'L' mode as implied in the pyglet docs)
            # Because of this, we must repeat each frame 3 times to simulate RGB, despite being grayscale
            state_images = [np.repeat(self.state[:, :, i, np.newaxis], 3, axis=2) for i in
                            range(self.state.shape[-1])]
            for i, state_image in enumerate(state_images):
                display_nparray(cv2.resize(state_image,
                                           dsize=(int(base_dimensions[1] * scale), int(base_dimensions[0] * scale)),
                                           interpolation=cv2.INTER_CUBIC)) \
                    .blit(10 + i * (84 * scale + 5), 10)

        # Display q-values
        if True:
            self.q_val_title.draw()
            LENGTH = 80
            STARTING_X = 400
            for i, (q_val, label) in enumerate(zip(self.q_vals[::-1], self.action_titles[::-1])):
                if q_val > self.max_q_val:
                    self.max_q_val = q_val
                elif q_val < self.min_q_val:
                    self.min_q_val = q_val

                # Draw square representation q-val
                x_value = STARTING_X + i * (LENGTH + 10)  # x-coordinate to draw square
                color = (150 ** (q_val * 2)) / (sum([150 ** (q * 2) for q in self.q_vals]) + 0.0001)
                pyglet.graphics.draw(4, GL_QUADS,
                                     ('v2f', (
                                         x_value, self.height - 50, x_value + LENGTH, self.height - 50,
                                         x_value + LENGTH,
                                         self.height - LENGTH - 50, x_value, self.height - LENGTH - 50)),
                                     ('c3f', (
                                         color, color, color, color, color, color, color, color, color, color, color,
                                         color)))

                # Draw action label
                glTranslatef(x_value + LENGTH / 2, self.height - 100 - LENGTH, 0.0)
                glRotatef(-90.0, 0.0, 0.0, 1.0)
                label.draw()
                glRotatef(90.0, 0.0, 0.0, 1.0)
                glTranslatef(-(x_value + LENGTH / 2), -(self.height - 100 - LENGTH), 0.0)

        # Display value history (adapted from https://learning.oreilly.com/library/view/matplotlib-plotting-cookbook
        # /9781849513265/ch08s06.html)
        if False:
            dpi_res = min(self.width, self.height) / 10
            fig = Figure((500 / dpi_res, 230 / dpi_res), dpi=dpi_res)
            ax = fig.add_subplot(111)

            # Set up plot
            ax.set_title('Estimated Value over Time', fontsize=20)
            ax.set_xticklabels([])
            ax.set_ylabel('V(s)')
            ax.plot(self.values[max(len(self.values) - 200, 0):])  # plot values

            w, h = fig.get_size_inches()
            dpi_res = fig.get_dpi()
            w, h = int(np.ceil(w * dpi_res)), int(np.ceil(h * dpi_res))
            canvas = FigureCanvasAgg(fig)
            pic_data = io.BytesIO()
            canvas.print_raw(pic_data, dpi=dpi_res)
            img = pyglet.image.ImageData(w, h, 'RGBA', pic_data.getvalue(), -4 * w)
            img.blit(375, 265)

        # Display heatmap
        if True and self.evaluate_frame_number > 1:
            self.heatmap_title.draw()
            base_dimensions = (84, 84)
            INTENSITY = 0.1
            scale = 10

            processed_frame = np.repeat(self.state[:, :, 3, np.newaxis], 3, axis=2)
            heatmap = generate_heatmap(self.state, policy.model)

            img = (heatmap * 255 * INTENSITY + processed_frame * 0.8).astype(np.uint8)

            display_nparray(cv2.resize(img + (heatmap * 255 * INTENSITY).astype(np.uint8),
                                       dsize=(int(base_dimensions[1] * scale), int(base_dimensions[0] * scale)),
                                       interpolation=cv2.INTER_CUBIC)).blit(880, 60)

        self.flip()

    def update(self, dt):
        if self.done:
            self.state = game_wrapper.reset()
            self.life_lost = True
            self.episode_reward_sum = 0
            self.done = False

        self.q_vals, value = get_intermediate_representation(policy, self.state, ['tf.__operators__.add', 'value_layer'],
                                                             stack_state=False)
        self.q_vals, value = self.q_vals[0], value[0]
        action = self.q_vals.argmax()

        self.values.append(value)

        state, reward, done, _ = game_wrapper.step(action)
        self.state = state
        self.done = done
        self.evaluate_frame_number += 1
        self.episode_reward_sum += reward

        if self.done:
            self.eval_rewards.append(self.episode_reward_sum)
            self.values = []


if __name__ == "__main__":
    config = PongConfig
    # Create environment
    game_wrapper = testEnvWrapper(config, GameEnv)
    print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n,
                                                                    game_wrapper.env.unwrapped.get_action_meanings()))
    policy = testPolicyWrapper(config, DuelingPolicy, game_wrapper)

    obs = game_wrapper.render(mode='rgb_array')

    print('Finished setup.  Visualizing...')
    window = VisWindow(1400, 720, "RL Visualizer", resizable=True)
    pyglet.clock.schedule_interval(window.update, window.frame_rate)
    pyglet.app.run()
