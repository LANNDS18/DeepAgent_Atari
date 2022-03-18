import io

import cv2
import numpy as np
import pyglet
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from DeepAgent.policy import DuelingPolicy
from DeepAgent.utils import testEnvWrapper, GameEnv, testPolicyWrapper
from atari_config import PongConfig


def display_nparray(arr, max_width=500):
    assert len(arr.shape) == 3

    height, width, _channels = arr.shape

    if width > max_width:
        scale = max_width / width
        width = int(scale * width)
        height = int(scale * height)

    image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
    pyglet.gl.glTexParameteri(pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MAG_FILTER, pyglet.gl.GL_NEAREST)
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
        pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((7, 7))
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) / 255

    return heatmap


class DeepAgent_Vis(pyglet.window.Window):
    def __init__(self, policy, width=1400, height=720, caption="RL Visualizer", resizable=False):
        super().__init__(width, height, caption, resizable)
        self.policy = policy
        self.font = 'Futura'
        self.gl_color = (3./255., 41./255., 81/255., 1)
        self.font_color = (205, 179, 128, 244)
        self.set_minimum_size(400, 300)
        self.frame_rate = 1 / 60
        self.max_q_val = 0.1
        self.min_q_val = -0.1
        self.fps_display = pyglet.window.FPSDisplay(self)
        self.fps_display.label.x = self.width - 100
        self.fps_display.label.y = self.height - 50

        # For drawing screens
        self.render_img = np.ones((210, 160, 3))

        # For keeping simulating the game
        self.done = True
        self.state = np.ones((84, 84, 4))
        self.eval_rewards = []
        self.evaluate_frame_number = 0
        self.episode_reward_sum = [0]

        self.q_vals = [0] * env.env.action_space.n
        self.values = []

        # Text
        self.human_title = pyglet.text.Label('Game Rendered Screen',
                                             font_size=15, color=self.font_color, font_name=self.font,
                                             x=10, y=self.height - 20, anchor_y='center')
        self.q_val_title = pyglet.text.Label('Action Q',
                                             font_size=15, color=self.font_color, font_name=self.font,
                                             x=500, y=self.height - 20, anchor_y='center')
        self.agent_title = pyglet.text.Label('Agent Vision',
                                             font_size=15, color=self.font_color, font_name=self.font,
                                             x=10, y=235, anchor_y='center')

        self.heatmap_title = pyglet.text.Label('Agent Attention',
                                               font_size=15, color=self.font_color, font_name=self.font,
                                               x=1000, y=self.height - 140, anchor_y='center')

        self.action_titles = []

        for i, action in enumerate(env.env.unwrapped.get_action_meanings()):
            self.action_titles.append(
                pyglet.text.Label(action, font_size=10, color=self.font_color, font_name=self.font,
                                  x=0, y=0, anchor_x='center'))

    def value_his(self):
        dpi_res = min(self.width, self.height) / 10
        fig = Figure((500 / dpi_res, 230 / dpi_res), dpi=dpi_res)
        ax = fig.add_subplot(111)

        # Set up plot
        ax.set_title('Episode Reward', fontsize=15)
        ax.set_xticklabels([])
        ax.set_ylabel('V(s)')
        ax.plot(self.episode_reward_sum[:])  # plot values

        w, h = fig.get_size_inches()
        dpi_res = fig.get_dpi()
        w, h = int(np.ceil(w * dpi_res)), int(np.ceil(h * dpi_res))
        canvas = FigureCanvasAgg(fig)
        pic_data = io.BytesIO()
        canvas.print_raw(pic_data, dpi=dpi_res)
        img = pyglet.image.ImageData(w, h, 'RGBA', pic_data.getvalue(), -4 * w)
        img.blit(375, 265)

    def action_q(self):
        self.q_val_title.draw()
        length = 80
        starting_x = 400
        for i, q_val in enumerate(self.q_vals):
            if q_val > self.max_q_val:
                self.max_q_val = q_val
            elif q_val < self.min_q_val:
                self.min_q_val = q_val

            # Draw square representation q-val
            x_value = starting_x + i * (length + 10)  # x-coordinate to draw square
            color = (150 ** (q_val * 2)) / (sum([150 ** (q * 2) for q in self.q_vals]) + 0.0001)
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                                 ('v2f', (
                                     x_value, self.height - 50, x_value + length, self.height - 50,
                                     x_value + length,
                                     self.height - length - 50, x_value, self.height - length - 50)),
                                 ('c3f', (
                                     color, color, color, color, color, color, color, color, color, color, color,
                                     color)))

            # Draw action label
            pyglet.gl.glTranslatef(x_value + length / 2, self.height - 100 - length, 0.0)
            pyglet.gl.glRotatef(-90.0, 0.0, 0.0, 1.0)
            self.action_titles[i].draw()
            pyglet.gl.glRotatef(90.0, 0.0, 0.0, 1.0)
            pyglet.gl.glTranslatef(-(x_value + length / 2), -(self.height - 100 - length), 0.0)

    def render_game(self):
        self.human_title.draw()
        base_dimensions = (210, 160)
        scale = 2
        display_nparray(cv2.resize(
            self.render_img,
            dsize=(int(base_dimensions[1] * scale), int(base_dimensions[0] * scale)),
            interpolation=cv2.INTER_CUBIC)) \
            .blit(50, self.height - base_dimensions[0] * scale - 50)

    def agent_vision(self):
        self.agent_title.draw()
        base_dimensions = (84, 84)
        scale = 2.5
        state_images = [np.repeat(self.state[:, :, i, np.newaxis], 3, axis=2) for i in
                        range(self.state.shape[-1])]
        for i, state_image in enumerate(state_images):
            display_nparray(cv2.resize(state_image,
                                       dsize=(int(base_dimensions[1] * scale), int(base_dimensions[0] * scale)),
                                       interpolation=cv2.INTER_CUBIC)) \
                .blit(10 + i * (84 * scale + 5), 10)

    def agent_attention(self):
        self.heatmap_title.draw()
        base_dimensions = (84, 84)
        intensity = 0.1
        scale = 10

        processed_frame = np.repeat(self.state[:, :, 3, np.newaxis], 3, axis=2)
        heatmap = generate_heatmap(self.state, self.policy.model)

        img = (heatmap * 255 * intensity + processed_frame * 0.8).astype(np.uint8)

        display_nparray(cv2.resize(img + (heatmap * 255 * intensity).astype(np.uint8),
                                   dsize=(int(base_dimensions[1] * scale), int(base_dimensions[0] * scale)),
                                   interpolation=cv2.INTER_CUBIC)).blit(880, 60)

    def on_draw(self):
        self.clear()
        pyglet.gl.glClearColor(32/255., 20/255., 40/255., 1)
        pyglet.gl.glTexParameteri(pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MAG_FILTER, pyglet.gl.GL_NEAREST)
        pyglet.gl.glTexParameteri(pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MIN_FILTER, pyglet.gl.GL_NEAREST)

        self.switch_to()
        self.dispatch_events()
        self.render_game()
        self.value_his()
        self.action_q()
        self.agent_vision()
        self.agent_attention()

    def update(self, dt):
        if self.done:
            self.state = env.reset()
            self.episode_reward_sum = [0]
            self.done = False

        state = tf.expand_dims(self.state, axis=0)
        q_vals = self.policy.predict(tf.constant(state)).numpy()[0]
        action = np.argmax(q_vals)

        state, reward, done, _ = env.step(action)
        self.render_img = env.render(mode='rgb_array')
        self.q_vals = q_vals
        self.state = state
        self.done = done
        self.values.append(q_vals[action])
        self.evaluate_frame_number += 1
        self.episode_reward_sum.append(reward + self.episode_reward_sum[-1])

        if self.done:
            self.eval_rewards.append(self.episode_reward_sum)
            self.values = []


if __name__ == "__main__":
    config = PongConfig
    # Create environment
    env = testEnvWrapper(config, GameEnv)
    print("The environment has the following {} actions: {}".format(env.env.action_space.n,
                                                                    env.env.unwrapped.get_action_meanings()))
    agent_policy = testPolicyWrapper(config, DuelingPolicy, env)
    agent_policy.load('./models/DDDQN_PongNoFrameskip-v4/best/main/')

    window = DeepAgent_Vis(agent_policy, width=1400, height=720, caption="RL Visualizer", resizable=False)

    pyglet.clock.schedule_interval(window.update, window.frame_rate)
    pyglet.app.run()
