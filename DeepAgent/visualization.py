import io
from collections import deque

import cv2
import numpy as np
import pyglet
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Futura']


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


class DeepAgent_Vis(pyglet.window.Window):
    def __init__(self, agent_id, policy, env, max_episode=5, width=1350, height=720, caption="RL Visualizer",
                 resizable=False):
        super().__init__(width, height, caption, resizable)
        self.id = agent_id
        self.policy = policy
        self.env = env
        self.max_episode = max_episode
        self.font = 'Futura'
        self.font_color_light = (255, 222, 0, 255)
        self.font_color_dark = (253, 178, 4, 240)
        self.frame_rate = 1 / 60

        # For drawing screens
        self.render_img = np.zeros(shape=env.unwrapped.observation_space.shape)

        # For keeping simulating the game
        self.done = True
        self.state = np.zeros(shape=(env.observation_space.shape[0], env.observation_space.shape[0], env.frame_stack))
        self.episode = 0
        self.total_step = 0
        self.episode_reward_sum = 0

        self.q_vals = np.zeros(shape=env.action_space.n)
        self.q_plot = deque(maxlen=200)
        self.values = []

        self.action_titles = []

        for i, action in enumerate(env.env.unwrapped.get_action_meanings()):
            self.action_titles.append(
                pyglet.text.Label(action, font_size=10, color=self.font_color_dark, font_name=self.font,
                                  x=0, y=0, anchor_x='center'))

    def show_max_q(self):
        dpi_res = min(self.width, self.height) / 10
        fig = Figure((500 / dpi_res, 230 / dpi_res), dpi=dpi_res)
        ax = fig.add_subplot(111)

        ax.set_title('Max Action Q', fontsize=15)
        ax.set_xticklabels([])
        ax.set_ylabel('V(s)')
        self.q_plot.append(np.amax(self.q_vals))
        ax.plot(self.q_plot)

        w, h = fig.get_size_inches()
        dpi_res = fig.get_dpi()
        w, h = int(np.ceil(w * dpi_res)), int(np.ceil(h * dpi_res))
        canvas = FigureCanvasAgg(fig)
        pic_data = io.BytesIO()
        canvas.print_raw(pic_data, dpi=dpi_res)
        img = pyglet.image.ImageData(w, h, 'RGBA', pic_data.getvalue(), -4 * w)
        img.blit(20, 275)

    def generate_heatmap(self, last_conv):
        with tf.GradientTape() as tape:
            last_conv_layer = self.policy.model.get_layer(last_conv)

            iterate = tf.keras.models.Model([self.policy.model.inputs],
                                            [self.policy.model.output, last_conv_layer.output])

            model_out, last_conv_layer = iterate(self.state[np.newaxis, :, :, :])
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((7, 7))
        heatmap = cv2.resize(heatmap, (self.state.shape[1], self.state.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) / 255

        return heatmap

    def show_action(self):
        length = 360.0 / self.q_vals.shape[0]
        starting_x = 900
        for i, q_val in enumerate(self.q_vals):
            x_value = starting_x + i * (length + 10)
            color = np.repeat((150 ** (q_val * 2)) / (sum([150 ** (q * 2) for q in self.q_vals]) + 0.0001), 12).tolist()
            color = tuple(color)

            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                                 ('v2f', (
                                     x_value, self.height - 50, x_value + length, self.height - 50,
                                     x_value + length,
                                     self.height - length - 50, x_value, self.height - length - 50)),
                                 ('c3f', color))

            # Draw action label
            pyglet.gl.glTranslatef(x_value + length / 2, self.height - 100 - length, 0.0)
            pyglet.gl.glRotatef(-90.0, 0.0, 0.0, 1.0)
            self.action_titles[i].draw()
            pyglet.gl.glRotatef(90.0, 0.0, 0.0, 1.0)
            pyglet.gl.glTranslatef(-(x_value + length / 2), -(self.height - 100 - length), 0.0)

    def show_attention(self):
        heatmap_title = pyglet.text.Label('Agent Attention',
                                          font_size=15, color=self.font_color_dark, font_name=self.font,
                                          x=1050, y=self.height - 240, anchor_y='center')
        heatmap_title.draw()
        base_dimensions = (84, 84)
        intensity = 0.1
        scale = 5

        processed_frame = np.repeat(self.state[:, :, 3, np.newaxis], 3, axis=2)
        last_conv = self.policy.get_last_conv2d_name()
        heatmap = self.generate_heatmap(last_conv)

        img = (heatmap * 255 * intensity + processed_frame * 0.8).astype(np.uint8)

        display_nparray(cv2.resize(img + (heatmap * 255 * intensity).astype(np.uint8),
                                   dsize=(int(base_dimensions[1] * scale), int(base_dimensions[0] * scale)),
                                   interpolation=cv2.INTER_CUBIC)).blit(910, 10)

    def show_render(self):
        base_dimensions = (self.render_img.shape[0], self.render_img.shape[1])
        scale = 2
        display_nparray(cv2.resize(
            self.render_img,
            dsize=(int(base_dimensions[1] * scale), int(base_dimensions[0] * scale)),
            interpolation=cv2.INTER_CUBIC)) \
            .blit(550, self.height - base_dimensions[0] * scale - 30)

    def show_agent_vision(self):
        agent_title = pyglet.text.Label('Agent Vision',
                                        font_size=15, color=self.font_color_dark, font_name=self.font,
                                        x=20, y=245, anchor_y='center')
        agent_title.draw()
        base_dimensions = (84, 84)
        scale = 2.5
        state_images = [np.repeat(self.state[:, :, i, np.newaxis], 3, axis=2) for i in
                        range(self.state.shape[-1])]
        for i, state_image in enumerate(state_images):
            display_nparray(cv2.resize(state_image,
                                       dsize=(int(base_dimensions[1] * scale), int(base_dimensions[0] * scale)),
                                       interpolation=cv2.INTER_CUBIC)) \
                .blit(10 + i * (84 * scale + 5), 10)

    def show_title(self):
        env_title = pyglet.text.Label(f'Environment: {self.env.id}',
                                      font_size=18, color=self.font_color_light, font_name=self.font,
                                      x=20, y=self.height - 50, anchor_y='center')

        n_action_title = pyglet.text.Label(f'Number of action: {self.env.action_space.n}',
                                           font_size=14, color=self.font_color_light, font_name=self.font,
                                           x=20, y=self.height - 90, anchor_y='center')

        agent_title = pyglet.text.Label(f'Agent: {self.id}',
                                        font_size=18, color=self.font_color_light, font_name=self.font,
                                        x=20, y=self.height - 130, anchor_y='center')

        episode_title = pyglet.text.Label(f'Episode: {self.episode + 1}\t'
                                          f'Reward: {self.episode_reward_sum}',
                                          font_size=14, color=self.font_color_light, font_name=self.font,
                                          x=20, y=self.height - 170, anchor_y='center')

        env_title.draw()
        n_action_title.draw()
        agent_title.draw()
        episode_title.draw()

    def on_draw(self):
        self.clear()
        pyglet.gl.glClearColor(0. / 255., 40. / 255., 68 / 255., 1)
        pyglet.gl.glTexParameteri(pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MAG_FILTER, pyglet.gl.GL_NEAREST)

        self.switch_to()
        self.dispatch_events()

        self.show_title()
        self.show_render()
        if self.total_step > 10:
            self.show_max_q()
        self.show_action()
        self.show_agent_vision()
        self.show_attention()

    def update(self, dt):
        if self.done:
            self.state = self.env.reset()
            self.episode_reward_sum = 0
            self.done = False

        state = tf.expand_dims(self.state, axis=0)
        q_vals = self.policy.predict(tf.constant(state)).numpy()[0]
        action = np.argmax(q_vals)

        state, reward, done, _ = self.env.step(action)
        self.render_img = self.env.render(mode='rgb_array')
        self.q_vals = q_vals
        self.state = state
        self.done = done
        self.values.append(q_vals[action])
        self.episode_reward_sum += reward
        self.total_step += 1

        if self.done:
            self.episode += 1
            self.values = []
            if self.episode >= self.max_episode:
                self.close()
