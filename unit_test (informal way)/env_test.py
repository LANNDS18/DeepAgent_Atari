from DeepRL.utils.game import GameEnv

game = GameEnv('DemonAttack-v0', frame_stack=4)
a = game.reset()
b = game.step(1)
print(a)