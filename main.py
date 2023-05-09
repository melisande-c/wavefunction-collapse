from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from gen_states import gen_states
from grid import Grid

state_config = "/Users/milly/Documents/Personal/Projects/wavefunction-collapse/tile_sets/orangewhite/states.json"
background_path = "/Users/milly/Documents/Personal/Projects/wavefunction-collapse/tile_sets/orangewhite/background.png"

background_img = np.array(Image.open(background_path))
tile_size = background_img.shape[:2]

states = gen_states(state_config)

grid = Grid(16, 32, states, tile_size, background_img)

fig, ax = plt.subplots()
while grid.logic():
    pass

img = grid.display()

ax.imshow(img)
plt.show()
