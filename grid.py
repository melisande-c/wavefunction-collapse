import copy
from typing import Set, Optional
from numpy.typing import NDArray
import numpy as np
from random import randint
from tile import Tile
from state import Connection, State


class Grid:
    def __init__(
        self,
        n: int,
        m: int,
        states: Set[State],
        tile_size: int = (64, 64),
        background_img: Optional[NDArray] = None,
    ):
        """
        Parameters
        ----------
        n: int
            Number of rows of tiles.
        m: int
            Number of columns of tiles.
        states: Set[State]
            Set of all possible states.
        """
        self.n = n
        self.m = m
        self.states = states
        self.tile_size = tile_size
        self.background_img = background_img

        self.grid = np.array(
            [[Tile(states) for _ in range(m)] for _ in range(n)]
        )
        self.entropy = np.full_like(self.grid, fill_value=np.inf, dtype=float)
        self.collapsed = np.zeros_like(self.entropy, dtype=bool)

    def collapse(self, i: int, j: int, state: Optional[State] = None):
        tile: Tile = self.grid[i, j]
        tile.collapse(state)
        self.collapsed[i, j] = tile.collapsed
        self.entropy[i, j] = tile.entropy
        self._update_neighbours(i, j)

    def _update_neighbours(self, i: int, j: int):
        tile: Tile = self.grid[i, j]

        for direction in range(8):
            if direction == 0:
                connections = [2, 4, 6]
                indices = [(i, j - 1), (i - 1, j - 1), (i - 1, j)]
            elif direction == 1:
                connections = [5]
                indices = [(i - 1, j)]
            elif direction == 2:
                connections = [4, 6, 0]
                indices = [(i - 1, j), (i - 1, j + 1), (i, j + 1)]
            elif direction == 3:
                connections = [7]
                indices = [(i, j + 1)]
            elif direction == 4:
                connections = [6, 0, 2]
                indices = [(i, j + 1), (i + 1, j + 1), (i + 1, j)]
            elif direction == 5:
                connections = [1]
                indices = [(i + 1, j)]
            elif direction == 6:
                connections = [0, 2, 4]
                indices = [(i + 1, j), (i + 1, j - 1), (i, j - 1)]
            elif direction == 7:
                connections = [3]
                indices = [(i, j - 1)]

            for connection_direction, (u, v) in zip(connections, indices):
                if (u < 0) or (self.n <= u):
                    continue
                if (v < 0) or (self.m <= v):
                    continue

                connection = tile.get_connection(direction)
                if connection is not None:
                    connection_id = connection.id
                    self.grid[u, v].add_connection(
                        Connection(connection_direction, connection_id)
                    )

                self.entropy[u, v] = self.grid[u, v].entropy
                self.collapsed[u, v] = self.grid[u, v].collapsed

    def minimum_entropy(self):
        if self.collapsed.all():
            return None

        entropy_selection = np.min(self.entropy[~self.collapsed])
        i, j = np.where(self.entropy == entropy_selection)
        idx = randint(0, len(i) - 1)
        return i[idx], j[idx]

    def logic(self):
        indices = self.minimum_entropy()

        if indices is None:
            return False
        i, j = indices

        self.collapse(i, j)

        return True

    def display(self):
        if self.background_img is None:
            img = np.zeros(
                (self.n * self.tile_size[0], self.m * self.tile_size[1], 4),
                dtype=np.uint8,
            )
            img[:, :, -1] = 255
        else:
            img = np.tile(self.background_img, (self.n, self.m, 1))

        for i in range(self.n):
            for j in range(self.m):
                tile: Tile = self.grid[i][j]
                if len(tile.possible_states) == 1:
                    state: State = list(tile.possible_states)[0]
                    img[
                        i * self.tile_size[0] : (i + 1) * self.tile_size[0],
                        j * self.tile_size[1] : (j + 1) * self.tile_size[1],
                    ] = state.img
        return img
