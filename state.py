from collections import namedtuple
from numpy.typing import NDArray
from typing import Set

# TODO explain connection

Connection = namedtuple("Connection", "value id")


class State:
    """Holds the information of a state."""

    def __init__(
        self, img: NDArray, connections: Set[Connection], weight: float
    ):
        """
        Parameters
        ----------
        img: PIL.Image.Image
            One tile image.
        connections: Set[Connection]
            The set of connections
        weight:
            A weighting for the pobability of selection.
        """

        self.img = img
        self.connections = frozenset(connections)
        self.weight = weight
