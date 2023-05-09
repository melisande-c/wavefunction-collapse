from typing import Set, Optional, Union
import numpy as np
from state import State, Connection


class Tile:

    """Contains all possible states for a given tile."""

    def __init__(self, states: Set[State]):
        """
        Parameters
        ----------
        states: Set[State]
            Set of possible states.
        """

        self.states = states
        self.possible_states: Set[State] = set()
        self.active_connections: Set[Connection] = set()
        self.possible_connections: Set[Connection] = set()

        for state in self.states:
            self.possible_connections = self.possible_connections.union(
                state.connections
            )
        self.update_possible_states()

        self.collapsed = False

    @property
    def entropy(self):
        if self.collapsed:
            return 0
        if len(self.active_connections) == 0:
            return np.inf
        weights = np.array(
            [
                state.weight
                for state in self.possible_states
                if self.active_connections.intersection(state.connections) != 0
            ]
        )
        denominator = np.sum(
            np.array([1 / state.weight for state in self.states])
        )
        return np.sum(1 / weights) / denominator
        # denominator = np.sum(np.array([state.weight for state in self.states]))
        # return np.sum(weights) / denominator

    def collapse(self, state: Optional[State] = None) -> None:
        """
        Collapses the tile to a single possible state.

        Parameters
        ----------
        state: optional[State]
            Can be a chosen state, if none the state will be selected at random
            from the possible states.
        """
        self.collapsed = True
        if state is not None:
            selected = state
        elif (state is None) and (len(self.possible_connections) == 0):
            return
        elif (state is None) and (len(self.possible_states) != 0):
            possible_states = list(self.possible_states)
            weights = np.array([s.weight for s in possible_states])
            probabilities = weights / weights.sum()
            selected = np.random.choice(possible_states, p=probabilities)

        self.possible_connections = set(selected.connections)
        self.active_connections = set(selected.connections)
        self.possible_states = {selected}

    def add_connection(self, connection: Connection) -> None:
        if self.collapsed:
            return

        # remove all connections for that direction
        cval = connection.value
        connections = [c for c in self.possible_connections if c.value == cval]
        for c in connections:
            self.possible_connections.discard(c)

        # add connection
        self.active_connections.add(connection)
        self.possible_connections.add(connection)

        # update
        self.update_possible_states()

    def update_possible_states(self) -> None:
        """
        Updates the possible states
        """
        self.possible_states = {
            s for s in self.states if self.is_state_possible(s)
        }
        self.possible_states.union(
            {s for s in self.possible_states if self.is_state_possible(s)}
        )
        if len(self.possible_states) == 0:
            self.collapsed = True
        else:
            self.collapsed = False

    def is_state_possible(self, state: State) -> bool:
        """Returns if the given state is possible"""
        outward_conn_compatible = self.active_connections.issubset(
            state.connections
        )
        possible_conn_compatible = state.connections.issubset(
            self.possible_connections
        )
        return outward_conn_compatible and possible_conn_compatible

    def get_connection(self, connection_direction: int) -> Union[str, None]:
        connections = [
            c
            for c in self.active_connections
            if c.value == connection_direction
        ]
        if len(connections) == 0:
            return None

        return connections[0]
