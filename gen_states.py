import os
import json
from PIL import Image
from typing import Set, Optional
import numpy as np
from state import State, Connection


def rotate90(connections, times=1):
    connections = list(connections)
    values = np.array([c.value for c in connections])
    transform_values = (values + 2 * times) % 8
    return {Connection(v, c.id) for v, c in zip(transform_values, connections)}


def xflip(connections):
    xflip_map = np.array([2, 1, 0, 7, 6, 5, 4, 3])
    connections = list(connections)
    values = np.array([c.value for c in connections])
    transform_values = xflip_map[values]
    return {Connection(v, c.id) for v, c in zip(transform_values, connections)}


def yflip(connections):
    yflip_map = np.array([6, 5, 4, 3, 2, 1, 0, 7])
    connections = list(connections)
    values = np.array([c.value for c in connections])
    transform_values = yflip_map[values]
    return {Connection(v, c.id) for v, c in zip(transform_values, connections)}


def state_rotate90(state: State, times=1):
    img = state.img
    connections = state.connections
    img_ = np.rot90(img, -times)
    connections_rot = rotate90(connections, times)
    return State(img_, connections_rot, state.weight)


def state_xflip(state: State):
    img = state.img
    connections = state.connections
    img_ = np.flip(img, axis=1)
    connections_flip = xflip(connections)
    return State(img_, connections_flip, state.weight)


def state_yflip(state: State):
    img = state.img
    connections = state.connections
    img_ = np.flip(img, axis=0)
    connections_flip = yflip(connections)
    return State(img_, connections_flip, state.weight)


def gen_states(
    state_config: os.PathLike, load_dir: Optional[os.PathLike] = None
) -> Set[State]:
    if load_dir is None:
        load_dir = os.path.dirname(state_config)

    state_config_path = state_config
    with open(state_config_path) as f:
        state_config = json.load(f)

    states = set()
    for state_dict in state_config:
        connections_set = set()
        path = os.path.join(load_dir, state_dict["name"] + ".png")
        img = np.array(Image.open(path))
        connections = {
            Connection(conn_dict["value"], conn_dict["id"])
            for conn_dict in state_dict["connections"]
        }
        state = State(img, connections, state_dict["weight"])
        states.add(state)
        connections_set.add(state.connections)
        if state_dict["rotate"]:
            for i in range(1, 4):
                state_rot = state_rotate90(state, i)
                if state_rot.connections not in connections_set:
                    states.add(state_rot)
                    connections_set.add(state_rot.connections)
                if state_dict["flip_x"]:
                    state_flip = state_xflip(state_rot)
                    if state_flip not in connections_set:
                        states.add(state_flip)
                        connections_set.add(state_flip.connections)
                if state_dict["flip_y"]:
                    state_flip = state_yflip(state_rot)
                    if state_flip not in connections_set:
                        states.add(state_flip)
                        connections_set.add(state_flip.connections)

        if state_dict["flip_x"]:
            state_flip = state_xflip(state)
            if state_flip not in connections_set:
                states.add(state_flip)
                connections_set.add(state_flip.connections)
        if state_dict["flip_y"]:
            state_flip = state_yflip(state)
            if state_flip not in connections_set:
                states.add(state_flip)
                connections_set.add(state_flip.connections)
    return states
