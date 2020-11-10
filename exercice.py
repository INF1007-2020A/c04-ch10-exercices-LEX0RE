#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import matplotlib.pyplot as plt


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return [(np.linalg.norm(c), np.arctan2(c[1], c[0])) for c in cartesian_coordinates]


def find_closest_index(values: np.ndarray, number: float) -> int:
    return np.abs(values - number).argmin()


def create_graph():
    x = np.linspace(-1, 1, 250)
    y = (x ** 2) * np.sin(1 / (x ** 2)) + x
    plt.scatter(x, y)
    plt.ylabel('y')
    plt.ylabel('x')
    plt.xlim((-1, 1))
    plt.title("y = x² * sin(1/x²) + x")
    plt.show()


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    # print(linear_values())
    # print(coordinate_conversion(np.array([(0, 0), (10, 10)])))
    # print(find_closest_index(np.array([0, 5, 10, 12, 8]), 10.5))
    create_graph()
    pass
