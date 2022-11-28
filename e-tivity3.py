import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import random
import math
import numpy as np

NUMBER_OF_NEEDLES = 10


class DefineNeedle:
    def __init__(self, x=None, y=None, theta=None, length=0.5):
        if x is None:
            x = random.uniform(0, 1)
        if y is None:
            y = random.uniform(0, 1)
        if theta is None:
            theta = random.uniform(0, math.pi)

        self.needle_coordinates = np.array([x, y])
        self.complex_representation = np.array(
            [length/2 * math.cos(theta), length/2*math.sin(theta)])
        self.end_points = np.array([np.add(self.needle_coordinates, -1*np.array(
            self.complex_representation)), np.add(self.needle_coordinates, self.complex_representation)])

    def intersects_with_y(self, y):
        return self.end_points[0][1] < y and self.end_points[1][1] > y


class BuffonSimulation:
    def __init__(self):
        self.floor = []
        self.boards = 2
        self.list_of_needle_objects = []
        self.number_of_intersections = 0

        fig = plt.figure(figsize=(10, 10))
        self.buffon = plt.subplot()
        self.results_text = fig.text(
            0, 0, self.estimate_pi(), size=15)
        self.buffon.set_xlim(-0.1, 1.1)
        self.buffon.set_ylim(-0.1, 1.1)

    def plot_floor_boards(self):
        for j in range(self.boards):
            self.floor.append(0+j)
            self.buffon.hlines(
                y=self.floor[j], xmin=0, xmax=1, color='black', linestyle='--', linewidth=2.0)

    def toss_needles(self):
        needle_object = DefineNeedle()
        self.list_of_needle_objects.append(needle_object)
        x_coordinates = [needle_object.end_points[0]
                         [0], needle_object.end_points[1][0]]
        y_coordinates = [needle_object.end_points[0]
                         [1], needle_object.end_points[1][1]]

        for board in range(self.boards):
            if needle_object.intersects_with_y(self.floor[board]):
                self.number_of_intersections += 1
                self.buffon.plot(x_coordinates, y_coordinates,
                                 color='green', linewidth=1)
                return
        self.buffon.plot(x_coordinates, y_coordinates,
                         color='red', linewidth=1)

    def estimate_pi(self, needles_tossed=0):
        if self.number_of_intersections == 0:
            estimated_pi = 0
        else:
            estimated_pi = (needles_tossed) / \
                (1 * self.number_of_intersections)
        error = abs(((math.pi - estimated_pi)/math.pi)*100)
       
        return (" Intersezioni:" + str(self.number_of_intersections) +
                "\n Aghi totali: " + str(needles_tossed) +
                "\n Approsimazione pi: " + str(estimated_pi) +
                "\n Numero casi favorevoli: " + str(self.number_of_intersections) +
                "\n Numero casi non favorevoli " + str(needles_tossed - self.number_of_intersections ) +
                #Approssimazione errore  
                "\n Errore: " + str(round(error,2)) + "%")

    def plot_needles(self):
        for needle in range(NUMBER_OF_NEEDLES):
            self.toss_needles()
            self.results_text.set_text(self.estimate_pi(needle+1))
            if (needle+1) % 200 == 0:
                plt.pause(1/200)
        plt.title("Probabilità lancio di aghi")


    
    #In questo caso simulo il lancio di 10 aghi e trovo il plot
    #dei numeri di lanci Vs Stima PI
    def plot_point(self,needles_tossed=10):
        needle_object = DefineNeedle()
        self.list_of_needle_objects.append(needle_object)
        for board in range(self.boards):
            if needle_object.intersects_with_y(self.floor[board]):
                self.number_of_intersections += 1
        estimated_pi = (needles_tossed) / \
                (1 * self.number_of_intersections)
        data = np.array([
            [needles_tossed, estimated_pi],
        ])
        x, y = data.T
        plt.xlabel('Lancio aghi'),
        plt.ylabel('Stima di Pi')
        plt.title("Lanci totali VS stima Pi")
        plt.scatter(x,y)


    def plot2(self):
        self.plot_point()
        plt.show()
    def plot(self):
        self.plot_floor_boards()
        self.plot_needles()
        plt.show()


simulation = BuffonSimulation()
simulation.plot()
simulation.plot2()
