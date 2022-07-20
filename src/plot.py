import math

import matplotlib.pyplot as plt
import numpy as np

class Curve:
    """class representing an element which can be drawn in a graph"""

    MAX_POINTS = 100 # max number of points which will be drawn in a plot

    def __init__(self, x_values :np.ndarray, y_values :np.ndarray, linestyle = 'b', description = ""):
        split = 1 if x_values.size < Curve.MAX_POINTS else x_values.size // Curve.MAX_POINTS

        self.x_values = x_values[~np.isnan(x_values)] [::split]
        self.y_values = y_values[~np.isnan(y_values)] [::split]
        self.linestyle = linestyle
        self.description = description


class Custom_Plotter:

    def __init__(self, main_title = None):
        self.main_title = main_title
        self.figures = {
            "regret" : [],
            "pulls" : [],
            "reward_functions" : []
        }
        plt.rc('figure', max_open_warning = 0)
        

    def plot_graph(self, category, curves, x_name = None, y_name = None, title = None):
        self.figures[category].append(plt.figure())
        for curve in curves:
            ymin, ymax = self.__find_y_lims(curves)
            plt.plot(curve.x_values, curve.y_values, curve.linestyle)
            plt.title(title, y=0.95)
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.ylim(ymin, ymax)
            plt.grid(True)
            plt.tick_params('x')
        col = max(len(curves) // 5, 1)
        plt.legend(list(map(lambda c : c.description, curves)), ncol=col)


    def display(self):
        plt.show()


    def reset(self):
        for figures in list(map(lambda key : self.figures[key], self.figures)):
            for figure in figures:
                plt.close("all")
        self.figures = {
            "regret" : [],
            "pulls" : [],
            "reward_functions" : []
        }


    def save_regret(self, directory="", name=""):
        for f in self.figures["regret"]:
            f.savefig(f"{directory}/{name}.png", dpi=240)


    def save_reward_functions(self, directory=""):
        for f in self.figures["reward_functions"]:
            f.savefig(f"{directory}/reward_functions.png", dpi=240)


    def save_pulls(self, directory="", name=""):
        for i, fig in enumerate(self.figures["pulls"]):
            # arms
            fig.savefig(f"{directory}/arm_{i}.png", dpi=240)


    def __find_y_lims(self, curves):
        ymin = 0
        ymax = 1
        for curve in curves:
            if curve.x_values.size > 0:
                ymin = min(ymin, min(map(lambda x: x if not math.isnan(x) else 1, curve.y_values)))
                ymax = max(ymax, max(map(lambda x: x if not math.isnan(x) else 0, curve.y_values)))

        return ymin - abs(ymin) / 2, (ymax) * 1.1
