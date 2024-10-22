import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, interactive: bool):
        self.fig, self.ax = plt.subplots()
        self.lines = {}

        if interactive:
            plt.ion()

    def add_line(self, label: str, x=[], y=[], color=None):
        """Adds a new line. Leave `x` and `y` empty to initialize only."""
        if color is None:
            (line,) = self.ax.plot(x, y)
        else:
            (line,) = self.ax.plot(x, y, color=color)

        line.set_label(label)
        self.lines[label] = line

    def update_line(self, label: str, y, x=None):
        """Updates an existing line."""
        line = self.lines.get(label, None)
        if line is None:
            raise Exception("Line does not exist.")

        y = np.append(line.get_ydata(), y)

        if x is None:
            x = list(range(len(y)))
        else:
            x = np.append(line.get_xdata(), x)

        line.set_data(x, y)

        # realignment and scaling
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend()
        plt.draw()
        plt.pause(0.1)

    def display(self):
        plt.show()
