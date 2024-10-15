import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, interactive: bool):
        self.fig, self.ax = plt.subplots()
        self.lines = {}

        if interactive:
            plt.ion()

    def add_line(self, label: str, x=[], y=[], color="black"):
        """Adds a new line. Leave `x` and `y` empty to initialize only."""
        (line,) = self.ax.plot(x, y, color=color)
        self.lines[label] = line

    def update_line(self, label: str, x=None, y=None):
        """Updates an existing line."""
        line = self.lines.get(label, None)
        if line is None:
            raise Exception("Line does not exist.")

        x = line.get_xdata() if x is None else x
        y = line.get_ydata() if y is None else y

        line.set_data(x, y)

        # realignment and scaling
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

    def display(self):
        plt.show()
