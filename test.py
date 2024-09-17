import numpy as np
from plot import Plotter

plot = Plotter(True)

x = np.linspace(0, 2 * np.pi, 100)
plot.add_line("sin", x, np.sin(x))

for i in range(100):
    y = np.sin(x + i * 0.1)
    plot.update_line("sin", y=y)
