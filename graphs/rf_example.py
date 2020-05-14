import numpy as np
import matplotlib.pyplot as plot

x_1 = [0.5, 0.7, 1.1]
y_1 = [1, 1.1, 0.9]
x_2 = [0.9, 1.3, 1.2]
y_2 = [1.5, 1.4, 1.4]
x_3 = [1.5, 1.7, 2]
y_3 = [0.5, 1.3, 1.9]

tresholds = [0.6, 0.8, 1, 1.15, 1.25, 1.4, 1.6, 1.8]
ytreshold = [1.45, 1.2, 1.05, 0.95]
plot.scatter(x_1,y_1, c="red", label="Class 1")
plot.scatter(x_2,y_2, c="blue", label="Class 2")
plot.scatter(x_3,y_3, c="green", label="Class 3")
plot.xlabel("x")
plot.ylabel("y")
plot.axvline(x=1.4, linestyle="-")
for i in ytreshold:
    plot.axhline(y=i, xmax=0.6, linestyle="--", c="black")
plot.legend()
plot.grid()
plot.show();
