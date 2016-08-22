import numpy as np
import matplotlib.pyplot as plt
from plt_save import save

# Make a quick sin plot
x = np.linspace(10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Save it in png and svg formats
save("signal", ext="png", close=False, verbose=True)
save("signal", ext="svg", close=True, verbose=True)