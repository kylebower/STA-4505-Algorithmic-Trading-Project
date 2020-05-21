import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Get a data set of optimal actions for pairs (inventory,price)
# At a fixed time
# Generate sample data
data = np.random.rand(10, 10)
print(data)
print(-data)
# Columns are x axis (inventory) and rows are y axis (price)
# I'm using 'coolwarm' here which plots low values in blue and
# large values in red so for our purposes we want to switch
# this and we plot -data. (There's probably a better colourmap we can find)

# heat_map = sb.heatmap(-data, cmap='coolwarm')
heat_map = sb.palplot(sb.diverging_palette(20, 220, n=10))
plt.xlabel("Inventory")
plt.ylabel("Price")
plt.show()
