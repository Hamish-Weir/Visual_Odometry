import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read coordinates from the file
coordinates = []
with open('data/path.txt', 'r') as f:
    for line in f:
        # Convert each line to a tuple of floats (X, Y, Z)
        coordinates.append(tuple(map(float, line.split())))

# Unzip the coordinates into separate X, Y, Z lists
x, y, z = zip(*coordinates)

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the path as a line connecting the coordinates
ax.plot(x, y, z, label='Path')

# Add labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Optional: Customize the view (rotate the plot)
ax.view_init(elev=20, azim=30)  # You can change the angles to your preference

# Show the plot
plt.show()
