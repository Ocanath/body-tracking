from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import matplotlib.animation as animation
import numpy as np
import time

figure = pyplot.figure()
axes=mplot3d.Axes3D(figure)


# Load the STL files and add the vectors to the plot
mesh = mesh.Mesh.from_file('test.STL')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))


# Auto scale to the mesh size
scale = mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

tstart = time.time()
datas = 0
def update_anim(datas):
	t = time.time()-tstart
	mesh.rotate([0,0,1],np.sin(t))
	
	axes.clear()
	axes.auto_scale_xyz(scale, scale, scale)
	axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))

line_ani = animation.FuncAnimation(figure, update_anim, 25, fargs=(datas),
                                   interval=50, blit=False)

# Show the plot to the screen
pyplot.show()