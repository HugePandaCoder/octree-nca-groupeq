import numpy as np

inshape = [64,64,48]
scale_factor = 4
kernel_size = [7,3]
levels =2

steps = [np.ceil((max(inshape)/((levels-1)*scale_factor))/((kernel_size[x-1] - 1)/2)) for x in range(1, levels+1)]

print(steps)