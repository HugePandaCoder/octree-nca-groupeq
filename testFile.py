import numpy as np


#a = np.array([[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1]])

#a[1:-1, 1:-1] = 2

#print(a)

size = 10
steps = 10

a = np.zeros((size, size))
a[int(size/2),int(size/2)] = 1

communication_Chance = 0.5

b = a.copy()

x = 5
y = 5

print(a[x-1:x+2, y-1:y+2])

for s in range(steps):
    for x in range(size):
        for y in range(size):
            if np.random.rand(1)[0] > communication_Chance:
                if np.sum(a[x-1:x+2, y-1:y+2]) > 0:
                    b[x, y] = 1
    a = b.copy()

print(a)