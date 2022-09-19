import string
import numpy as np
import matplotlib.pyplot as plt

result = np.array([
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]])


count = 1
closest = 25

world = np.random.randint(low=0, high=2, size=(7, 7))
out = world[1:-1, 1:-1]

def plot(world):
    world = world.astype(np.int)

    #print_world[world == True] = "1"#"■"
    #print_world[world == False] = "0"#"▢"

    imgplot = plt.imshow(world)
    plt.show()

    print(world)

while not np.array_equal(out, result):
    world = np.random.randint(low=0, high=2, size=(7, 7))
    out = world[1:-1, 1:-1]

    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            sum = np.sum(world[x:x+3][y:y+3])
            # Currently alive
            if out[x, y] == 1:
                if sum < 2 or sum > 3:
                    out[x, y] = 0
                else:
                    out[x, y] = 1
            else:
                if sum == 3:
                    out[x, y] = 1
    print("Tries: "  + str(count) + ", Closest: " + str(closest), end="\r")
    count = count +1

    #print("____________________")
    #print(result.astype(np.int))
    #print(out.astype(np.int))
    #print(np.abs(np.subtract(result.astype(np.int), out.astype(np.int))))
    closeness = np.sum(result != out)
    if closeness < closest:
        closest = closeness
        plot(world)
    #print(closest)

plot(world)


