from matplotlib.animation import FuncAnimation  # test
import matplotlib.pyplot as plt
y = []
x = []
fig = plt.figure()
ln, = plt.plot([], [], 'r')


def update(frame):
    y.append(frame**2)
    x.append(frame)
    plt.xlim([-1, x[-1]+1])
    plt.ylim([-1, y[-1]+1])
    ln.set_data(x, y)
    return ln,


ani = FuncAnimation(fig, update, frames=1000, interval=200)
plt.show()
