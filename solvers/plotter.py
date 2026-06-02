import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl


def plot_evolution(result):

    plt.figure()
    plt.plot(result.ts, result.xs, label="x")
    plt.plot(result.ts, result.ys, label="y")
    plt.plot(result.ts, result.zs, label="z")
    plt.legend(loc="best")
    plt.xlabel("time t")
    plt.ylabel("Bloch vector component")
    plt.show()


def plot_evolution_bloch(result):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.array(result.xs)
    y = np.array(result.ys)
    z = -np.array(result.zs)

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    t = np.linspace(0, 1, len(x))
    norm = mpl.colors.Normalize(0, 1)
    cmap = plt.cm.jet

    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(t)
    ax.add_collection3d(lc)

    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.12, linewidth=0.5)

    L = 1.2
    ax.quiver(0, 0, 0, L, 0, 0, color="black", arrow_length_ratio=0.08, linewidth=1.5)
    ax.quiver(0, 0, 0, 0, L, 0, color="black", arrow_length_ratio=0.08, linewidth=1.5)
    ax.quiver(0, 0, 0, 0, 0, L, color="black", arrow_length_ratio=0.08, linewidth=1.5)
    ax.text(L, 0, 0, "X", fontsize=12)
    ax.text(0, L, 0, "Y", fontsize=12)
    ax.text(0, 0, L, "Z", fontsize=12)

    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(t)
    plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="time")

    plt.show()