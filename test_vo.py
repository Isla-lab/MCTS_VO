import math

import numpy as np
from matplotlib import pyplot as plt

from bettergym.environments.robot_arena import Config, RobotArenaState


def compute_int(r0, r1, d, x0, x1, y0, y1):
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(r0 ** 2 - a ** 2)
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d
    x3 = x2 + h * (y1 - y0) / d
    y3 = y2 - h * (x1 - x0) / d

    x4 = x2 - h * (y1 - y0) / d
    y4 = y2 + h * (x1 - x0) / d

    return (x3, y3), (x4, y4)


def get_intersections(p0, p1, r0, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    x0, y0 = p0
    x1, y1 = p1

    d = math.hypot(p0[0] - p1[0], p0[1] - p1[1])

    # non-intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        compute_int(r0, r1, d, x0, x1, y0, y1)


def plot_circles(i, center1, radius1, center2, radius2, points):
    # Create a figure and an axes object
    fig, ax = plt.subplots()
    # Set the aspect ratio to be equal
    ax.set_aspect("equal")
    # Plot the first circle using the center and radius
    circle1 = plt.Circle(center1, radius1, color="blue", fill=False)
    ax.add_patch(circle1)
    # Plot the second circle using the center and radius
    circle2 = plt.Circle(center2, radius2, color="red", fill=False)
    ax.add_patch(circle2)

    # Plot the points of intersection if they are not None
    if points is not None:
        # Unpack the list of points into x and y coordinates
        x, y = zip(*points)
        # Plot the points using a scatter plot
        ax.scatter(x, y, color="green", marker="x")

    # Set the limits of the axes
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 11.5)
    ax.grid()

    # Add labels to the circles using text annotations
    ax.text(center1[0], center1[1], "R", color="blue", fontsize=14)
    ax.text(center2[0], center2[1], "O", color="red", fontsize=14)
    ax.legend([circle1, circle2], ["vR*dt+vO*dt", "Robot + Obs rad"], loc="upper right")

    # Add the values of r1 and r2 on the right of the plot using text annotations
    ax.text(12, 10, f"r1 = {radius1}", color="blue", fontsize=12)
    ax.text(12, 9, f"r2 = {radius2}", color="red", fontsize=12)

    # Show the plot
    plt.savefig(f'{i}.png', dpi=300)


def velocity_obstacle(x, obs, dt, ROBOT_RADIUS, OBS_RADIUS):
    VMAX = 0.3
    # 0 is the velocity of the obstacle, if its moving then change
    r0 = VMAX * dt + obs[:, 3] * dt
    r1 = ROBOT_RADIUS + OBS_RADIUS
    for i in range(len(obs)):
        plot_circles(
            i=i,
            center1=x[:2],
            radius1=r0[i],
            center2=obs[i],
            radius2=r1[i],
            points=get_intersections(p0=x[:2], r0=r0[i], p1=obs[i][:2], r1=r1[i])
        )


def main():
    initial_pos = (1, 1)
    goal = (10, 10)
    c = Config()
    obstacles_positions = np.array([
        [3.0, 2.0],
        [1.0, 4.0],
        [4.0, 7.0],
        [9.0, 7.0],
    ])
    radiuses = [1, 1, 2, 2]
    obs = [RobotArenaState(np.pad(obstacles_positions[i], (0, 2), 'constant'), goal=None, obstacles=None,
                           radius=radiuses[i]) for i in
           range(len(obstacles_positions))]
    initial_state = RobotArenaState(
        x=np.array([initial_pos[0], initial_pos[1], math.pi / 8.0, 0.0]),
        goal=np.array([goal[0], goal[1]]),
        obstacles=obs,
        radius=c.robot_radius
    )
    c.num_discrete_actions = 1
    velocity_obstacle(
        x=initial_state.x,
        obs=np.array([ob.x for ob in obs]),
        dt=0.2,
        ROBOT_RADIUS=initial_state.radius,
        OBS_RADIUS=np.array(radiuses)
    )


if __name__ == '__main__':
    main()
