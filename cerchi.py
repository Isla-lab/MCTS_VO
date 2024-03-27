# intersection circles
import math

import numpy as np
from matplotlib import pyplot as plt


# [1.32661683 1.556276  ]
class Cerchi:
    def __init__(self):
        self.x0, self.y0 = None, None
        self.r0 = 0.6
        self.r1 = 0.2 + 0.3
        # r1_2 = 0.2 + 0.3 + r0
        self.obstacles = None

    # self.gym_env.robot_motion(np.array([1.,  0.,  0.,  0.3]), [0.05, 0])

    def get_intersections(self, r0, r1, x0, y0, x1, y1):
        # circle 1: (x0, y0), radius r0
        # circle 2: (x1, y1), radius r1

        d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # non intersecting
        if d > r0 + r1:
            return None
        # One circle within other
        if d < abs(r0 - r1):
            print(np.inf)
            return None
        # coincident circles
        if d == 0 and r0 == r1:
            return None
        else:
            a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
            h = math.sqrt(r0 ** 2 - a ** 2)
            x2 = x0 + a * (x1 - x0) / d
            y2 = y0 + a * (y1 - y0) / d
            x3 = x2 + h * (y1 - y0) / d
            y3 = y2 - h * (x1 - x0) / d

            x4 = x2 - h * (y1 - y0) / d
            y4 = y2 + h * (x1 - x0) / d

            return x3, y3, x4, y4

    def plot_intesection(self, intersection, ax):
        x0 = self.x0
        y0 = self.y0
        r0 = self.r0
        i_x3, i_y3, i_x4, i_y4 = intersection
        plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

        # Plot lines of the intersection points
        vec_p1 = np.array([i_x3 - x0, i_y3 - y0])
        vec_p2 = np.array([i_x4 - x0, i_y4 - y0])
        # Compute the angles using arctan2
        angle_p1 = np.arctan2(vec_p1[1], vec_p1[0])
        angle_p2 = np.arctan2(vec_p2[1], vec_p2[0])
        end_y1 = y0 + r0 * math.sin(angle_p1)
        end_x1 = x0 + r0 * math.cos(angle_p1)
        end_y2 = y0 + r0 * math.sin(angle_p2)
        end_x2 = x0 + r0 * math.cos(angle_p2)

        radius_line_p1 = plt.Line2D([x0, end_x1], [y0, end_y1],
                                    color='g')  # Line from center to second intersection point
        radius_line_p2 = plt.Line2D([x0, end_x2], [y0, end_y2],
                                    color='g')  # Line from center to first intersection point
        ax.add_artist(radius_line_p1)
        ax.add_artist(radius_line_p2)

    def plot(self, rad1, name):
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        obstacles = self.obstacles
        x0 = self.x0
        y0 = self.y0
        r0 = self.r0

        intersections = []
        for i, o in enumerate(obstacles):
            x1, y1 = o[:2]
            intersec = self.get_intersections(x0=x0, y0=y0, r0=r0, x1=x1, y1=y1, r1=rad1)
            circle2 = plt.Circle((x1, y1), rad1, color='b', fill=False)
            ax.add_artist(circle2)
            if intersec is not None:
                print(intersec)
                circle2_1 = plt.Circle((x1, y1), 0.2, color='r', fill=False)
                ax.add_artist(circle2_1)
                plt.scatter(x1, y1, c='b', marker='x')
                self.plot_intesection(intersec, ax)
                intersections.append(i)
        circle1 = plt.Circle((x0, y0), r0, color='b', fill=False)
        circle1_1 = plt.Circle((x0, y0), 0.3, color='r', fill=False)
        ax.add_artist(circle1)
        ax.add_artist(circle1_1)
        # ax.add_artist(circle1_2)
        plt.scatter(x0, y0, c='g', marker='x')
        # plt.scatter(1.04975186, 0.4975186 , c='k', marker='x')
        # ax.annotate('P', (1.71832245, 1.58689256))
        #
        # plt.scatter(1.04029777, 0.40297769, c='k', marker='x')
        # ax.annotate('Q', ([1.67861359, 1.52865567]))

        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f'debug/circle_{name}.png', dpi=500)
        return intersections
