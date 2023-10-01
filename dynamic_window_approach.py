"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
from enum import Enum
import time

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


def dwa_control(x, config, goal, ob, robot_ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob, robot_ob)

    return u, trajectory


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 0.3  # [m/s]
        self.min_speed = -0.1  # [m/s]
        self.max_yaw_rate = 1.9  # [rad/s]
        self.max_accel = 6  # [m/ss]
        self.max_delta_yaw_rate = 40 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.1  # [m/s]
        self.yaw_rate_resolution = self.max_yaw_rate / 11.0  # [rad/s]
        self.dt = 1.0  # [s] Time tick for motion prediction
        self.predict_time = 15.0 * self.dt  # [s]
        self.to_goal_cost_gain = 1.
        self.speed_cost_gain = 0.0
        self.obstacle_cost_gain = 100.
        self.robot_stuck_flag_cons = 0.0  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.3  # [m] for collision check
        self.ob_radius = 0.6
        self.xlim = 11.
        self.ylim = 11.

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        # self.ob = []
        self.ob = np.array([[4,4], [4,6], [5,5], [6,4], [6,6]])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()


def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob, robot_ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # v_list = np.arange(dw[0], dw[1], config.v_resolution)
    # if 0. not in v_list:
    #     v_list = np.append(v_list, 0.)
    # y_list = np.arange(dw[2], dw[3], config.yaw_rate_resolution)
    # if x_init[2] not in y_list:
    #     y_list = np.append(y_list, x_init[2])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal, config)
            if to_goal_cost == -float("Inf"):
                return best_u, best_trajectory
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config, robot_ob)

            # final_cost = to_goal_cost
            # print("to_goal_cost")
            # print(to_goal_cost)
            # print("ob_cost")
            # print(ob_cost)
            final_cost = np.nan_to_num(to_goal_cost + speed_cost + ob_cost)

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate

        # print(min_cost)
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config, robot_ob):
    """
    calc obstacle cost inf: collision
    """
    try:
        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r1 = np.hypot(dx, dy).min()
        theta1 = np.arctan2(dy, dx).min()
    except:
        theta1=100.
        r1=100.
    try:
        rob_ox = robot_ob[:, 0]
        rob_oy = robot_ob[:, 1]
        dx_rob = trajectory[:, 0] - rob_ox[:, None]
        dy_rob = trajectory[:, 1] - rob_oy[:, None]
        r2 = np.hypot(dx_rob, dy_rob).min()
        theta2 = np.arctan2(dy_rob, dx_rob).min()
    except:
        theta2=100.
        r2=100.

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_width / 2 
        right_check = local_ob[:, 1] <= config.robot_length / 2 
        bottom_check = local_ob[:, 0] >= -config.robot_width / 2 
        left_check = local_ob[:, 1] >= -config.robot_length / 2 
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r1 <= config.robot_radius + config.ob_radius).any() or np.array(r2 <= config.robot_radius + config.robot_radius).any() or np.array(trajectory[:, 0]+config.robot_radius > config.xlim).any() or np.array(trajectory[:, 1]+config.robot_radius > config.ylim).any():
            return 1.0
            # return float("Inf")

    # min_theta = min(theta1.min(), theta2)
    # cost_theta = math.pi / abs(min_theta)

    # min_r1 = r1.min()
    # min_r2 = r2
    # cost_dist = 0.
    # if min_r1 <= min_r2:
    #     min_r1_arg = [np.where(r1 == min_r1)[0][0], np.where(r1 == min_r1)[1][0]]
    #     cost_dist = np.linalg.norm([dx[min_r1_arg[0], min_r1_arg[1]], dy[min_r1_arg[0], min_r1_arg[1]]]) / r1[min_r1_arg[0], min_r1_arg[1]]
    # else:
    #     min_r2_arg = [np.where(r2 == min_r2)[0][0], np.where(r2 == min_r2)[1][0]]
    #     cost_dist = np.linalg.norm([dx[min_r2_arg[0], min_r2_arg[1]], dy[min_r2_arg[0], min_r2_arg[1]]]) / r2[min_r2_arg[0], min_r2_arg[1]]

    # return np.sqrt(cost_dist**2 + cost_theta**2)
    return 0.0  # OK
    # return 1.0 / min(min_r1, min_r2)  # OK


def calc_to_goal_cost(trajectory, goal, config):
    """
        calc to goal cost with angle difference and pos difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    # if np.linalg.norm([dx, dy]) <= config.robot_radius:
    #     return -float("Inf")
    error_angle = math.atan2(dy, dx)
    cost_angle = abs(error_angle - trajectory[-1, 2]) / math.pi
    # cost_angle = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    cost_dist = np.linalg.norm([dx, dy])# / np.linalg.norm([goal[0]-trajectory[0, 0], goal[1]-trajectory[0, 1]])

    if cost_dist < config.robot_radius:
        return -10.

    # return np.linalg.norm([cost_dist, cost_angle])
    return cost_dist


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")
    
    #plot obstacles
    for ob in config.ob:
        circle = plt.Circle(ob, config.ob_radius, color="k")
        plt.gcf().gca().add_artist(circle)
        # out_x, out_y = (np.array([x, y]) +
        #                 np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        # plt.plot([x, out_x], [y, out_y], "-k")


def main(robot_type=RobotType.circle):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    # xs = [np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])]
    xs = [np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0]), np.array([10.0, 10.0, -math.pi * 7.0 / 8.0, 0.0, 0.0]), np.array([10.0, 0.0, 5.0 * math.pi / 8.0, 0.0, 0.0]), np.array([0.0, 10.0, -math.pi * 3.0 / 8.0, 0.0, 0.0])]
    # goal position [x(m), y(m)]
    # goals = [np.array([10., 10.])]
    goals = [np.array([10.0, 10.0]), np.array([0.0, 0.0]), np.array([0.0, 10.0]), np.array([10.0, 0.0])]

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type
    trajectories = [xs[i] for i in range(len(xs))]
    ob = config.ob
    robot_ob = np.array([])
    finished = np.ones(len(trajectories))*False
    cost = 0.
    count = 0.
    times = []
    while not np.prod(finished):
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            # plt.gcf().canvas.mpl_connect(
            #     'key_release_event',
            #     lambda event: [exit(0) if event.key == 'escape' else None])
        for r in range(len(trajectories)):
            for r1 in range(len(trajectories)):
                if r1 != r:
                    if np.shape(robot_ob)[0] == 0:
                        robot_ob = np.array([xs[r1][0], xs[r1][1]])
                    else:
                        robot_ob = np.vstack((robot_ob, np.array([xs[r1][0], xs[r1][1]])))
            start = time.time()
            u, predicted_trajectory = dwa_control(xs[r], config, goals[r], ob, robot_ob)
            end = time.time()
            times.append(end-start)
            robot_ob = np.array([])
            xs[r] = motion(xs[r], u, config.dt)  # simulate robot
            trajectories[r] = np.vstack((trajectories[r], xs[r]))  # store state history
            if not finished[r]:
                cost -= np.linalg.norm(xs[r][:2] - goals[r]) / (config.xlim * np.sqrt(2))
                count += 1.

            if show_animation:
                # plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
                plt.plot(xs[r][0], xs[r][1], "xr")
                plt.plot(goals[r][0], goals[r][1], "xb")
                # if ob is not None:
                #     plt.plot(ob[:, 0], ob[:, 1], "ok")
                plot_robot(xs[r][0], xs[r][1], xs[r][2], config)
                # plot_arrow(xs[r][0], xs[r][1], xs[r][2])
                plt.axis("equal")
                plt.grid(True)
            
        if show_animation:
            plt.pause(0.0001)

        for r in range(len(trajectories)):
            # check reaching goal
            dist_to_goal = math.hypot(xs[r][0] - goals[r][0], xs[r][1] - goals[r][1])
            if dist_to_goal <= config.robot_radius and not finished[r]:
                # print("Goal!!")
                finished[r] = True
                cost += 10.

    print("Done")
    print("Average time: " + str(np.mean(times)) + "; std time: " + str(np.std(times)))
    print(cost / len(goals[r]))
    if show_animation:
        for r in range(len(trajectories)):
            plt.plot(trajectories[r][:, 0], trajectories[r][:, 1], "-r")
            plt.pause(0.0001)
    plt.show()


if __name__ == '__main__':
    # main(robot_type=RobotType.rectangle)
    main(robot_type=RobotType.circle)
