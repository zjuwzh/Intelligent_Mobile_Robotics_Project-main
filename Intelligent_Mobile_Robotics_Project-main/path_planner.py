"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""

from flight_environment import FlightEnvironment            
from scipy.spatial import KDTree
import numpy as np
import random
import math


class Node:
    def __init__(self, x, y, z, cost, parent):
        self.x = x
        self.y = y
        self.z = z
        self.cost = cost
        self.parent = parent


class RRT_star:
    def __init__(self, MAX_EPOCH=2000, N_SAMPLE=100, STEP_PER_SEARCH=1, DIST_OPT=4, ARRIVE_Radius=1):
        self.MAX_EPOCH = MAX_EPOCH
        self.N_SAMPLE = N_SAMPLE
        self.STEP_PER_SEARCH = STEP_PER_SEARCH
        self.DIST_OPT = DIST_OPT # optimization distance
        self.ARRIVE_Radius = ARRIVE_Radius # the goal: a circle
        self.avoid_obs = 0.02
        self.max_cost = 1000

    def plan(self, start_given, goal_given, env: FlightEnvironment):
        # Type conversion to Node
        start = Node(start_given[0], start_given[1], start_given[2], 0.0, -1)
        goal = Node(goal_given[0], goal_given[1], goal_given[2], 0.0, -1)
        roadmap = dict()
        roadmap[0] = start
        path = []
        
        for _ in range(self.MAX_EPOCH):
            # Search Tree(one epoch)
            for _ in range(self.N_SAMPLE): # sample 100 points at most
                # sample a point
                sample = self.sampling(env)

                # find the nearest point in roadmap to the sampled point
                dist = []
                for j in range(len(roadmap)):
                    dx = sample[0] - roadmap[j].x
                    dy = sample[1] - roadmap[j].y
                    dz = sample[2] - roadmap[j].z
                    d = math.hypot(dx, dy, dz)
                    if j < len(dist):
                        dist[j] = d
                    else:
                        dist.append(d)
                dist_min = min(dist)
                index = dist.index(dist_min)

                # calculate new point
                dx = sample[0] - roadmap[index].x
                dy = sample[1] - roadmap[index].y
                dz = sample[2] - roadmap[index].z
                new_x = roadmap[index].x + self.STEP_PER_SEARCH * dx / math.hypot(dx, dy, dz)
                new_y = roadmap[index].y + self.STEP_PER_SEARCH * dy / math.hypot(dx, dy, dz)
                new_z = roadmap[index].z + self.STEP_PER_SEARCH * dz / math.hypot(dx, dy, dz)
                new_point = np.array([new_x, new_y, new_z])

                if env.is_collide(new_point) is True:
                    continue
                else:
                    pass

                # optimize roadmap
                cost = [0] * len(roadmap)
                i = 0
                flag = 0
                for k in range(len(roadmap)):
                    dx_op = new_x - roadmap[k].x
                    dy_op = new_y - roadmap[k].y
                    dz_op = new_z - roadmap[k].z
                    d_op = math.hypot(dx_op, dy_op, dz_op)
                    if d_op <= self.DIST_OPT:
                        cost[k] = roadmap[k].cost + d_op
                    else:
                        cost[k] = self.max_cost
                        i += 1
                        if i == len(roadmap):
                            flag = 1
                            break
                if flag == 1:
                    continue
                
                # Check whether a line in 3D space collides with a given set of cylinders
                cost_min = min(cost)
                index_cost_min = cost.index(cost_min)
                parent_point_node = roadmap[index_cost_min]
                parent_point = np.array([parent_point_node.x, parent_point_node.y, parent_point_node.z])
                new_point = np.array([new_x, new_y, new_z])
                d_point = np.linalg.norm(new_point - parent_point)
                flag_obs = 0

                for i in range(int(self.STEP_PER_SEARCH/self.avoid_obs)):
                    point = parent_point + i * self.avoid_obs * (new_point - parent_point) / d_point
                    if env.is_collide(point) is True:
                        flag_obs = 1
                        break
                    else:
                        pass
                if flag_obs == 1:
                    continue

                # add new point to roadmap
                node = Node(new_x, new_y, new_z, cost_min, parent_point_node)
                roadmap[len(roadmap)] = node
                break
            
            # check if the new point is close enough to goal
            if math.hypot(new_x-goal_given[0], new_y-goal_given[1], new_z-goal_given[2]) < self.ARRIVE_Radius:
                goal.parent = node
                print("Goal is found!")
                break

        # if there is not a new point in goal's radius after MAX_EPOCH
        if goal.parent == -1:
            cost = [0] * len(roadmap)
            for k in range(len(roadmap)):
                dx_op = new_x - roadmap[k].x
                dy_op = new_y - roadmap[k].y
                dz_op = new_z - roadmap[k].z
                d_op = math.hypot(dx_op, dy_op, dz_op)
                if d_op <= self.DIST_OPT: # 此处可能需要检查碰撞
                    cost[k] = roadmap[k].cost + d_op
                else:
                    cost[k] = self.max_cost
            cost_min = min(cost)
            index_cost_min = cost.index(cost_min)
            goal.parent = roadmap[index_cost_min]
            roadmap[len(roadmap)] = goal
            if goal.parent == -1:
                print("Goal is not found.")
            else:
                print("Goal is found!")
        
        # trace back the path from start to goal
        path.append([goal.x, goal.y, goal.z])
        parent = goal.parent
        while parent != -1:
            path.append([parent.x, parent.y, parent.z])
            parent = parent.parent

        return np.array(path)

    def sampling(self, env: FlightEnvironment):
        # sample until N_SAMPLE
        num = 0

        while num < self.N_SAMPLE:
            num = num + 1

            # randomly sampling from the environment
            sample_x = (random.random() * (env.env_width - 0)) + 0
            sample_y = (random.random() * (env.env_length - 0)) + 0
            sample_z = (random.random() * (env.env_height - 0)) + 0

            # Type conversion to tuple
            sample = (sample_x, sample_y, sample_z)

            # Check whether a 3D point lies outside the environment boundary
            if env.is_outside(sample) is True:
                continue
            else:
                # Check whether a point in 3D space collides with a given set of cylinders
                if env.is_collide(sample) is True:
                    continue
                else:
                    break
        
        if num == self.N_SAMPLE:
            print("sampling fail!")

        return sample