"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


class TrajectoryPlanner:
    def __init__(self, smoothness=3, dt=0.1):
        self.smoothness = smoothness  # Order of B-Splines
        self.dt = dt  # step of time
        
    def bspline_trajectory(self, path, num_points=100):
        # lack of waypoints
        if len(path) < 4:
            return path, np.linspace(0, 1, len(path)), None
        
        x = path[:, 0]
        y = path[:, 1]
        z = path[:, 2]
        
        # calculate parameters of path
        u = self.chord_length_parameterization(path)
        
        # B-Splines
        tck, _ = splprep([x, y, z], u=u, k=self.smoothness, s=0)
        
        # generate sampling points of equal distance
        u_fine = np.linspace(0, 1, num_points)
        
        # calculate trajectory points
        trajectory = np.array(splev(u_fine, tck)).T
        
        # calculate velocity
        velocity = np.array(splev(u_fine, tck, der=1)).T
        
        # calculate speed
        speed = np.linalg.norm(velocity, axis=1)
        speed[speed < 1e-6] = 1e-6
        dt_cumsum = np.cumsum(1.0 / speed)

        # calculate time of trajectory and path
        t_points = dt_cumsum / dt_cumsum[-1] * self.calculate_total_time(speed)
        t_path = np.zeros(len(path))

        # corresbonding of path and time
        for i, point in enumerate(path):
            distances = np.linalg.norm(trajectory - point, axis=1)
            min_idx = np.argmin(distances)
            t_path[i] = t_points[min_idx]
        
        return trajectory, t_points, t_path
    
    def chord_length_parameterization(self, path):
        if len(path) < 2:
            return np.array([0])
        
        # calculate cumulative distance
        distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
        
        # normalize to [0,1]
        if cumulative_dist[-1] > 0:
            return cumulative_dist / cumulative_dist[-1]
        else:
            return np.linspace(0, 1, len(path))
    
    def calculate_total_time(self, speed, desired_max_speed=2.0):
        avg_speed = np.mean(speed)
        total_time = len(speed) * self.dt * (desired_max_speed / avg_speed)
        return max(total_time, 1.0)
    
    def plot_trajectory(self, path, trajectory, t_points, t_path):
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', linewidth=2, label='trajectory')
        ax1.plot(path[:, 0], path[:, 1], path[:, 2], 
                'ro--', alpha=0.5, label='waypoint')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D trajectory')
        ax1.legend()
        ax1.grid(True)

        # 2. x-t plot
        ax2 = fig.add_subplot(222)
        ax2.plot(t_points, trajectory[:, 0], 'b-', linewidth=2, label='trajectory')
        ax2.plot(t_path, path[:, 0], 'ro', alpha=0.5, label='waypoint')
        ax2.set_xlabel('t(s)')
        ax2.set_ylabel('x axis')
        ax2.set_title('x-t curve')
        ax2.legend()
        ax2.grid(True)
        
        # 3. y-t plot
        ax3 = fig.add_subplot(223)
        ax3.plot(t_points, trajectory[:, 1], 'g-', linewidth=2, label='trajectory')
        ax3.plot(t_path, path[:, 1], 'ro', alpha=0.5, label='waypoint')
        ax3.set_xlabel('t(s)')
        ax3.set_ylabel('y axis')
        ax3.set_title('y-t curve')
        ax3.legend()
        ax3.grid(True)
        
        # 4. z-t plot
        ax4 = fig.add_subplot(224)
        ax4.plot(t_points, trajectory[:, 2], 'r-', linewidth=2, label='trajectory')
        ax4.plot(t_path, path[:, 2], 'ro', alpha=0.5, label='waypoint')
        ax4.set_xlabel('t(s)')
        ax4.set_ylabel('z axis')
        ax4.set_title('z-t curve')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()