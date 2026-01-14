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
import bisect


class TrajectoryPlanner:
    def __init__(self, smoothness=3, dt=0.1):
        """
        轨迹规划类
        
        参数:
            smoothness: 轨迹平滑度（B样条阶数）
            dt: 时间步长（秒）
        """
        self.smoothness = smoothness  # Order of B-Splines
        self.dt = dt  # step of time
        
    def bspline_trajectory(self, path, num_points=100):
        """
        使用B样条曲线生成平滑轨迹
        
        参数:
            path: N×3的路径点数组
            num_points: 生成的轨迹点数
            
        返回:
            trajectory: 平滑轨迹点数组
            t_points: 对应的时间数组
            derivatives: 各阶导数（速度、加速度等）
        """
        if len(path) < 4:
            print("路径点太少，直接返回原路径")
            return path, np.linspace(0, 1, len(path)), None
        
        # 提取坐标
        x = path[:, 0]
        y = path[:, 1]
        z = path[:, 2]
        
        # 参数化路径（基于累积弦长）
        u = self.chord_length_parameterization(path)
        
        # 拟合B样条曲线
        try:
            tck, u_new = splprep([x, y, z], u=u, k=self.smoothness, s=0)
        except Exception as e:
            print(f"B样条拟合失败: {e}")
            return path, u, None
        
        # 生成等间距的采样点
        u_fine = np.linspace(0, 1, num_points)
        
        # 计算轨迹点
        trajectory = np.array(splev(u_fine, tck)).T
        
        # 计算一阶导数（速度）
        velocity = np.array(splev(u_fine, tck, der=1)).T
        
        # 计算二阶导数（加速度）
        acceleration = np.array(splev(u_fine, tck, der=2)).T
        
        # 计算时间（基于速度归一化）
        speed = np.linalg.norm(velocity, axis=1)
        # 避免除零
        speed[speed < 1e-6] = 1e-6
        dt_cumsum = np.cumsum(1.0 / speed)
        t_points = dt_cumsum / dt_cumsum[-1] * self.calculate_total_time(speed)
        
        derivatives = {
            'velocity': velocity,
            'acceleration': acceleration,
            'speed': speed
        }
        
        return trajectory, t_points, derivatives
    
    def chord_length_parameterization(self, path):
        """
        基于弦长的参数化
        """
        if len(path) < 2:
            return np.array([0])
        
        # 计算累积弦长
        distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
        
        # 归一化到[0,1]
        if cumulative_dist[-1] > 0:
            return cumulative_dist / cumulative_dist[-1]
        else:
            return np.linspace(0, 1, len(path))
    
    def calculate_total_time(self, speed, desired_max_speed=2.0):
        """
        计算总时间，基于期望的最大速度
        """
        avg_speed = np.mean(speed)
        # 确保总时间合理
        total_time = len(speed) * self.dt * (desired_max_speed / avg_speed)
        return max(total_time, 1.0)  # 至少1秒
    
    def bezier_trajectory(self, path, num_points=100):
        """
        使用贝塞尔曲线生成轨迹（替代方法）
        """
        if len(path) < 2:
            return path, np.linspace(0, 1, len(path)), None
        
        n = len(path) - 1
        trajectory = []
        t_values = np.linspace(0, 1, num_points)
        
        for t in t_values:
            point = np.zeros(3)
            for i in range(n + 1):
                # 贝塞尔基函数
                coeff = self.bernstein_poly(n, i, t)
                point += coeff * path[i]
            trajectory.append(point)
        
        trajectory = np.array(trajectory)
        
        # 简单的时间参数化
        t_points = np.linspace(0, self.calculate_total_time_simple(path), num_points)
        
        return trajectory, t_points, None
    
    @staticmethod
    def bernstein_poly(n, i, t):
        """伯恩斯坦多项式"""
        from math import comb
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def calculate_total_time_simple(self, path, avg_speed=1.0):
        """简单计算总时间"""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += np.linalg.norm(path[i+1] - path[i])
        return total_distance / avg_speed
    
    def polynomial_trajectory(self, path, num_points=100):
        """
        使用多项式插值生成轨迹
        每段路径使用五次多项式
        """
        if len(path) < 2:
            return path, np.linspace(0, 1, len(path)), None
        
        trajectory = []
        t_segments = []
        
        # 为每段路径生成多项式轨迹
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            # 使用五次多项式（位置、速度、加速度连续）
            t_segment = np.linspace(0, 1, num_points // (len(path) - 1))
            
            # 简单线性插值（可替换为更高阶多项式）
            for t in t_segment:
                point = start + t * (end - start)
                trajectory.append(point)
                t_segments.append(t + i)  # 每段在时间上连续
        
        trajectory = np.array(trajectory)
        t_points = np.array(t_segments) * self.dt
        
        return trajectory, t_points, None
    
    def plot_trajectory(self, path, trajectory, t_points, derivatives=None):
        """
        绘制轨迹和原始路径点
        
        参数:
            path: 原始路径点
            trajectory: 平滑轨迹
            t_points: 时间点
            derivatives: 导数信息
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 3D轨迹图
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
        
        # 2. X坐标随时间变化
        ax2 = fig.add_subplot(222)
        ax2.plot(t_points, trajectory[:, 0], 'b-', linewidth=2, label='trajectory')
        ax2.plot(np.linspace(0, t_points[-1], len(path)), path[:, 0], 
                'ro', alpha=0.5, label='waypoint')
        ax2.set_xlabel('t(s)')
        ax2.set_ylabel('x axis')
        ax2.set_title('x-t curve')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Y坐标随时间变化
        ax3 = fig.add_subplot(223)
        ax3.plot(t_points, trajectory[:, 1], 'g-', linewidth=2, label='trajectory')
        ax3.plot(np.linspace(0, t_points[-1], len(path)), path[:, 1], 
                'ro', alpha=0.5, label='waypoint')
        ax3.set_xlabel('t(s)')
        ax3.set_ylabel('y axis')
        ax3.set_title('y-t curve')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Z坐标随时间变化
        ax4 = fig.add_subplot(224)
        ax4.plot(t_points, trajectory[:, 2], 'r-', linewidth=2, label='trajectory')
        ax4.plot(np.linspace(0, t_points[-1], len(path)), path[:, 2], 
                'ro', alpha=0.5, label='waypoint')
        ax4.set_xlabel('t(s)')
        ax4.set_ylabel('z axis')
        ax4.set_title('z-t curve')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 如果存在导数信息，绘制速度加速度图
        if derivatives is not None:
            self.plot_derivatives(t_points, derivatives)