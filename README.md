# 题目名称：Robot Navigation

## 一、问题解答：

1. ### 基于particles的位置，计算最终定位robot的唯一位置并输出到屏幕上，说明计算方式。

   答：我采取了算数平均值的计算方法，并分别采用了加权和不加权的方式，从效果上来看没有太大差别。

   <img src="https://github.com/gitxiangxiang/robot_navigation/blob/master/%E6%88%AA%E5%9B%BE/problem1.png" alt="问题1的截图" style="zoom: 33%;" />

   ```python
   # 不加权值做算数平均值
   pred_loc = [sum(particles[:, 0])/N, sum(particles[:, 1])/N]
   cv2.putText(img, "predicted location", (30, 80), 1, 1.0, (0, 0, 255))
   cv2.putText(img, "(%.2f, %.2f)" % (pred_loc[0], pred_loc[1]), (200, 80), 1, 1.0, (0, 0, 255))
   # 加权算数平均值
   mean_x = np.average(particles[:, 0], weights=weights)
   mean_y = np.average(particles[:, 1], weights=weights)
   cv2.putText(img, "predicted location(with weights) (%.2f, %.2f)" % (mean_x, mean_y), (30, 120), 1, 1.0, (0, 0, 255))
   ```

2. ### 修改weights的分布为帕累托分布（当前使用的是正态分布）

   <img src="https://github.com/gitxiangxiang/robot_navigation/blob/master/%E6%88%AA%E5%9B%BE/problem2.png" alt="问题2截图" style="zoom:33%;" />

   我不太清楚我注释掉的采用正态分布的那条语句是如何起作用的，因此我根据我自己的理解，将robot与landmarks之间的距离和粒子与landmarks之间的距离作差并取绝对值，经帕累托分布后作为weights并连乘。

   ```python
   def update(particles, weights, z, R, landmarks):
       weights.fill(1.)
       for i, landmark in enumerate(landmarks):
           # distance为每个粒子与指定landmark之间的距离
           distance = np.power((particles[:, 0] - landmark[0]) ** 2 + (particles[:, 1] - landmark[1]) ** 2, 0.5)
           # 修改为帕累托分布，还不知道R变量的作用，所以忽略了它
           weights *= scipy.stats.pareto.pdf(abs(z[i]-distance), 3)
           # weights *= scipy.stats.norm(distance, R).pdf(z[i])
   
       weights += 1.e-300  # avoid round-off to zero
       weights /= sum(weights)
   
   ```

3. ### 为landmark和robot之间的距离增加随机误差，观察定位结果

   在这个问题上我有点疑惑，因为原有代码已经包含了一个符合标准正态分布的误差，可能我对“随机误差”这个概念的理解不准确，实话说我也不清楚什么是随机误差，因此我就在原来的基础上增加了一个∈[-5, 5)的随机数。尽管此处距离为负没有什么影响，但我还是打算保证其为正值。观察定位结果没有很大变化。

   <img src="https://github.com/gitxiangxiang/robot_navigation/blob/master/%E6%88%AA%E5%9B%BE/problem3.png" alt="问题3截图" style="zoom:33%;" />

   ```python
   # 修改为随机误差（在原来的基础上添加了[-5,5)的随机误差）
   zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
   zs = [i+(np.random.rand()-0.5)*sensor_std_err*2 if i > sensor_std_err else i*np.random.rand() for i in zs]
           
   ```

4. ### 修改particle filtering过程，消除随机误差对定位结果的影响（不一定完成，可讨论思路）

   经过观察，在将weights的分布修改为帕累托分布后，重采样的粒子数变得很少。我刚开始盲目的将帕累托分布中的参数设为3，在我尝试将参数设为0.1时，可见的粒子数变多了。但这还未涉及修改particle filtering过程。可能是我对该算法涉及的原理不理解，而且我也没有观察到当前模型有哪些不足，因此没有想出来如何修改。不过我觉得重采样那部分有改进的空间。而且每次定位可以再加一些随机粒子进去。增加landmark的数量应该也会提高准确度。

   <img src="https://github.com/gitxiangxiang/robot_navigation/blob/master/%E6%88%AA%E5%9B%BE/problem4_1.png" alt="问题4截图" style="zoom:33%;" />

## 二、过程与总结

我首先在网上查阅了particles filtering算法的基本流程。在看了几篇帖子并似懂非懂之后，我开始阅读代码。由于我的Python水平仅限于简单的语法，涉及的这些库的方法都不了解，因此只能一个一个查，不过毕竟代码比较短，很快就把除了核心算法之外的代码看完了。之后我从鼠标回调的函数开始一行一行阅读，这部分花了不少时间，凭着一行行调试，查看变量的内容以及多次重复才大体看懂。之后修改就比较容易了。

### 对particle filtering的理解

这个算法很像加权平均数的思想，而且利用大量粒子附带一定误差的运动来减少定位的误差。

### 修改后的源码

```python
import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)
import cv2

# 绘制多边形
def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))


def drawCross(img, center, r, g, b):
    d = 5
    # 线宽
    t = 2
    LINE_AA = cv2.LINE_AA
    # LINE_AA = cv2.LINE_AA if cv2.__version__[0] == '3' else cv2.CV_AA
    color = (r, g, b)
    ctrx = center[0, 0]
    ctry = center[0, 1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)


def mouseCallback(event, x, y, flags, null):
    global center
    global trajectory
    global previous_x
    global previous_y
    global zs

    # center为实际位置
    center = np.array([[x, y]])
    # 用于画轨迹
    trajectory = np.vstack((trajectory, np.array([x, y])))
    # noise=sensorSigma * np.random.randn(1,2) + sensorMu

    # 计算朝向
    if previous_x > 0:
        heading = np.arctan2(np.array([y - previous_y]), np.array([previous_x - x]))

        if heading > 0:
            heading = -(heading - np.pi)
        else:
            heading = -(np.pi + heading)

        # 应该是上一次采样与本次采样点的距离，没有加入误差
        distance = np.linalg.norm(np.array([[previous_x, previous_y]]) - np.array([[x, y]]), axis=1)

        std = np.array([2, 4])
        u = np.array([heading, distance])
        predict(particles, u, std, dt=1.)
        # landmarks 与 robot之间的距离，加入了按标准正态分布的误差
        # 修改为随机误差（在原来的基础上添加了10以内的随机误差）
        zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
        """
        这里修改了！！！
        """
        zs = [i+(np.random.rand()-0.5)*sensor_std_err*2 if i > sensor_std_err else i*np.random.rand() for i in zs]
        update(particles, weights, z=zs, R=50, landmarks=landmarks)

        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)

    previous_x = x
    previous_y = y


WIDTH = 800
HEIGHT = 600
WINDOW_NAME = "Particle Filter"

# sensorMu=0
# sensorSigma=3

sensor_std_err = 5

# 生成均匀分布的粒子
def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles


def predict(particles, u, std, dt=1.):
    N = len(particles)
    # 加一个标准正态分布的误差？
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    # 粒子按robot运动方式来运动
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist


def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        # distance为每个粒子与指定landmark之间的距离
        distance = np.power((particles[:, 0] - landmark[0]) ** 2 + (particles[:, 1] - landmark[1]) ** 2, 0.5)
        """
        这里修改了！！！
        """
        # 修改为帕累托分布，还不知道R变量的作用，所以忽略了它
        weights *= scipy.stats.pareto.pdf(abs(z[i]-distance), 0.1)
        # weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)


def neff(weights):
    return 1. / np.sum(np.square(weights))


def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    # 好奇怪的采样方式
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

# 期望方差（加权）
def estimate(particles, weights):
    pos = particles[:, 0:1]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)


x_range = np.array([0, 800])
y_range = np.array([0, 600])

# Number of partciles
N = 400

landmarks = np.array([[144, 73], [410, 13], [336, 175], [718, 159], [178, 484], [665, 464]])
NL = len(landmarks)
particles = create_uniform_particles(x_range, y_range, N)

weights = np.array([1.0] * N)

# Create a black image, a window and bind the function to window
img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouseCallback)

center = np.array([[-10, -10]])

trajectory = np.zeros(shape=(0, 2))
robot_pos = np.zeros(shape=(0, 2))
previous_x = -1
previous_y = -1
DELAY_MSEC = 50

while (1):

    cv2.imshow(WINDOW_NAME, img)
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    drawLines(img, trajectory, 0, 255, 0)
    drawCross(img, center, r=255, g=0, b=0)

    # landmarks
    for landmark in landmarks:
        cv2.circle(img, tuple(landmark), 10, (255, 0, 0), -1)

    # draw_particles:
    for particle in particles:
        cv2.circle(img, tuple((int(particle[0]), int(particle[1]))), 1, (255, 255, 255), -1)

    # 基于粒子的预测点(不加权值做算数平均值)
    pred_loc = [sum(particles[:, 0])/N, sum(particles[:, 1])/N]

    if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
        break

    cv2.circle(img, (10, 10), 10, (255, 0, 0), -1)
    cv2.circle(img, (10, 30), 3, (255, 255, 255), -1)
    cv2.putText(img, "Landmarks", (30, 20), 1, 1.0, (255, 0, 0))
    cv2.putText(img, "Particles", (30, 40), 1, 1.0, (255, 255, 255))
    cv2.putText(img, "Robot Trajectory(Ground truth)", (30, 60), 1, 1.0, (0, 255, 0))
    cv2.putText(img, "predicted location", (30, 80), 1, 1.0, (0, 0, 255))
    cv2.putText(img, "(%.2f, %.2f)" % (pred_loc[0], pred_loc[1]), (200, 80), 1, 1.0, (0, 0, 255))
    cv2.putText(img, "actual location (%.2f, %.2f)" % (center[0][0], center[0][1]), (30, 100), 1, 1.0, (0, 0, 255))
    # 补充加权平均值
    mean_x = np.average(particles[:, 0], weights=weights)
    mean_y = np.average(particles[:, 1], weights=weights)
    cv2.putText(img, "predicted location(with weights) (%.2f, %.2f)" % (mean_x, mean_y), (30, 120), 1, 1.0, (0, 0, 255))
    drawLines(img, np.array([[10, 55], [25, 55]]), 0, 255, 0)

cv2.destroyAllWindows()
```

