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

    #center为实际位置
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
        zs = (np.linalg.norm(landmarks - center, axis=1)
              + (np.random.randn(NL) * sensor_std_err) + (np.random.rand(NL) * 10))
        # zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
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
        # 修改为帕累托分布，还不知道R变量的作用，所以忽略了它
        weights *= scipy.stats.pareto.pdf(abs(z[i]-distance), 3)
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