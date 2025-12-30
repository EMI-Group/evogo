import numpy as np
import gym
from .helper import run_env


def _landing_nn_controller(obs, ws):
    # 8 -> 8 -> 4 total 108
    w1 = ws[:8 * 8].reshape(8, 8)
    b1 = ws[8 * 8:8 * 8 + 8]
    w2 = ws[8 * 8 + 8:8 * 8 + 8 + 8 * 4].reshape(4, 8)
    b2 = ws[8 * 8 + 8 + 8 * 4:]
    hidden = np.tanh(w1 @ obs + b1)
    output = w2 @ hidden + b2
    select = np.argmax(output)
    return select


def _landing_heuristic_controller(s, w):
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        a = 2
    elif angle_todo < -w[11]:
        a = 3
    elif angle_todo > +w[11]:
        a = 1
    return a


def get_heuristic_landing_func():
    env = gym.make("LunarLander-v2", continuous=False)

    def f(xs):
        r = [run_env(env, _landing_heuristic_controller, [x / 2 for x in xs]) for _ in range(20)]
        r = np.asarray(r)
        order = np.argsort(r)
        r[order[:5]] = np.nan
        r[order[-5:]] = np.nan
        return np.nanmean(r)

    return f, 12


def get_nn_landing_func():
    env = gym.make("LunarLander-v2", continuous=False)
    MAX_RANGE = 5

    def f(_x):
        ws = np.array(_x)
        ws = ws * MAX_RANGE * 2 - MAX_RANGE
        return run_env(env, _landing_nn_controller, ws)

    return f, 108


def _walker_controller(obs, ws):
    # 17 -> 8 -> 6 total 198
    # [0:136] -> w1, [136:144] -> b1, [144:192] -> w2, [192:198] -> b2
    w1 = ws[:136].reshape(8, 17)
    b1 = ws[136:144]
    w2 = ws[144:192].reshape(6, 8)
    b2 = ws[192:]
    hidden = np.tanh(w1 @ obs + b1)
    output = np.tanh(w2 @ hidden + b2)
    return output


def get_walker_func():
    env = gym.make("Walker2d-v4")
    MAX_RANGE = 5

    def f(_x):
        ws = np.array(_x)
        ws = ws * MAX_RANGE * 2 - MAX_RANGE
        return run_env(env, _walker_controller, ws)

    return f, 198


def _hopper_controller(obs, ws):
    # 11 -> 13 -> 3 total 198
    w1 = ws[:11 * 13].reshape(13, 11)
    b1 = ws[11 * 13:11 * 13 + 13]
    w2 = ws[11 * 13 + 13:11 * 13 + 13 + 13 * 3].reshape(3, 13)
    b2 = ws[11 * 13 + 13 + 13 * 3:]
    hidden = np.tanh(w1 @ obs + b1)
    output = np.tanh(w2 @ hidden + b2)
    return output


def get_hopper_func():
    env = gym.make("Hopper-v4")
    MAX_RANGE = 5

    def f(_x):
        ws = np.array(_x)
        ws = ws * MAX_RANGE * 2 - MAX_RANGE
        return run_env(env, _hopper_controller, ws)

    return f, 198


def _cheetah_controller(obs, ws):
    # 17 -> 8 -> 6 total 198
    # [0:136] -> w1, [136:144] -> b1, [144:192] -> w2, [192:198] -> b2
    w1 = ws[:136].reshape(8, 17)
    b1 = ws[136:144]
    w2 = ws[144:192].reshape(6, 8)
    b2 = ws[192:]
    hidden = np.tanh(w1 @ obs + b1)
    output = np.tanh(w2 @ hidden + b2)
    return output


def get_half_cheetah_func():
    env = gym.make("HalfCheetah-v4")
    MAX_RANGE = 5

    def f(_x):
        ws = np.array(_x)
        ws = ws * MAX_RANGE * 2 - MAX_RANGE
        return run_env(env, _cheetah_controller, ws)

    return f, 198


def _ant_controller(obs, ws):
    # 27 -> 11 -> 8 total 404
    w1 = ws[:27 * 11].reshape(11, 27)
    b1 = ws[27 * 11:27 * 11 + 11]
    w2 = ws[27 * 11 + 11:27 * 11 + 11 + 11 * 8].reshape(8, 11)
    b2 = ws[27 * 11 + 11 + 11 * 8:]
    hidden = np.tanh(w1 @ obs + b1)
    output = np.tanh(w2 @ hidden + b2)
    return output


def get_ant_func():
    env = gym.make("Ant-v4")
    MAX_RANGE = 5

    def f(_x):
        ws = np.array(_x)
        ws = ws * MAX_RANGE * 2 - MAX_RANGE
        return run_env(env, _ant_controller, ws)

    return f, 404


if __name__ == '__main__':
    ff, nn = get_walker_func()
    fs = []
    for _ in range(10):
        x = np.random.uniform(0, 1, size=(nn, ))
        fs.append(ff(x))
    print(fs)

    ff, nn = get_hopper_func()
    fs = []
    for _ in range(10):
        x = np.random.uniform(0, 1, size=(nn, ))
        fs.append(ff(x))
    print(fs)

    ff, nn = get_half_cheetah_func()
    fs = []
    for _ in range(10):
        x = np.random.uniform(0, 1, size=(nn, ))
        fs.append(ff(x))
    print(fs)

    ff, nn = get_ant_func()
    fs = []
    for _ in range(10):
        x = np.random.uniform(0, 1, size=(nn, ))
        fs.append(ff(x))
    print(fs)
