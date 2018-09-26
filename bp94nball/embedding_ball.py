
import numpy as np
from bp94nball.qsr_util import get_qsr
from bp94nball.qsr_util import dis_between_ball_centers


def train_nballs(p0, p1, psyn=[], pant=[], pns = []):
    """

    :param p0: vec of word
    :param p1: vec of uppercategroy of word
    :param psyn: vec of synonmy of word
    :param pant: vec of antonmy of word
    :param pns: vec of words not related with word (negative example)
    :return: % of psyn are in n-ball of p1
             % of pant are in n-ball of p1
             % of pns are not in n-ball of p1
    """
    pass


def update_to_disconnected(ball1, ball2, lrate = 0.0005, epsilon = 0.1, min_radius = 0.1, max = 1000000):
    """

    each dimension of Center points shall vibrate within the limit of epsilon
    r1, r2 decreases till two times of the minimal radius, then vibrates with a delta in (-minimal radius, minimal radius)

    :param ball1:
    :param ball2:
    :param lrate: learning rate
    :param epsilon: vibration limit
    :param min_radius: minimal value of the radius
    :param max: maximum number of iteration
    :return: ball1, ball2, True|False
    """
    count = 0
    qsr = get_qsr(ball1, ball2)
    while (qsr != 'disconnect' and count < max):
        new_center1 = ball1[:-1]
        new_center2 = ball2[:-1]
        if np.random.choice([True, False]):
            new_center1 = random_vibrating_within(ball1[:-1], epsilon)
            ball1[-1] = decrease_and_vibrating_around_min(ball1[-1], lrate = lrate, min_radius = min_radius)
        else:
            new_center2 = random_vibrating_within(ball2[:-1], epsilon)
            ball2[-1] = decrease_and_vibrating_around_min(ball2[-1], lrate = lrate, min_radius = min_radius)
        qsr = get_qsr(np.append(new_center1, ball1[-1]), np.append(new_center2, ball2[-1]))
        count += 1

    return np.append(new_center1, ball1[-1]), np.append(new_center2, ball2[-1]), qsr == 'disconnect'


def update_to_part_of_by_dev(ball1, ball2, lrate = 0.0005, ratio = 100, max = 1000000):
    """

    :param ball1:
    :param ball2:
    :param lrate: learning rate
    :param ratio: learning ratio between ball2 and ball1
    :param max: maximum iteration
    :return:
    """
    count = 0
    qsr = get_qsr(ball1, ball2)
    while qsr != 'part_of' and count < max:
        d = dis_between_ball_centers(ball1, ball2)
        if d == 0:
            if ball1[-1] - lrate > 0:
                ball1[-1] -= lrate
            ball2[-1] += ratio * lrate
        else:
            if ball1[-1] - lrate > 0:
                ball1[-1] -= lrate
            ball2[-1] += ratio * lrate
            ball1[:-1] = ball1[:-1] - lrate * (ball1[:-1] - ball2[:-1]) / d
            ball2[:-1] = ball2[:-1] - lrate * (ball2[:-1] - ball1[:-1]) / d
        qsr = get_qsr(ball1, ball2)
        count += 1
        print(ball1, ball2)
    return ball1, ball2, qsr == 'part_of'


def update_to_part_of(ball1, ball2, lrate = 0.0005, epsilon = 0.1, min_radius = 0.1, max = 1000000):
    """
    each dimension of Center points shall vibrate within the limit of epsilon
    r1 decreases till two times of the minimal radius, then vibrates with a delta in (-minimal radius, minimal radius)
    r2 increases without limit

    :param ball1:
    :param ball2:
    :param lrate: learning rate
    :param epsilon: vibration limit
    :param min_radius: minimal value of the radius
    :param max: maximum number of iteration
    :return: ball1, ball2, True|False
    """
    count = 0
    qsr = get_qsr(ball1, ball2)
    while qsr != 'part_of' and count < max:
        new_center1 = ball1[:-1]
        new_center2 = ball2[:-1]
        if np.random.choice([True, False]):
            new_center1 = random_vibrating_within(ball1[:-1], epsilon)
            ball1[-1] = decrease_and_vibrating_around_min(ball1[-1], lrate = lrate, min_radius = min_radius)
        else:
            new_center2 = random_vibrating_within(ball2[:-1], epsilon)
            ball2[-1] = increase_with_rate(ball2[-1], lrate = lrate)

        qsr = get_qsr(np.append(new_center1, ball1[-1]), np.append(new_center2, ball2[-1]))
        count += 1

    return np.append(new_center1, ball1[-1]), np.append(new_center2, ball2[-1]), qsr == 'part_of'


def random_vibrating_within(nparray, epsilon = 0.1):
    """

    :param nparray:
    :param epsilon:
    :return:
    """
    sz = np.size(nparray)
    delta = (np.random.rand(sz) - np.random.rand(sz)) * epsilon / 2
    return nparray + delta


def decrease_and_vibrating_around_min(radius, lrate = 0.0005, min_radius = 0.1):
    """

    :param radius:
    :param lrate:
    :param min_radius:
    :return:
    """
    if radius < min_radius:
        return radius + np.random.random() * min_radius
    elif radius < 2*min_radius:
        return radius - np.random.random() * min_radius
    else:
        return radius - lrate


def increase_with_rate(radius, lrate = 0.0001):
    """

    :param radius:
    :param lrate:
    :return:
    """
    return radius + lrate