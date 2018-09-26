import numpy as np
import collections
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import copy
import random
import time
from bp94nball.qsr_util import circles
from bp94nball.qsr_util import qsr_part_of_characteristic_function
from bp94nball.qsr_util import qsr_disconnect_characteristic_function
from bp94nball.qsr_util import vec_length
from bp94nball.colors import cnames

colorList = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow','black', 'seagreen', 'pink',
             'navy','violet', 'crimson'] + list(cnames.keys())
rock_1 = [np.cos(np.pi / 4), np.sin(np.pi / 4), 10, 3]
stone = [np.cos(np.pi / 4 + np.pi / 100), np.sin(np.pi / 4 + np.pi / 100), 10, 3]
basalt = [np.cos(np.pi / 4 - np.pi / 100), np.sin(np.pi / 4 - np.pi / 100), 10, 3]
material = [np.cos(np.pi / 4 + np.pi / 30), np.sin(np.pi / 4 + np.pi / 30), 10, 3]
substance = [np.cos(np.pi / 4 + np.pi / 15), np.sin(np.pi / 4 + np.pi / 15), 10, 3]
entity = [np.cos(np.pi / 4 - np.pi / 200), np.sin(np.pi / 4 - np.pi / 200), 10, 3]

rock_2 = [np.cos(np.pi / 4), np.sin(np.pi / 4), 15, 3]
pop = [np.cos(np.pi / 4 + np.pi / 50), np.sin(np.pi / 4 + np.pi / 50), 10, 3]
jazz = [np.cos(np.pi / 4 - np.pi / 50), np.sin(np.pi / 4 - np.pi / 50), 10, 3]
music = [np.cos(np.pi / 3), np.sin(np.pi / 3), 10, 3]
communication = [np.cos(np.pi / 3 + np.pi / 50), np.sin(np.pi / 3 + np.pi / 50), 10, 3]
event = [np.cos(np.pi / 3 - np.pi / 50), np.sin(np.pi / 3 - np.pi / 50), 10, 3]

wDic = {
        'rock_1': rock_1,
        'stone': stone,
        'basalt': basalt,
        'material' : material,
        'substance' : substance,
        'entity': entity,
        'rock_2': rock_2,
        'pop': pop,
        'jazz': jazz,
        'music': music,
        'communication': communication,
        'event': event
    }

kw0 = [
        ['rock_1', 'material', 'rock_2'],
        ['stone', 'material', 'jazz'],
        ['basalt', 'material', 'communication'],
        ['material', 'substance', 'event'],
        ['substance', 'entity', 'music']
    ]

kw1=[
        ['rock_2', 'music', 'material'],
        ['pop', 'music', 'substance'],
        ['jazz', 'music', 'basalt'],
        ['music', 'communication', 'entity'],
        ['communication', 'event', 'entity'],
        ]

def do_func(funcName="part_of"):
    fdic = {
            "part_of": qsr_part_of_characteristic_function,
            "disconnect": qsr_disconnect_characteristic_function

            }
    if funcName in fdic.keys():
        return fdic[funcName]
    else:
        print("unknown qsr reltion:", funcName)
        return -1


def energy_2(ball1, ball2, func="part_of"):
    """
    compute the energy of ball1 being part of ball2

    ball1 \part_of ball2 = distance

    :param ball1:
    :param ball2:
    :return:
    """
    qsr_func = do_func(funcName=func)
    assert qsr_func != -1
    qsrIndicator = qsr_func(ball1, ball2)
    if qsrIndicator <= 0:
        return 0
    else:
        return min(0.999, 2/(1 + np.exp(-qsrIndicator)) - 1)


def loss(ball_w, ball_u, negBalls=[], func="part_of", func_neg="disconnect"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls: a list of balls as negative sample
    :return:
    """
    qsr_func = do_func(funcName=func)
    qsrIndicator = qsr_func(ball_u, ball_w)
    if qsrIndicator <= 0:
        Lw = 0
    else:
        Lw = energy_2(ball_u, ball_w, func=func)
    for ball_i in negBalls:
        qsr_func = do_func(funcName=func_neg)
        qsrIndicator = qsr_func(ball_i, ball_w)
        if qsrIndicator > 0:
            Lw += energy_2(ball_i, ball_w, func=func_neg)
    return Lw


def total_loss(klst, wDic, func="part_of", func_neg="disconnect", numNeg=0):
    value = 0
    for piece in klst:
        value += loss(wDic[piece[1]], wDic[piece[0]], negBalls=[wDic[nball] for nball in piece[2:]][0:numNeg],
                      func=func, func_neg=func_neg)
    return value


def partial_derative_lw(ball_w, ball_u, negBalls=[], func="part_of", func_neg="disconnect"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    alpha_w, lw, rw = ball_w[:-2], ball_w[-2], ball_w[-1]
    alpha_u, lu, ru = ball_u[:-2], ball_u[-2], ball_u[-1]
    hight = np.dot(alpha_w, alpha_u)
    e1 = energy_2(ball_u, ball_w, func=func)
    if e1 == 0:
        result = 0
    else:
        dis2 = lw * lw + lu * lu - np.multiply(2*lu*lw, hight)
        assert dis2 > 0
        result = (2 - 2*e1)* e1 * (lw - np.multiply(lu, hight))\
             / np.sqrt(dis2)
    i = 0
    for ball_i in negBalls:
        # print('neg i', i)
        i += 1
        alpha_i, li, ri = ball_i[:-2], ball_i[-2], ball_i[-1]
        hight = np.dot(alpha_i, alpha_w)
        e2 = energy_2(ball_i, ball_w, func=func_neg)
        if e2 != 0:
            dis2 = lw * lw + li * li - np.multiply(2*li*lw, hight)
            assert (dis2 > 0), "lw: {0}, li: {1}, hight: {2}, dis: {3}".format(float(lw), float(li),
                                                                               float(hight), float(dis2))
            result -= (2 - 2*e2)* e2 * (lw - np.multiply(li, hight))\
                 / np.sqrt(dis2)
    return result


def partial_derative_rw(ball_w, ball_u, negBalls=[], func="part_of", func_neg="disconnect"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    e1 = energy_2(ball_u, ball_w, func=func)
    result = (2*e1 - 2)*e1
    for ball_i in negBalls:
        e2 = energy_2(ball_i, ball_w, func=func_neg)
        result += (2 - 2*e2) * e2
    return result


def partial_derative_lu(ball_w, ball_u, negBalls=[], func="part_of", func_neg="disconnect"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    alpha_w, lw, rw = ball_w[:-2], ball_w[-2], ball_w[-1]
    alpha_u, lu, ru = ball_u[:-2], ball_u[-2], ball_u[-1]
    hight = np.dot(alpha_w, alpha_u)
    e1 = energy_2(ball_u, ball_w, func=func)
    result = 0
    if e1 != 0:
        dis2 = lw * lw + lu * lu - np.multiply(2 * lu * lw, hight)
        assert dis2 > 0
        result = (2 - 2*e1)* e1 * (lu - np.multiply(lw, hight)) \
                 / np.sqrt(dis2)
    return result


def partial_derative_ru(ball_w, ball_u, negBalls=[], func="part_of", func_neg="disconnect"):
    """

    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return:
    """
    e1 = energy_2(ball_u, ball_w, func=func)
    return (2 - 2*e1)*e1


def partial_derative_li(ball_w, ball_i, func="part_of", func_neg="disconnect"):
    """

    :param ball_w:
    :param ball_i:
    :param func:
    :return:
    """
    alpha_w, lw, rw = ball_w[:-2], ball_w[-2], ball_w[-1]
    alpha_i, li, ri = ball_i[:-2], ball_i[-2], ball_i[-1]
    hight = np.dot(alpha_i, alpha_w)
    result = 0
    e2 = energy_2(ball_i, ball_w, func=func_neg)
    if e2 != 0:
        dis2 = lw * lw + li * li - np.multiply(2 * li * lw, hight)
        assert dis2 > 0
        result = (2*e2 - 2)*e2*(li - np.multiply(lw, hight))\
             / np.sqrt(dis2)
    return result


def partial_derative_ri(ball_w, ball_i, func="part_of", func_neg="disconnect"):
    """

    :param ball_w:
    :param ball_i:
    :param func:
    :return:
    """
    e2 = energy_2(ball_i, ball_w, func=func_neg)
    return (2 - 2*e2) * e2


def update_e2_balls(ball_w, ball_u, negBalls=[], rmin=0, Lshift=100, func="part_of", func_neg="disconnect", dLoss=None, rate=0.1):
    """
    ball_w shall contain ball_u, and disconnects from balls in negBalls
    that is, ball_u is 'part of' ball_w
    :param ball_w:
    :param ball_u:
    :param negBalls:
    :param func:
    :return: new ball_w, ball_u, negBalls=[]
    """

    if dLoss == None:
        pass
    elif dLoss != 0:
        rate = 0.1 / dLoss
        rate = min(rate, 10)
    else:
        rate = 10000

    dL_dlw = partial_derative_lw(ball_w, ball_u, negBalls=negBalls, func=func, func_neg=func_neg)
    while ball_w[-2] - rate * dL_dlw < 0:
        ball_w[-2] += 2*rmin
    ball_w[-2] -= dL_dlw * rate

    dL_drw = partial_derative_rw(ball_w, ball_u, negBalls=negBalls, func=func, func_neg=func_neg)
    while ball_w[-1] - rate * dL_drw < rmin:
        # ball_w[-2] += Lshift
        ball_w[-1] += 2*rmin
        # ball_u[-1] += dL_drw * rate
    ball_w[-1] -= dL_drw * rate

    dL_dlu = partial_derative_lu(ball_w, ball_u, negBalls=negBalls, func=func, func_neg=func_neg)
    while ball_u[-2] - rate* dL_dlu < 0:
        ball_u[-2] += 2*rmin
        #ball_w[-2] += dL_dlu * rate
    ball_u[-2] -= dL_dlu * rate

    dl_dru = partial_derative_ru(ball_w, ball_u, negBalls=negBalls, func=func, func_neg=func_neg)
    while ball_u[-1] - rate * dl_dru < rmin:
        ball_u[-1] += 2*rmin
    ball_u[-1] -= dl_dru * rate

    for ball_i in negBalls:

        dL_dli = partial_derative_li(ball_w, ball_i, func=func, func_neg=func_neg)
        while ball_i[-2] - rate * dL_dli < 0:
            ball_i[-2] += 2*rmin
        ball_i[-2] -= rate * dL_dli

        dl_dri = partial_derative_ri(ball_w, ball_i, func=func, func_neg=func_neg)
        while ball_i[-1] - rate * dL_dli < 0:
            # print('*', ball_i[-1],rate, dL_dli)
            ball_i[-1] += 2*rmin
        ball_i[-1] -= rate * dl_dri

    return loss(ball_w, ball_u, negBalls=negBalls, func=func, func_neg=func_neg)


def train_e2_2D_batch(klst, func="part_of", negFunc="disconnect", rmin=0.1, Lshift=100, rate=0.1):
    """
    :param klst: a list of node knowledge
    :param func:
    :param negFunc:
    :param rate:
    :return:
    """
    lossLst = collections.deque(maxlen=5)

    while len(lossLst) <=2 or lossLst[-1] != 0 or (len(lossLst) > 2 and lossLst[-1] != lossLst[-2]):
        if len(lossLst) < 2:
            deltaLoss = None
        else:
            deltaLoss = np.abs(lossLst[-2] - lossLst[-1])
        loss1 = 0
        for kw in klst:
            loss1 += update_e2_balls(kw[1], kw[0], negBalls=kw[2:], rmin=rmin, Lshift=Lshift,
                                 func=func, func_neg=negFunc, dLoss=deltaLoss, rate=rate)
        lossLst.append(loss1)
        # print(lossLst[-1])
    return klst


def make_unique_ele(llst):
    rlt = llst[:1]
    for ele in llst[1:]:
        if ele not in rlt:
            rlt.append(ele)
    return rlt


def get_implicit_DC_list(klst):
    wlst = list(set([element[1] for element in klst]))
    rlt = []
    for w in wlst:
        elst = list(set([element[0] for element in klst if element[1] == w]))
        for u1, u2 in zip(elst[:-1], elst[1:]):
            rlt.append([u1, u1, u2])
    return rlt


def _get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


def train_e2_2D_iDC_batch_without_animation(klst, wDic={}, func="part_of", negFunc="disconnect", rmin=0.1,
                                            Lshift=100, rate=0.1):
    """
    iDC: consider implicity DC relations
    :param klst: a list of node knowledge
    :param func:
    :param negFunc:
    :param rate:
    :return:
    """
    lossLst = collections.deque(maxlen=5)
    iDCLst = get_implicit_DC_list(klst)
    print('iDCList', iDCLst)
    klst += iDCLst

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    balls = []
    for kpiece in klst:
        for ball in kpiece:
            if ball not in balls:
                balls.append(ball)
    N = len(balls)
    if N <= len(colorList):
        colList = colorList
    else:
        colList = colorList + _get_spaced_colors(N - len(colorList))

    while len(lossLst) <=2 or (lossLst[-1] != 0 and (len(lossLst) > 2 and lossLst[-1] != lossLst[-2])) :

        if len(lossLst) < 2:
            deltaLoss = None
        else:
            deltaLoss = np.abs(lossLst[-2] - lossLst[-1])
        for kw in klst:
           update_e2_balls(wDic[kw[1]], wDic[kw[0]], negBalls=[wDic[ele] for ele in kw[2:]],
                                         rmin=rmin, Lshift=Lshift, func=func, func_neg=negFunc,
                                         dLoss=deltaLoss, rate=rate)

        lossLst.append(total_loss(klst, wDic))
        # print(lossLst[-1])
    j = 0
    time.sleep(1)
    print(wDic)
    for ball in balls:
        vball = wDic[ball]
        circles(vball[0] * vball[-2], vball[1] * vball[-2], vball[-1],
                c=colList[j], alpha=0.5, edgecolor='none', label=ball)
        j += 1
    plt.show()
    return klst


def train_e2_2D_iDC_batch_animation(klst, wDic={}, func="part_of", negFunc="disconnect", rmin=0.1, Lshift=100, rate=0.1,
                                    restartTF = True):
    """
    iDC: consider implicity DC relations
    with animation
    :param klst: a list of node knowledge
    :param func:
    :param negFunc:
    :param rate:
    :return:
    """
    initDic = copy.deepcopy(wDic)
    lossLst = collections.deque(maxlen=5)
    iDCLst = get_implicit_DC_list(klst)
    print('iDCList', iDCLst)
    klst += iDCLst

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    balls = []
    for kpiece in klst:
        for ball in kpiece:
            if ball not in balls:
                balls.append(ball)

    N = len(balls)
    if N <= len(colorList):
        colList = colorList
    else:
        colList = colorList + _get_spaced_colors(N -len(colorList))

    def one_round_update(i):
        global wDic

        if len(lossLst) <=2 or (lossLst[-1] != 0 and (len(lossLst) > 2 and lossLst[-1] != lossLst[-2])):

            if len(lossLst) < 2:
                deltaLoss = None
            else:
                deltaLoss = np.abs(lossLst[-2] - lossLst[-1])

            for kw in klst:
                update_e2_balls(wDic[kw[1]], wDic[kw[0]], negBalls=[wDic[ele] for ele in kw[2:]],
                                         rmin=rmin, Lshift=Lshift, func=func, func_neg=negFunc,
                                         dLoss=deltaLoss, rate=rate)

            lossLst.append(total_loss(klst, wDic))
            print(lossLst)
            plt.clf()
            j = 0
            for ball in balls:
                vball = wDic[ball]
                circles(vball[0] * vball[-2], vball[1] * vball[-2], vball[-1],
                        c = colList[j], alpha=0.5, edgecolor='none', label=ball)
                j += 1

        elif len(lossLst) > 2 and lossLst[-1] == lossLst[-2] and restartTF:
            print('+++++ re-initialize +++++')
            plt.clf()
            wDic = copy.deepcopy(initDic)
            j = 0
            for ball in balls:
                vball = wDic[ball]
                circles(vball[0] * vball[-2], vball[1] * vball[-2], vball[-1],
                        c=colList[j], alpha=0.5, edgecolor='none', label=ball)
                j += 1
            lossLst.append(random.randint(1,10))


    a = anim.FuncAnimation(fig, one_round_update, frames=512, repeat=False)
    plt.show()


def train_e2_XD_iDC_batch(klst, iklst = True, wDic={}, func="part_of", negFunc="disconnect", numNeg=0, rmin=0.08,
                   Lshift=100, rate=0.1):
    """
    :param klst:
    :param wDic:
    :param func:
    :param negFunc:
    :param rmin:
    :param Lshift:
    :param rate:
    :return:
    """
    lossLst = collections.deque(maxlen=5)
    if iklst:
        iDCLst = get_implicit_DC_list(klst)
        print('iDCList', iDCLst)
        klst += iDCLst

    while len(lossLst) <= 2 or lossLst[-1] != 0:

        if len(lossLst) < 2:
            deltaLoss = None
        else:
            deltaLoss = np.abs(lossLst[-2] - lossLst[-1])
        for kw in klst :
            #print(kw)
            update_e2_balls(wDic[kw[1]], wDic[kw[0]], negBalls=[wDic[ele] for ele in kw[2:]][0:numNeg],
                            rmin=rmin, Lshift=Lshift, func=func, func_neg=negFunc,
                            dLoss=deltaLoss, rate=rate)

        lossLst.append(total_loss(klst, wDic, numNeg=numNeg))
        # print(lossLst[-1])
    return klst


def _write_klst_to_file(klst, kFile = "/Users/tdong/data/glove/glove.6B/klst.txt"):
    """
    save the time to generate klst every time
    :param klst:
    :param kFile:
    :return:
    """
    with open(kFile, 'w+') as kfh:
        for k in klst:
            kfh.write(" ".join(k) + "\n")


def step_train_e2_XD_klst_batch(klstFile="/Users/tdong/data/glove/glove.6B/klst.txt", wDic={},
                                func="part_of", negFunc="disconnect", rmin=0.08, Lshift=100, rate=0.1):
    """
    :param klst:
    :param wDic:
    :param func:
    :param negFunc:
    :param rmin:
    :param Lshift:
    :param rate:
    :return:
    """
    klst = []
    with open(klstFile, 'r') as kfh:
        for ln in kfh.readlines():
            klst.append(ln[:-1].split())
    totalLen = len(klst)
    size = 1
    while size < totalLen:
        start = 0
        while start + size < totalLen:
            end = start + size
            sublst = klst[start:end]
            train_e2_XD_iDC_batch(sublst, iklst=False, wDic=wDic, func=func, negFunc=negFunc, numNeg=0, rmin=rmin,
                                  Lshift=Lshift, rate=rate)
            print('size:', size, ' start: ', start, ' end: ', end)
            start += size
        size += size


def step_train_e2_XD_iDC_batch(klst, iklst = True,  wDic={}, func="part_of", negFunc="disconnect", rmin=0.08,
                   Lshift=100, rate=0.1):
    """
    :param klst:
    :param wDic:
    :param func:
    :param negFunc:
    :param rmin:
    :param Lshift:
    :param rate:
    :return:
    """
    if iklst:
        iDCLst = get_implicit_DC_list(klst)
        print('iDCList', iDCLst)
        klst += iDCLst

    _write_klst_to_file(klst)
    totalLen = len(klst)
    size = 1
    while size < totalLen:
        start = 0
        while start + size < totalLen:
            end = start + size
            sublst = klst[start:end]
            train_e2_XD_iDC_batch(sublst, iklst=False, wDic=wDic, func=func, negFunc=negFunc, rmin=rmin,
                                  Lshift=Lshift, rate=rate)
            print('size:', size, ' start: ', start, ' end: ', end)
            start += size
        size += size


def training_balls(klst=kw0+kw1, wDic=wDic, animate = True, d=2,   func="part_of", negFunc="disconnect", rmin=0.08,
                   Lshift=100, rate=0.1,  step_train = True, step_train_klst = True, restart=True):
    if animate and d<=3:
        print('in train_e2_2D_iDC_batch_with_animation\n')
        train_e2_2D_iDC_batch_animation(klst, wDic=wDic, func=func, negFunc=negFunc,
                                        rmin=rmin, Lshift=Lshift, rate=rate, restartTF =restart)
    elif not animate and d<=3:
        print('in train_e2_2D_iDC_batch_without_animation\n')
        train_e2_2D_iDC_batch_without_animation(klst, wDic=wDic, func=func, negFunc=negFunc,
                                                rmin=rmin, Lshift=Lshift, rate=rate)
    elif step_train_klst:
        print('in step_train_e2_XD_iDC_batch using klst stored in file\n')
        step_train_e2_XD_klst_batch(klstFile="/Users/tdong/data/glove/glove.6B/klst.txt", wDic=wDic, func=func,
                                    negFunc=negFunc, rmin=rmin, Lshift=Lshift, rate=rate)

    elif step_train:
        print('in step_train_e2_XD_iDC_batch\n')
        step_train_e2_XD_iDC_batch(klst,  wDic=wDic, func=func, negFunc=negFunc,
                              rmin=rmin, Lshift=Lshift, rate=rate)
    else:
        print('in train_e2_XD_iDC_batch\n')
        train_e2_XD_iDC_batch(klst, wDic=wDic, func=func, negFunc=negFunc,
                                            rmin=rmin, Lshift=Lshift, rate=rate)


def init_ball_with_word2vec(vecFile, wordsenseFile, IncDim = 1,  d=50, l0 = 10000, r0 = 30, rmin = 0.1):
    """
    vecFile is a file of pre-trained word embedding.
    :param vecFile:
    :param trainFile:
    :param IndDim: 0, not to increase the dimension, >0, increase d+1th dimension with the value IncDim
    :param d: dimension of the word embedding of word2vec
    :param
    :return:
    """
    dic0 = {}
    dic1 = {}
    with open(vecFile, 'r') as vfh:
        for ln in vfh.readlines():
            wlst = ln.split()
            vec = [float(ele) for ele in wlst[1:]]
            l1 = vec_length(vec)
            dic0[wlst[0]] = [ele/l1 for ele in vec]
            if IncDim:
                dic0[wlst[0]].append(IncDim)
                l1 = vec_length(dic0[wlst[0]])
                dic0[wlst[0]] = [ele / l1 for ele in dic0[wlst[0]]]


    with open(wordsenseFile, 'r') as tfh:
        cnt = tfh.read()
        wslst = cnt.split('\n')

    last_wd = ""
    for ws in wslst:
        current_wd = ws.split('.')[0]
        dic1[ws] = copy.copy(dic0[current_wd])
        if last_wd == current_wd:
            deltaL += l0 + l0
            dic1[ws].append(deltaL)
            dic1[ws].append(r0)
        else:
            deltaL = l0
            dic1[ws].append(l0)
            dic1[ws].append(r0)
        last_wd = current_wd
        assert len(dic1[ws]) == 52 + IncDim
    return dic1


def load_training_file(trainFile):
    """
    :param trainFile:
    :return:
    """
    klst = []
    with open(trainFile, 'r') as tfh:
        for ln in tfh.readlines():
            wlst = ln.split()
            klst.append(wlst)
    return klst


def save_balls_to_file(ballDic, outFile, d = 300):
    lines = []
    for key in ballDic.keys():
        lines.append("{0} {1}\n".format(key, " ".join([str(ele) for ele in ballDic[key]])))
    with open(outFile, 'w') as ofh:
        ofh.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart', type=bool, default=True)
    args = parser.parse_args()
    restartTF = args.restart
    training_balls(restart = restartTF)