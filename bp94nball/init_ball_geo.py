import numpy as np
from collections import defaultdict
import random
from bp94nball.qsr_util import cos_of_vec
from bp94nball.qsr_util import dis_between
from bp94nball.qsr_util import qsr_part_of_characteristic_function
from bp94nball.qsr_util import qsr_disconnect_characteristic_function


def init_ball_geo(dic1, klst, l0=0, rmin= 0.1):
    """
    for u, w in klst:
        cos(u,w)
        lu = lw cos(u,w)
        rw = lw sin(u,w)
        if dic1[w][-2] == l0 and dic1[u][-2] == l0:
            dic1[w][-1] = rw + rmin
            dic1[u][-1] = rmin
        elif dic1[w][-2] == l0 and dic1[u][-2] != l0:
            dic1[w][-2] = dic1[u][-2] / cos(u,w) + dic1[u][-1]
        elif dic1[w][-2] != l0 and dic1[u][-2] == l0:
            dic1[w][-1] = rw + rmin
            dic1[u][-1] = rmin
        else: #dic1[w][-2] != l0 and dic1[u][-2] != l0:
            dic1[w][-2] = max(dic1[w][-2], lw)
            dic1[w][-1] = max(dic1[w][-1], rw +  dic1[u][-1])

    :param dic1:
    :param klst:
    :return:
    """
    for kp in klst:
        u, w = kp[0], kp[1]
        cos_u_w = cos_of_vec(dic1[u][:-2], dic1[w][:-2])
        lu = dic1[w][-2] * cos_u_w
        lw = dic1[u][-2] / cos_u_w
        rw = dic1[w][-2] * np.sqrt(1 - cos_u_w*cos_u_w)

        if dic1[w][-2] % l0 == 0 and dic1[u][-2] % l0 == 0:
            dic1[w][-1] = rw + rmin
            dic1[u][-1] = rmin
            dic1[u][-2] = lu
        elif dic1[w][-2] % l0 == 0 and dic1[u][-2] % l0 != 0:
            dic1[w][-2] = lw + dic1[u][-1]
        elif dic1[w][-2] % l0 != 0 and dic1[u][-2] % l0 == 0:
            dic1[w][-1] = rw + rmin
            dic1[u][-2] = lu
            dic1[u][-1] = rmin
        elif dic1[w][-2] % l0 != 0 and dic1[u][-2] % l0 != 0:
            dic1[w][-2] = max(dic1[w][-2], lw)
            dic1[w][-1] = max(dic1[w][-1], rw + dic1[u][-1])


def update_one_ball(xball, rate, ballDic):
    """
    update l, r of xball with the rate
    :param xball:
    :param rate:
    :param ballDic:
    :return:
    """
    ballDic[xball][-1] *= 1 + rate
    ballDic[xball][-2] *= 1 + rate


def update_all_family_together(ballDic, ball, rate, tFDic):
    """
    If ball needs to change size (with rate), its l, r, and all family members (all members in the family tree) shall
    change with the same rate.
    :param ballDic:
    :param ball:
    :param rate:
    :param tFDic:
    :return:
    """
    for xball in tFDic[ball]:
        update_one_ball(xball, rate, ballDic)


def init_ball_family_geo(ballDic, fklstFile, l0=100, rmin=0.1):
    """
    this is the first step in init
    :param ballDic:
    :param fklstFile:
    :param l0:
    :param rmin:
    :return:
    """
    fdic = dict()
    num = len(ballDic)
    with open(fklstFile, 'r') as ffh:
        i = 0
        for ln in ffh.readlines():
            fdic[i] = ln[:-1].split()
            ball0 = ballDic[fdic[i][0]]
            ball0[-2] = random.randint(l0, num*10)
            ball0[-1] = rmin

            for j in range(len(fdic[i]) -1):
                ball1 = ballDic[fdic[i][j]]
                ball2 = ballDic[fdic[i][j+1]]
                cos_b1_b2 = cos_of_vec(ball1[:-2], ball2[:-2])
                assert cos_b1_b2 > 0 and cos_b1_b2 < 1
                ball2[-2] = ball1[-2] / cos_b1_b2
                ball2[-1] = ball2[-2] * np.sqrt(1 - cos_b1_b2 * cos_b1_b2) + ball1[-1]
                ballDic[fdic[i][j + 1]][-2] = ball2[-2]
                ballDic[fdic[i][j + 1]][-1] = ball2[-1] + 0.01
                if ballDic[fdic[i][j]] != ballDic[fdic[i][j + 1]]:
                    pIndex = qsr_part_of_characteristic_function(ballDic[fdic[i][j]], ballDic[fdic[i][j + 1]])
                    assert pIndex <=0, "{0} pIndex: {1}".format(" ".join([fdic[i][j], fdic[i][j+1]]), pIndex)
        i += 1


def separate_family_tree(ballDic, tFDic):
    """
    this is the second step in init: balls in diffferet family trees shall be separated
    all_roots = list(tFDic.keys())
    updated_root = all_roots[:1]
    for root in allFamilyRoots[1:]:
        while not all(disconnects(root, ele) for ele in updated_root):
            update_all_family_together(ballDic, root, 0.1, tFDic)

    :param ballDic:
    :param rFamilyFile:
    :return:
    """
    allFamilyRoots = list(tFDic.keys())
    updated_root = allFamilyRoots[:1]
    for root in allFamilyRoots[1:]:
        while not all(qsr_disconnect_characteristic_function(root, ele)<0 for ele in updated_root):
            update_all_family_together(ballDic, root, 0.1, tFDic)


def init_balls_of_family_forests(ballDic, fklstFile, rFamilyFile, l0=100, rmin=0.1, step1=False, step2=True):
    """
    this the main function for the carefully-designed init state
    :param ballDic: if step1==False, ballDic shall be the result of the step 1.
    :param fklstFile:
    :param rFamilyFile:
    :param l0:
    :param rmin:
    :return:
    """
    if step1:
        init_ball_family_geo(ballDic, fklstFile, l0=l0, rmin=rmin)
    if step2:
        tFDic = load_family_tree(rFamilyFile)
        separate_family_tree(ballDic, tFDic)


def enlarge_ball_geo(dic1, klst):
    """
    for u, w in klst:
        cos(u,w)
        lu = lw cos(u,w)
        rw = lw sin(u,w)
    :param dic1:
    :param klst:
    :return:
    """
    for kp in klst:
        u, w = kp[0], kp[1]
        dis = dis_between(np.multiply(dic1[u][-2], dic1[u][:-2]), np.multiply(dic1[w][-2], dic1[w][:-2]))
        if dic1[w][-1] < dis + dic1[u][-1]:
            dic1[w][-1] = dis + dic1[u][-1]


def get_relation_chains(klst, familyFile):
    """
    k['a'] = 'b'  -> d[1] = ['a', 'b']
    k['b'] = 'c'  -> if 'b' in d[1], d[1].append('c')
    k['c'] = 'd'
    k: child-parent relation
    :param klst: [['a','c'],['c','d'],['d','e']]
    :return:
    """
    d = dict()
    for k in klst:
        u, w  = k[0], k[1]
        found = False
        num = len(d.keys())
        for dk in d.keys():
            if u in d[dk] and w not in d[dk]:
                d[dk].append(w)
                found = True
                break
        if not found:
            d[num] = [u, w]

    with open(familyFile, 'w') as ffh:
        for dk in d.keys():
            lst = []
            for ele in d[dk]:
                if ele not in lst:
                    lst.append(ele)
            ffh.write(' '.join(lst)+'\n')


def get_family_tree(familyFile, rFamilyFile):
    """
    just inverse the order of words in each line of familyFile,
    and save it to rFamilyFile
    :param familyFile:
    :param rFamilyFile:
    :return:
    """
    with open(rFamilyFile, 'a+') as rfh:
        with open(familyFile, 'r') as ffh:
            for ln in ffh.readlines():
                words = ln[:-1].split()
                words.reverse()
                print(words)
                rfh.write(" ".join(words)+"\n")


def load_family_tree(rFamilyFile):
    """
    return a diction, dic[top grandparent] = [all members in the family tree]
    :param rFamilyFile:
    :return: a dictionary
    """
    dic = defaultdict(list)
    with open(rFamilyFile, 'r') as rfh:
        for ln in rfh.readlines():
            words = ln[:-1].split()
            dic[words[0]] += words[1:]
            dic[words[0]] = list(set(dic[words[0]]))
    return dic


if __name__ == "__main__":
    klstFile = "/Users/tdong/data/glove/glove.6B/klst.txt"
    fklstFile = "/Users/tdong/data/glove/glove.6B/familyKlst.txt"
    rFamilyFile = "/Users/tdong/data/glove/glove.6B/rFamilyKlst.txt"
    # klst = []
    # with open(klstFile, 'r') as kfh:
    #    for ln in kfh.readlines():
    #        klst.append(ln[:-1].split())

    # get_relation_chains(klst, fklstFile)
    # get_family_tree(fklstFile, rFamilyFile)
    load_family_tree(rFamilyFile)