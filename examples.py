import numpy as np
import matplotlib.pyplot as plt

u_H = 15
c_a = 4
c_l = 1.5
# c_h > u_H
la = 0.4
print(['a', u_H * 0.4])
"""
f_thetah_q_h = 4 / 7
f_thetah_q_l = 2 / 7
f_thetal_q_h = 3 / 7
f_thetal_q_l = 5 / 7


f_thetah_q_h = 5 / 7
f_thetah_q_l = 2 / 7
f_thetal_q_h = 2 / 7
f_thetal_q_l = 5 / 7


"""
f_thetah_q_h = 6/7 #   # 4.5/7, 5.6/7 6/7,6.9/7
f_thetah_q_l = 1 - f_thetah_q_h
f_thetal_q_h = f_thetah_q_l
f_thetal_q_l = f_thetah_q_h


# ========================================================
# this is the expected return if:
# (1) firm conducts short-term project
# (2) investor has a belief on long-term project when CSR is observed
# ==========================================================
def f(alpha_A):  # r_{S}
    aa = u_H * alpha_A * f_thetah_q_h * f_thetah_q_l
    cc = alpha_A * f_thetah_q_h + (1 - alpha_A) * f_thetah_q_l
    bb = u_H * alpha_A * f_thetal_q_h * f_thetal_q_l
    dd = alpha_A * f_thetal_q_h + (1 - alpha_A) * f_thetal_q_l
    return aa / cc + bb / dd


# ========================================================
# this is the expected return if:
# (1) firm conducts long-term project
# (2) investor has a belief on long-term project when CSR is observed
# ==========================================================
def ff(alpha_A):  # r_{L}
    aa = u_H * alpha_A * f_thetah_q_h * f_thetah_q_h
    cc = alpha_A * f_thetah_q_h + (1 - alpha_A) * f_thetah_q_l
    bb = u_H * alpha_A * f_thetal_q_h * f_thetal_q_h
    dd = alpha_A * f_thetal_q_h + (1 - alpha_A) * f_thetal_q_l
    return aa / cc + bb / dd


# ========================================================
# this is the expected return if:
# (1) firm conducts short-term project
# (2) investor has a belief for long-term project when non-CSR activity
#     is observed
# ==========================================================

def m(alpha_B):  # r_{S}
    aa = u_H * alpha_B * f_thetah_q_h * f_thetah_q_l
    cc = alpha_B * f_thetah_q_h + (1 - alpha_B) * f_thetah_q_l
    bb = u_H * alpha_B * f_thetal_q_h * f_thetal_q_l
    dd = alpha_B * f_thetal_q_h + (1 - alpha_B) * f_thetal_q_l
    return aa / cc + bb / dd


# ========================================================
# this is the expected return if:
# (1) firm conducts long-term project
# (2) investor has a belief for long-term project when non-CSR activity
#     is observed
# ==========================================================

def mm(alpha_B):  # r_{L}
    aa = u_H * alpha_B * f_thetah_q_h * f_thetah_q_h
    cc = alpha_B * f_thetah_q_h + (1 - alpha_B) * f_thetah_q_l
    bb = u_H * alpha_B * f_thetal_q_h * f_thetal_q_h
    dd = alpha_B * f_thetal_q_h + (1 - alpha_B) * f_thetal_q_l
    return aa / cc + bb / dd


def g():
    alpha_a = np.linspace(0.001, 0.999, 100)
    aa = 1000
    bb = 0
    for each in alpha_a:
        res = f(each)
        if res < aa:
            bb = each
            aa = res
    return aa, bb


def test(x, y, la):
    a = x * x / (la * y + (1 - la) * x)
    b = (1 - x) * (1 - x) / (la * (1 - y) + (1 - la) * (1 - x))
    print(a + b)


# Non-CSR pooling
def non_csr_pooling():
    alpha_A_list = np.linspace(0.01, 0.99, 500)
    alphaB = 0
    rSB = m(alphaB)
    result = []
    for alpha_A in alpha_A_list:
        rSA = f(alpha_A)
        rLA = ff(alpha_A)
        if rSA - c_a < rSB and rLA - c_a - c_l < rSB:
            result.append(alpha_A)
    if len(result) != 0:
        # print(['alphab', result])
        print(['Non-CSR pooling', 'alphaB=0', 'alpha_A is in [{},{}]'.format(str(min(result)), str(max(result)))])
    return len(result)


#
def non_csr_separating():
    alphB = la
    rSB = m(alphB)
    rLB = mm(alphB)
    alpha_A_list = np.linspace(0, 0.99, 500)
    result = []
    result_rSB = []
    result_rLB = []
    result_rLA = []
    result_rSA = []
    for alpha_A in alpha_A_list:
        rSA = f(alpha_A)
        # print([rSA,rSA-c_a,rSB])

        # for type c_{l}, alpha^{A} is off-equilibrium, and need additional restriction
        # rLB-c_l>rLA-c_l-c_a and rLB-c_l>rSA-c_a and rLB-c_l>rSB (they will choose LB)
        rLA = ff(alpha_A)
        if rSB > rSA - c_a and rLB - c_l > rLA - c_l - c_a and rLB - c_l > rSA - c_a and rLB - c_l > rSB:
            result.append(alpha_A)
            #result_rSB.append(rSB)
            #rLB = mm(alphB)
            #result_rLB.append(rLB - c_l)
            #rLA = ff(alpha_A)
            #result_rLA.append(rLA - c_l - c_a)
            #rSA = f(alpha_A)
            #result_rSA.append(rSA - c_a)
    #plot_fig(result, result_rSB, result_rLB, result_rLA, result_rSA)

    if len(result) != 0:
        print(['Non-CSR separating', 'alphaB={}'.format(str(la)),
               'alpha_A is in [{},{}]'.format(str(min(result)), str(max(result)))])

    return len(result)


def csr_hybrid():
    alpha_A_list = np.linspace(0.01 + la, 0.99, 500)
    result = []
    for alpha_A in alpha_A_list:
        # rH = ff(alpha_A)
        rSA = f(alpha_A)
        #rLA = ff(alpha_A)
        if abs(rSA - c_a) < 0.01:
            result.append(alpha_A)
    if len(result) != 0:
        print(['CSR hybrid', 'alphaB=0', 'alpha_A is in [{},{}]'.format(str(min(result)), str(max(result)))])
    return len(result)


def csr_pooling():
    alpha_B_list = np.linspace(0.01, 0.99, 500)
    alpha_A_list = np.linspace(0.01, 0.99, 500)
    result_B = []
    result_A = []
    for alpha_B in alpha_B_list:
        for alpha_A in alpha_A_list:
            rSA = f(alpha_A)
            rSB = m(alpha_B)
            rLA = ff(alpha_A)
            rLB = mm(alpha_B)
            if abs(rSA - c_a - rSB) < 0.01 and abs(rLA-c_a-c_l-(rLB-c_l))<0.01 and rLB-c_l>rSB and rLB-c_l>rSA-c_a:
                result_B.append(alpha_B)
                result_A.append(alpha_A)
    if len(result_A) != 0:
        print(['CSR pooling', 'alpha_A is in [{},{}]'.format(str(min(result_A)), str(max(result_A))),
               'alpha_B is in [{},{}]'.format(str(min(result_B)), str(max(result_B)))])


def plot_fig(x, y1, y2, y3, y4, title='a'):
    plt.plot(x, y1, 'k,--', alpha=0.8, label='rSB')
    plt.plot(x, y2, 'g,--', alpha=0.8, label='rLB')
    plt.plot(x, y3, 'r,--', alpha=0.8, label='rLA')
    plt.plot(x, y4, 'b,--', alpha=0.8, label='rSA')
    plt.legend(loc="upper left")
    plt.xlabel('alpha_{A}')
    plt.ylabel('value')
    plt.title(title)
    plt.show()


# la = 0.2
# b = ff(alpha_A=la)
# a = f(alpha_A=la)
# print(b)
# print(a)

non_csr_pooling()
non_csr_separating()
csr_hybrid()
csr_pooling()
