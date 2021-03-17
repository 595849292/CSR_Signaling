import numpy as np

u_H = 15
c_a = 4
c_l = 1.5
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
f_thetah_q_h = 6.7/7#4.7,6.7,6.7/7
f_thetah_q_l = 1-f_thetah_q_h
f_thetal_q_h =f_thetah_q_l
f_thetal_q_l = f_thetah_q_h


def f(alpha_A):  # r_{L}
    aa = u_H * alpha_A * f_thetah_q_h * f_thetah_q_l
    cc = alpha_A * f_thetah_q_h + (1 - alpha_A) * f_thetah_q_l
    bb = u_H * alpha_A * f_thetal_q_h * f_thetal_q_l
    dd = alpha_A * f_thetal_q_h + (1 - alpha_A) * f_thetal_q_l
    return aa / cc + bb / dd


def ff(alpha_A):  # r_{H}
    aa = u_H * alpha_A * f_thetah_q_h * f_thetah_q_h
    cc = alpha_A * f_thetah_q_h + (1 - alpha_A) * f_thetah_q_l
    bb = u_H * alpha_A * f_thetal_q_h * f_thetal_q_h
    dd = alpha_A * f_thetal_q_h + (1 - alpha_A) * f_thetal_q_l
    return aa / cc + bb / dd


def m(alpha_B):  # r_{L}
    aa = u_H * alpha_B * f_thetah_q_h * f_thetah_q_l
    cc = alpha_B * f_thetah_q_h + (1 - alpha_B) * f_thetah_q_l
    bb = u_H * alpha_B * f_thetal_q_h * f_thetal_q_l
    dd = alpha_B * f_thetal_q_h + (1 - alpha_B) * f_thetal_q_l
    return aa / cc + bb / dd


def mm(alpha_B):  # r_{H}
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
    alpha_A_list = np.linspace(0.01, la, 500)
    result = []
    for alpha_A in alpha_A_list:
        rH = ff(alpha_A)
        rL = f(alpha_A)
        if rL < c_a and rH < c_a + c_l:
            result.append(alpha_A)
    if len(result) != 0:
        #print(['alphab', result])
        print(['Noc-CSR pooling', 'alphaB=0', 'alpha_A is in [{},{}]'.format(str(min(result)), str(max(result)))])
    alpha_B_list = np.linspace(0, 0.99, 500)
    result1 = []
    for alpha_B in alpha_B_list:
        rH = mm(alpha_B)
        rL = m(alpha_B)
        if rL >= rH - c_l:
            result1.append(alpha_B)
    if len(result1) != 0:
        #print(['alphab',result1])
        print(['Noc-CSR pooling', 'alphaA=0', 'alpha_B is in [{},{}]'.format(str(min(result1)), str(max(result1)))])


#
def non_csr_separating():
    alphB = la
    rLB = m(alphB)
    rHB = m(alphB)
    alpha_A_list = np.linspace(0, la - 0.001, 500)
    result = []
    for alpha_A in alpha_A_list:
        rHA = ff(alpha_A)
        rLA = f(alpha_A)
        if rLA - c_a < rLB and rHB > rHA - c_a and rHB - c_l > rLB:
            result.append(alpha_A)
    if len(result) != 0:
        print(['Non-CSR separating', 'alphaB={}'.format(str(la)),
               'alpha_A is in [{},{}]'.format(str(min(result)), str(max(result)))])

def csr_hybrid():
    alpha_A_list = np.linspace(0.01+la, 0.999, 500)
    result = []
    for alpha_A in alpha_A_list:
        rH = ff(alpha_A)
        rL = f(alpha_A)
        if abs(rL- c_a)<0.01 and rH-rL >c_l:
            result.append(alpha_A)
    if len(result) != 0:
        print(['CSR hybrid', 'alphaB=0', 'alpha_A is in [{},{}]'.format(str(min(result)), str(max(result)))])

def csr_separating():
    alpha_B_list = np.linspace(0, la, 500)
    rLA=f(la)
    rHA=ff(la)
    result = []
    for alpha_B in alpha_B_list:
        #rH = ff(alpha_A)
        rLB = m(alpha_B)
        if rLA-c_a>rLB and rHA-rLA>c_l:
            result.append(alpha_B)
    if len(result) != 0:
        print(['CSR separating', 'alphaA={}'.format(str(la)), 'alpha_B is in [{},{}]'.format(str(min(result)), str(max(result)))])

# la = 0.2
# b = ff(alpha_A=la)
# a = f(alpha_A=la)
# print(b)
# print(a)

non_csr_pooling()
non_csr_separating()
csr_hybrid()
#csr_separating()

