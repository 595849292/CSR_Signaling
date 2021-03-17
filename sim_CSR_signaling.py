from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


class Sim_CSR:
    def __init__(self, uH, llambda, CA, CL, k0qh, k0ql):
        self.u_H = uH
        self.llambda = llambda
        self.CA = CA
        self.CL = CL
        self.k0qh = k0qh
        self.k0ql = k0ql
        #res=integrate.quad(self.check, 0, 1)
        #print(res)

        if self.llambda * self.u_H <= self.CA + self.CL:
            print(['assumption lambda_l*u_{H}>CA+CL does not hold'])
        # print('CA need be greater than ' + str(R_LB[0]))

    def check(self,theta):
        dem = self.llambda * self.f_qH(theta) + (1 - self.llambda) * self.f_qL(theta)
        return dem
    # =====================================================
    # pdf: f(theta|q_{H}) AND f(theta|q_{L})
    # =====================================================

    def f_qH(self, theta):
        k1 = 1 - 0.5 * self.k0qh
        return self.k0qh * theta + k1
        # return theta + 0.5

    def f_qL(self, theta):
        k1 = 1 - 0.5 * self.k0ql
        return self.k0ql * theta + k1
        # return -theta + 1.5

    # =====================================================
    # the degree of f(theta|q_{H}) AND f(theta|q_{L})
    # =====================================================
    def m(self):
        degree1 = np.arctan(self.k0qh) * 360 / 2 / np.pi
        degree2 = np.arctan(self.k0ql) * 360 / 2 / np.pi + 180
        m = 180 - (degree2 - degree1)
        return m

    # =====================================================
    # check if  R_{H}^{A} is increasing?
    # =====================================================
    def RHA_montone(self, theta, *arg):
        p_ACh, = arg
        alphaA = self.llambda / (self.llambda + (1 - self.llambda) * p_ACh)
        nume = (alphaA * self.f_qH(theta) + (1 - alphaA) * self.f_qL(theta)) ** 2
        dem = self.f_qH(theta) * self.f_qH(theta) * (self.f_qH(theta) - self.f_qL(theta))
        # print(['RHA_montone',nume,dem,nume / dem])
        return nume / dem

    # ==================================================
    # prob_fun and prob_fun2
    # R_{L}^{A}(\alpha^{A}(m),m)=C_{A}
    # where, prob_fun present the inside part of integral
    # ===================================================

    def prob_R_LA(self, theta, *arg):
        p_ACh, = arg

        dem = self.llambda * self.f_qH(theta) + (1 - self.llambda) * self.f_qL(theta) * p_ACh
        nume = self.u_H * self.llambda * self.f_qH(theta) * self.f_qL(theta)

        return nume / dem

    def pro_fun2(self, p_ACh):
        v = integrate.quad(self.prob_R_LA, 0, 1, args=(p_ACh,))
        # print(v)
        return v[0] - self.CA

    # ==================================================
    # find p(A|C_{H})
    # ===================================================

    def p_ACH(self):
        x0 = optimize.root_scalar(self.pro_fun2, bracket=[0.001, 0.999], method='brentq')
        if x0.converged:
            #print([x0,'valid'])
            return x0.root
        else:
            #print(x0)
            return None

    # ==================================================
    # check:
    # (1) R_{H}^{A}(\alpha^{A}(m),m)\geq C_{L}+C_{A}
    # (2) R_{L}^{B}(\lambda_{L},m)<C_{A}
    # ===================================================

    def fun_R_HA(self, theta, *args):
        p_ACh, = args
        dem = self.llambda * self.f_qH(theta) + (1 - self.llambda) * self.f_qL(theta) * p_ACh
        nume = self.u_H * self.llambda * self.f_qH(theta) * self.f_qH(theta)
        return nume / dem

    def fun_R_LB(self, theta):
        # dem = self.llambda * self.f_qH(theta) + (1 - self.llambda) * self.f_qL(theta)
        # print(['dem',dem])
        dem = self.llambda * self.f_qH(theta) + (1 - self.llambda) * self.f_qL(theta)
        nume = self.u_H * self.llambda * self.f_qH(theta) * self.f_qL(theta)
        return nume / dem

    # =========================================================
    # CSR seperating equilibrium vs CSR hybrid equilibrium
    # ========================================================
    def CSR_Separating(self, message=True):
        R_HB = integrate.quad(self.fun_R_HB, 0, 1)
        R_LB = integrate.quad(self.fun_R_LB, 0, 1)
        distance = R_HB[0] - R_LB[0]
        m = self.m()
        #print(['distance', distance,'csr'])
        if distance >= self.CL and R_LB[0] >= self.CA:
            if message:
                print([m, 'CSR separating equilibrium '])
            return m, 'csr separating'

        else:
            try:
                prob = self.p_ACH()
                R_HA = integrate.quad(self.fun_R_HA, 0, 1, args=(prob,))
                if R_HA[0] > self.CA + self.CL:
                    return m, 'csr hybrid'
            except:
                pass
        return None, None

    # =========================================================
    # No CSR separating equilibrium necessary and sufficient condition
    # No CSR pooling equilibrium necessary and sufficient condition
    # =========================================================

    def fun_R_HB(self, theta):
        dem = self.llambda * self.f_qH(theta) + (1 - self.llambda) * self.f_qL(theta)
        nume = self.u_H * self.llambda * self.f_qH(theta) * self.f_qH(theta)
        return nume / dem

    def funRHA(self, theta, *args):
        alpha, = args
        dem = alpha * self.f_qH(theta) + (1 - alpha) * self.f_qL(theta)
        nume = self.u_H * alpha * self.f_qH(theta) * self.f_qH(theta)
        return nume / dem

    def funRLA(self, theta, *args):
        alpha, = args
        dem = alpha * self.f_qH(theta) + (1 - alpha) * self.f_qL(theta)
        nume = self.u_H * alpha * self.f_qH(theta) * self.f_qL(theta)
        return nume / dem

    def NoCSR_pooling(self, message=True):
        alpha = 0.02
        R_HA = integrate.quad(self.funRHA, 0, 1, args=(alpha,))
        R_LA = integrate.quad(self.funRLA, 0, 1, args=(alpha,))
        m = self.m()
        if R_HA[0] < self.CA + self.CL and R_LA[0] < self.CA:
            if message:
                print([m, 'Non-CSR pooling equilibrium '])
            return m, 'non-csr pooling'
        alphaB = np.linspace(0, 0.99, 1000)
        for each_alphaB in alphaB:
            RHB = integrate.quad(self.funRHA, 0, 1, args=(each_alphaB,))
            RLB = integrate.quad(self.funRLA, 0, 1, args=(each_alphaB,))
            d = RHB[0] - RLB[0]
            if d <= self.CL:
                if message:
                    print([m, 'Non-CSR pooling equilibrium '])
                return m, 'non-csr pooling'
        return None, None

    def NoCSR_separating(self, message=True):
        R_HB = integrate.quad(self.fun_R_HB, 0, 1)
        R_LB = integrate.quad(self.fun_R_LB, 0, 1)
        distance = R_HB[0] - R_LB[0]
        m = self.m()
        if distance >= self.CL:
            if message:
                print([m, 'No CSR separating equilibrium '])
            return m, 'non-csr separating'
        return None, None

    # =========================================================
    # x1:No CSR pooling     equilibrium --> 1
    # x2:No CSR separating  equilibrium --> 2
    # x3:CSR separating  equilibrium --> 3
    # x4:CSR    Hybrid      equilibrium --> 4
    # =========================================================

    def fun_main(self):
        x1, y1, x2, y2, x3, y3, x4,y4 = None, None, None, None, None, None, None,None
        m, res = self.NoCSR_pooling(message=False)
        if res == 'non-csr pooling':
            x1 = m
            y1 = 1
        m, res = self.NoCSR_separating(message=False)
        if res == 'non-csr separating':
            x2 = m
            y2 = 2
        m, res = self.CSR_Separating(message=False)
        if res == 'csr separating':
            x3=m
            y3=3
        elif res == 'csr hybrid':
            x4 =m
            y4 =3
        return x1, y1, x2, y2, x3, y3, x4,y4

    def fun_main_with_csr(self):
        x1, y1, x2, y2, x3, y3 = None, None, None, None, None, None
        mm, ress = self.CSR_Separating()
        if ress == 'pooling':
            x1 = mm
            y1 = 1
        elif ress == 'separating':
            x2 = mm
            y2 = 2
        elif ress == 'hybrid':
            x3 = mm
            y3 = 3
        return x1, y1, x2, y2, x3, y3


def plot_fig(x1, y1, x2, y2, x3, y3, x4, y4,title='a'):
    plt.plot(x1, y1, 'g,--', alpha=0.8,label='Non-CSR completely pooling equilibrium')
    plt.plot(x2, y2, 'r,--', alpha=0.8, label='Non-CSR partially pooling equilibrium')
    #plt.plot(x3, y3, 'b,--', alpha=0.8, label='CSR separating equilibrium')
    plt.plot(x4, y4, 'k,--', alpha=0.8, label='CSR hybrid equilibrium')

    plt.legend(loc="upper left")
    plt.xlabel('m')
    plt.ylabel('value')
    plt.title(title)
    plt.show()


def plot_fig2(x1, y1, x2, y2, x3=None, y3=None, title='a'):
    plt.plot(x1, y1, 'k,--', alpha=0.8)
    plt.plot(x2, y2, 'k', alpha=0.8, label='CSR separating equilibrium')
    # plt.plot(x3, y3, 'k', alpha=0.8, label='CSR hybrid equilibrium')

    plt.legend(loc="upper left")
    plt.xlabel('m')
    plt.ylabel('value')
    plt.title(title)
    plt.show()


def sim1():
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    x4 = []
    y4 = []
    llambda = 0.5
    CA = 1.2
    CL = 0.01
    uH = 2.5
    epsilon = np.linspace(0, 1.5, 100)
    for each in epsilon:
        k0qh = 1 / 10 + each
        k0ql = -1 / 10 - each
        instance = Sim_CSR(uH, llambda, CA, CL, k0qh, k0ql)
        xx1, yy1, xx2, yy2, xx3, yy3, xx4,yy4 = instance.fun_main()
        if xx1 is not None:
            x1.append(xx1)
            y1.append(yy1)
        if xx2 is not None:
            x2.append(xx2)
            y2.append(yy2)
        if xx3 is not None:
            x3.append(xx3)
            y3.append(yy3)
        if xx4 is not None:
            x4.append(xx4)
            y4.append(yy4)

    plot_fig(x1, y1, x2, y2, x3, y3,x4,y4, title='$C_{A}=1.2,\lambda_{L}=0.5,C_{L}=0.01,u_{H}=2.5$')


def sim2():
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    y33 = []
    llambda = 0.1
    CA = 0.6
    CL = 1
    uH = 5
    epsilon = np.linspace(0, 1.5, 300)
    for each in epsilon:
        k0qh = 1 / 100 + each
        k0ql = -1 / 100 - each
        instance = Sim_CSR(uH, llambda, CA, CL, k0qh, k0ql)
        xx1, yy1, xx2, yy2, xx3, yy3 = instance.fun_main_with_csr()
        if xx1 is not None:
            x1.append(xx1)
            y1.append(yy1)
        if xx2 is not None:
            x2.append(xx2)
            y2.append(yy2)
        if xx3 is not None:
            x3.append(xx3)
            y3.append(yy3)

    plot_fig2(x1, y1, x2, y2, x3, y3, title='$C_{A}=0.3,\lambda_{L}=0.3,C_{L}=0.1,u_{H}=8$')


if __name__ == '__main__':
    sim1()
