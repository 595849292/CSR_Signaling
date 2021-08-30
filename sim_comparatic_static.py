from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


class Comparative_static:
    def __init__(self, uH, llambda, CA, CL):
        self.u_H = uH
        self.llambda = llambda
        self.CA = CA
        self.CL = CL

        #R_LB = integrate.quad(self.fun_R_LB, 0, 1)
        #print('CA need be greater than ' + str(R_LB[0]))

    # =====================================================
    # pdf: f(theta|q_{H}) AND f(theta|q_{L})
    # =====================================================

    def f_qH(self, theta):
        return theta + 0.5

    def f_qL(self, theta):
        return -theta + 1.5

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
        #print([p_ACh,v[0] - self.CA])
        return v[0] - self.CA

    # ==================================================
    # find p(A|C_{H})
    # ===================================================

    def p_ACH(self):
        x0 = optimize.root_scalar(self.pro_fun2, bracket=[0.001, 0.999], method='brentq')
        if x0.converged:
            return x0.root
        else:
            print(x0)
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
        dem = self.llambda * self.f_qH(theta) + (1 - self.llambda) * self.f_qL(theta)
        nume = self.u_H * self.llambda * self.f_qH(theta) * self.f_qH(theta)
        return nume / dem

    def fun_main(self, message=True):
        if self.CA >= self.u_H:
            if message:
                print('CA>u_{H}')
            return None
        if self.CL > self.u_H:
            if message:
                print('C_{L}>u_{H}')
            return None
        prob = self.p_ACH()
        if prob is None:
            if message:
                print('p(A|C_{H}) does not exist')
            return None
        else:

            """
            R_LB = integrate.quad(self.fun_R_LB, 0, 1)
                        if R_LB[0] > self.CA:
                if message:
                    print('R_{L}^{B}(\lambda_{L},m)<C_{A} does not hold')
                return None
            """

            R_HA = integrate.quad(self.fun_R_HA, 0, 1, args=(prob,))
            if R_HA[0] < self.CA + self.CL:
                if message:
                    print(R_HA[0])
                    print('R_{H}^{A}(\alpha^{A}(m),m)\geq C_{L}+C_{A} does not hold')
                return None
            pq_hA = 1 / (1 + prob * (1 / self.llambda - 1))
            return prob, pq_hA


def plot_fig(x, y1, y2, label1='lower bound', label2='a', xlabel='$C_{A}$', ylabel='value', title='a'):
    plt.plot(x, y1, 'k,--', alpha=0.8, label=label1)
    plt.plot(x, y2, 'k', alpha=0.8, label=label2)
    plt.legend(loc="upper right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# ==================================================
# in these parameters, the effect of lambda on
# p(A|C_{H}) and p(q_{H}|A)
# ===================================================

def sim_lambda():
    x = []
    y1 = []
    y2 = []
    llambda = np.linspace(0.1, 0.7, 100)
    CA = 1
    CL = 0.02
    uH = 1.5
    for each in llambda:
        if each*uH<=CA+CL:
            print('a')
            continue
        instance = Comparative_static(uH, each, CA, CL)
        pACH, pQHA = instance.fun_main()
        print(pACH)
        x.append(each)
        y1.append(pACH)
        y2.append(pQHA)
    return x,y1
    #plot_fig(x, y1, y2, label1='$p(A|C_{H})$', label2=r'$\alpha^{A}$', xlabel='$\lambda_{L}$', ylabel='value',
    #         title='$c_{A}=1,c_{L}=0.02,u_{H}=1.5$')


# ==================================================
# in these parameters, the effect of C_{A} on
# p(A|C_{H}) and p(q_{H}|A)
# ===================================================

def sim_CA():
    x = []
    y1 = []
    y2 = []
    llambda = 0.5
    CA = np.linspace(1, 1.3, 100)
    CL = 0.01
    uH = 2.5
    for each in CA:
        if llambda*uH<=each+CL:
            print('a')
            continue

        instance = Comparative_static(uH, llambda, each, CL)
        try:
            pACH, pQHA = instance.fun_main()
            print(['each', each])
            print(pACH)
            x.append(each)
            y1.append(pACH)
            y2.append(pQHA)
        except:
            pass
    return x,y1
    #plot_fig(x, y1, y2, label1='$p(A|C_{H})$', label2=r'$\alpha^{A}$', xlabel='$c_{A}$', ylabel='value',
    #         title='$\lambda_{L}=0.5,c_{L}=0.01,u_{H}=2.5$')


# ==================================================
# in these parameters, the effect of u_{H} on
# p(A|C_{H}) and p(q_{H}|A)
# ===================================================

def sim_uH():
    x = []
    y1 = []
    y2 = []
    llambda = 0.5
    CA = 1.2
    CL = 0.01
    uH = np.linspace(1.5, 2.61, 100)
    for each in uH:
        if llambda*each<=CA+CL:
            print('a')
            continue
        print(['**',each])
        instance = Comparative_static(each, llambda, CA, CL)
        pACH, pQHA = instance.fun_main()
        print(pACH)
        x.append(each)
        y1.append(pACH)
        y2.append(pQHA)
    return x,y1
    #plot_fig(x, y1, y2, label1='$p(A|C_{H})$', label2=r'$\alpha^{A}$', xlabel='$u_{H}$', ylabel='value',
    #         title='$\lambda_{L}=0.5,c_{L}=0.01,c_{A}=1.2$')


if __name__ == '__main__':
    plt.figure(1)

    #ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(1, 2, 1)
    ax3 = plt.subplot(1, 2, 2)

    plt.sca(ax2)
    x, y1 =sim_CA()
    label2 = '$p(A|c_{h})$'
    #label2 = r'$\alpha^{A}$'
    xlabel = '$c_{A}$'
    #ylabel = '$p(A|C_{H})$',
    title='$\lambda_{L}=0.5,c_{l}=0.01,v=2.5$'
    plt.plot(x, y1, 'k,--', alpha=0.8,label=label2)
    plt.legend(loc="upper right")

    plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    plt.title(title)

    plt.sca(ax3)
    x, y1 =sim_uH()
    label3 = '$p(Y|c_{h})$'
    xlabel = '$v$'
    #ylabel = '$p(A|C_{H})$',
    title='$\lambda_{L}=0.5,c_{l}=0.01,c_{A}=1.2$'
    plt.plot(x, y1, 'k,--', alpha=0.8,label=label3)
    plt.legend(loc="upper left")
    plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    plt.title(title)

    plt.show()

