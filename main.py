from numpy import inf
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def get_initial_values(k):
    result = []
    for i in range(k+1):
        result.append((2*i+1)/2)
    return result


def a_n_set(t, y, k, sigm):
    result = []
    for j in range(k+1):
        result.append(0)
    result[0] = 0.5
    for i in range(1, k+1):
        result[i] = -0.5*y[i]*i*(i+1) + (sigm*y[i]*i*(i+1))/((2*i-1)*(2*i+3))
        if i-2 > 0:
            result[i] += (sigm*y[i-2]*(i-1)*i*(i+1))/((2*i-1)*(2*i+1))
        elif i-2 == 0:
            result[i] += (sigm*0.5*(i-1)*i*(i+1))/((2*i-1)*(2*i+1))
        if i+2 <= k:
            result[i] -= (sigm*y[i+2]*(i+2)*i*(i+1))/((2*i+1)*(2*i+3))
    return result


def Q1(a1):
    np_a = np.array(a1)
    np_a = np_a*(2/6)
    return np_a


def error_is_small(y1, y2):
    for i in range(len(y1)):
        v1 = np.array(y1[i])
        v2 = np.array(y2[i])
        err = np.linalg.norm(v2 - v1)
        if err == inf:
            raise ValueError
        if err/np.linalg.norm(v1) > 10**-5:
            return False
    return True


t_eval = [0]
for i in range(320):
    t_eval.append(t_eval[i] + 0.5)
t_span = [0, 160]


def find_appropriate_k_and_count_moment(sigm):
    err_small = False
    k = 1
    while not err_small:
        y_0 = get_initial_values(k)
        sol_1 = solve_ivp(a_n_set, t_span, y_0, 'RK45', t_eval, args=(k, sigm))

        k += 1
        y_0 = get_initial_values(k)
        sol_2 = solve_ivp(a_n_set, t_span, y_0, 'RK45', t_eval, args=(k, sigm))

        try:
            err_small = error_is_small(sol_1.y, sol_2.y)
            k += 1
        except ValueError:
            k += sigm
    print(f'sigma: ' + str(sigm) + ' k: ' + str(k-1))
    q1 = Q1(sol_2.y[1])
    plt.plot(t_eval, q1, label='sigma: ' + str(sigm) + ' k: ' + str(k-1))


def main():
    sigma_array = [0.05, 0.5, 1, 3, 5, 7]
    for sigm in sigma_array:
        find_appropriate_k_and_count_moment(sigm)
    plt.title("Зависимость суммарного магнитного момента частиц от времени")
    plt.xlabel("t'")
    plt.ylabel("Q1")
    plt.legend()
    plt.show()


main()
