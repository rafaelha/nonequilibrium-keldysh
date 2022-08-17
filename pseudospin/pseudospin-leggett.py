
# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
from scipy import integrate
from scipy.fftpack import fftfreq, fftshift
import scipy

# probe parameters
A0 = 0.01
w = 0.2
tau = 30  # Gaussian width
te = 187  # time delay

# pump
A0p = 0.1
wp = 0.3
taup = 4  # Gaussian width
tep = 0  # time delay

Ne = 500  # number of energy points

# time points
tmin = -40
tmax = 400
Nt = 1700

nb = 2  # number of bands
u_temp = 116.032
T = 0.001/u_temp
wd = 5
s = np.array([-1, 1])
m = np.array([0.85, 1.38])
ef = np.array([290, 70])
pre_d0 = np.array([0.3, 0.7])  # initial value of delta
v_leggett = 0.05

# Constants
hbar = 1
kb = 1
e_charge = 1

beta = 1/(kb*T)
kf = np.sqrt(2*m*ef)/hbar
vf = kf/m
n = kf**3/(3*np.pi**2)
N0 = m*kf/(2*np.pi**2)

e = np.linspace(-wd, wd, Ne)


def d0_integrand(x, d):
    # This is an auxiliary function used in find_d0 to calculate an integral
    return 0.5*1/np.sqrt(x**2+d**2)*np.tanh(beta/2*np.sqrt(x**2+d**2))


def find_d0(U):  # compute the gap given interaction matrix U
    delta_guess = np.array([1, 1])
    while True:
        integral = np.zeros(2)
        for j in [0, 1]:
            integral[j] = integrate.quad(
                d0_integrand, -wd, wd, (delta_guess[j],))[0]
        dd = U@(N0*delta_guess*integral)
        if np.sum(np.abs(dd-delta_guess)) < 1e-15:
            return dd
        delta_guess = dd


def find_U(d0, v):  # compute interaction matrix U, given d0, v
    I = np.zeros(2)
    for j in [0, 1]:
        I[j] = integrate.quad(d0_integrand, -wd, wd, (d0[j],))[0]

    U11 = d0[0]/(N0[0]*d0[0]*I[0]+v*N0[1]*d0[1]*I[1])
    U22 = (d0[1]-v*U11*N0[0]*d0[0]*I[0])/(N0[1]*I[1]*d0[1])
    U12 = v*U11
    U = np.array([[U11, U12],
                  [U12, U22]])
    return U


def nm(x):
    return x / np.max(np.abs(x))


def fft(t, f, axis=0):
    # uses angular frequencies omega
    dt = t[1]-t[0]
    yf = scipy.fft.fft(f, axis=axis)
    xf = fftfreq(len(t), dt)
    return fftshift(xf)*2*np.pi, fftshift(yf, axes=axis)


def integ(x, axis):
    """ Integrate the function 'x' over the axis 'axis'. The integration can be performed over one or two dimensions """
    if hasattr(axis, "__len__"):
        return integrate.simps(integrate.simps(x, dx=de, axis=axis[1]), dx=de, axis=axis[0])
    else:
        return integrate.simps(x, dx=de, axis=axis)


def plotA(t, A):
    plt.figure('A')
    tp = t
    plt.subplot(131)
    plt.plot(tp, A)
    plt.ylabel(f'$A(t)$')
    plt.xlabel(f'$t$')

    plt.subplot(132)
    tw, aw = fft(t, A)
    plt.plot(tw, np.abs(nm(aw)), '-')
    plt.xlim((0, 5*pre_d0[0]))
    plt.ylabel(f'$A(\omega)$')
    plt.xlabel(f'$\omega$')
    plt.axvline(2*pre_d0[0], c='gray', lw=1)
    plt.tight_layout()

    plt.subplot(133)
    tw, aw2 = fft(t, A**2)
    plt.plot(tw, np.abs(nm(aw2)), '-')
    plt.xlim((0, 5*pre_d0[0]))
    plt.ylabel(f'$A^2(\omega)$')
    plt.xlabel(f'$\omega$')
    plt.axvline(2*pre_d0[0], c='gray', lw=1)
    plt.xlim((0, 4*pre_d0[1]))
    if len(pre_d0) > 1:
        plt.axvline(2*pre_d0[1], c='gray', lw=1)
    plt.tight_layout()


def A(t):  # probe pulse
    return A0*np.exp(-(t-te)**2/(2*tau**2))*np.cos(w*(t-te))


def Ap(t):  # pump pulse
    return A0p*np.exp(-(t-tep)**2/(2*taup**2))*np.cos(wp*(t-tep))


t = np.linspace(tmin, tmax, Nt)


# %%
plt.figure('A')
plt.clf()
plotA(t, A(t))
plotA(t, Ap(t))

# %%

beta = 1/(kb*T)
ep = np.linspace(-wd, wd, Ne)
U = find_U(pre_d0, v_leggett)
UN0 = U*N0[:, np.newaxis]

ax = np.newaxis

d_eq0 = pre_d0
d_eq1 = d_eq0[:, ax]
d_eq = d_eq0[:, ax, ax]
s1 = s[:, ax]
m1 = m[:, ax]
vf1 = vf[:, ax]

e1_ = np.linspace(-wd, wd, Ne)

de = e1_[1] - e1_[0]
de2 = de**2

e1 = e1_[ax, :]
e_ = e1_[:, ax]
ep_ = e1_[ax, :]

e = e1_[ax, :, ax]
ep = e1_[ax, ax, :]

E1 = np.sqrt(e1**2 + d_eq1**2)
E = np.sqrt(e**2 + d_eq**2)
Ep = np.sqrt(ep**2 + d_eq**2)

b = np.zeros((3, nb, Ne))

# set initial pseudospin
s0 = np.zeros((3, nb, Ne))
s0[0] = d_eq1.real / 2 / E1 * np.tanh(E1 / (2 * kb * T))
s0[1] = -d_eq1.imag / 2 / E1 * np.tanh(E1 / (2 * kb * T))
s0[2] = -e1 / 2 / E1 * np.tanh(E1 / (2 * kb * T))

# set initial pseudo magnetic field
delta = U @ (N0 * integ(s0[0] - 1j * s0[1], axis=1))
b0 = np.zeros((3, nb, Ne))
b0[0] = -2*delta[:, ax].real
b0[1] = 2*delta[:, ax].imag
b0[2] = 2*e1

s0_ = np.copy(s0)
s0 = s0.reshape((3*nb*Ne,))


def ds(t, s):  # differential equation
    s_ = np.copy(s).reshape(3, nb, Ne)

    delta = U @ (N0 * integ(s_[0] - 1j * s_[1], axis=1))
    b[0] = -2*delta[:, ax].real
    b[1] = 2*delta[:, ax].imag
    b[2] = 2*e1 + s1 * (A(t) + Ap(t))**2 / (2*m1)

    ds_ = np.cross(b, s_, axisa=0, axisb=0, axisc=0).reshape((3*nb*Ne,))
    return ds_


# the built in integrator solves for the r values numerically:
sols = integrate.solve_ivp(ds, (tmin, tmax), s0, t_eval=t)

# extracting the solutions from the solver output:
Y = sols.y.reshape(3, nb, Ne, len(sols.t))
t = sols.t

# compute the gap for each time
d = np.einsum('ij,jt->ti', U, N0[:, ax]*integ(Y[0] - 1j * Y[1], axis=1))

si = s[:, ax]
mi = m[:, ax]
N0i = N0[:, ax]

# compute the density
rho = -2*np.sum(si/mi*N0i*integ(Y[2], axis=-2), axis=0)

# %% plot the results

plt.figure('delta-real')
plt.clf()
plt.subplot(121)
plt.plot(t, d.real)
dp_ = np.copy(d).real
dp_ -= np.mean(dp_, axis=0)
plt.xlabel(f'$t$')
plt.ylabel('Real$[\delta\Delta]$')
plt.subplot(122)
w_, dpw_ = fft(t, dp_)
plt.axvline(d_eq[0]*2, c='gray', lw=1)
plt.axvline(d_eq[1]*2, c='gray', lw=1)
plt.axvline(np.mean(d[:-100, 0]).real*2, c='r')
plt.axvline(np.mean(d[:-100, 1]).real*2, c='r')
plt.plot(w_, np.abs(dpw_))
plt.xlim((0, 4*d_eq[1]))
plt.xlabel(f'$\omega$')
plt.ylabel('Real$[\delta\Delta]$')
plt.tight_layout()
plt.pause(0.01)


plt.figure('Leggett')
plt.clf()
plt.subplot(121)
dp_ = np.angle(d[:, 0]) - np.angle(d[:, 1])
plt.plot(t, dp_)
dp_ -= np.mean(dp_, axis=0)
plt.xlabel(f'$t$')
plt.ylabel('$\\varphi$')
plt.subplot(122)
w_, dpw_ = fft(t, dp_)
plt.axvline(d_eq[0]*2, c='gray', lw=1)
plt.axvline(d_eq[1]*2, c='gray', lw=1)
plt.axvline(np.mean(d[:-100, 0]).real*2, c='r')
plt.axvline(np.mean(d[:-100, 1]).real*2, c='r')
plt.plot(w_, np.abs(dpw_))
plt.xlim((0, 4*d_eq[1]))
plt.xlabel(f'$\omega$')
plt.ylabel('$\\varphi$')
plt.tight_layout()
plt.pause(0.01)

plt.figure('THG')
plt.clf()
plt.subplot(121)
plt.plot(t, rho*A(t))
plt.xlabel('$t$')
plt.ylabel('$j^{(3)}$')
w_, jw_ = fft(t, rho*A(t))
plt.subplot(122)
plt.plot(w_, np.abs(jw_))
plt.xlim((0, 1))
plt.tight_layout()
plt.xlabel('$\omega$')
