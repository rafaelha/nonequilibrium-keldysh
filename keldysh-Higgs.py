# %%
import grp
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, cos, sin
from scipy import integrate
from scipy import interpolate
from scipy.fftpack import fftfreq, fftshift, ifftshift
import scipy
import pickle
import sys
import os
import gc
import time
from operator import mul
import pickle
from cycler import cycler
import matplotlib as mpl
import numpy.linalg


# define plotting style for matplotlub
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
                  '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
tableau10 = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
             '#edc948', '#b07aa1', '#ff9da7', '#9c755f']  # , '#bab0ac']
tableauCB = ['#1170aa', '#fc7d0b', '#a3acb9', '#57606c',
             '#5fa2ce', '#c85200', '#7b848f', '#a3cce9', '#ffbc79', '#c8d0d9']
mpl.rcParams['axes.prop_cycle'] = cycler(color=tableau10)
SMALL_SIZE = 11
MEDIUM_SIZE = 13
BIGGER_SIZE = 15
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def integ(x, axis):
    """ Integrate the function 'x' over the axis 'axis'. The integration can be performed over one or two dimensions """
    if hasattr(axis, "__len__"):
        if len(axis) == 2:
            return integrate.simps(integrate.simps(x, dx=de, axis=axis[0]), dx=de, axis=axis[1])
        elif len(axis) == 3:
            return integrate.simps(integrate.simps(integrate.simps(x, dx=de, axis=axis[0]), dx=de, axis=axis[1]), dx=de, axis=axis[2])
    else:
        return integrate.simps(x, dx=de, axis=axis)


def nm(x):
    return x / np.max(np.abs(x))


def fft(t, f, axis=0):
    # uses angular frequencies omega
    dt = t[1]-t[0]
    yf = scipy.fft.fft(fftshift(f, axes=axis), axis=axis) * dt  # / len(t)
    xf = fftfreq(len(t), dt)
    return fftshift(xf)*2*np.pi, fftshift(yf, axes=axis)


def ifft(t, f, axis=0):
    # uses angular frequencies omega
    dt = t[1]-t[0]
    yf = scipy.fft.ifft(fftshift(f, axes=axis), axis=axis) * len(t) * dt
    xf = fftfreq(len(t), dt)
    return fftshift(xf)*2*np.pi, fftshift(yf, axes=axis)


def s(t, tau, tpp=0):
    # envelope of probe pulse
    return 1 / (tau * np.sqrt(2 * np.pi)) * np.exp(-(t-tpp)**2/(2 * tau**2))

def checkmax(x):
    # print out max(abs()) of array. This is to check if an array is all-zero
    print('checkmax: ', np.max(np.abs(x)))

start = time.time()


tmax = 40  # bigger tmax makes eta smaller
wmax = 15
# Ne = 40
Ne = 100
N = 2*tmax*wmax / np.pi
N = int((N // 2) * 2)
eta = 1/tmax
# print('N = ', N)
print('Ne=', Ne, ', N=', N)

a0 = 4
vcoulomb = 0

tpps = [-30, -20, -10, 0, 10, 20, 30]
tpp = tpps[0]
tau = 9
tau_pump = 0.4
Omega_pump = 0.5#2
d = 1
wd = 5

e_ = np.linspace(-wd, wd, Ne)


if len(e_) > 1:
    de = e_[1] - e_[0]
    Ek_ = sqrt(e_**2 + d**2)
else:
    de = 1
    e_ = e_*0
    Ek_ = e_ + d
    


t_ = np.linspace(-tmax, tmax, N+1)[:-1]
dt = t_[1] - t_[0]
# define heaviside theta function
theta0 = 1/2
# theta(t)
thetat_ = t_*0
thetat_[t_ > 0] = 1
thetat_[t_ == 0] = theta0
# theta(-t)
theta_nt_ = t_*0
theta_nt_[t_ < 0] = 1
theta_nt_[t_ == 0] = theta0



# use the following index convention
# [e,i,t,j,t2]
# e - energy
# i,j - Pauli matrix indices
# t,t2 - time indices

# build numpy arrays to be used in python broadcasting
ax = np.newaxis
thetat = thetat_[ax, ax, :, ax, ax] # as a function of t
thetat2 = thetat_[ax, ax, ax, ax, :] # as a function of t2
t = t_[ax, ax, :, ax, ax]
t2 = t_[ax, ax, ax, ax, :]
tt = t - t2
tt2 = t_[:, ax] - t_[ax, :]
# build theta(t-t2)
thetatt = np.zeros(tt2.shape)
thetatt[tt2 == 0] = theta0
thetatt[tt2 > 0] = 1
# build theta(-t+t2)
theta_ntt = thetatt.T

thetatt = thetatt[ax, ax, :, ax, :]
theta_ntt = theta_ntt[ax, ax, :, ax, :]

Ek = Ek_[:, ax, ax, ax, ax]
e = e_[:, ax, ax, ax, ax]
Ek2 = Ek**2

# define Pauli matrices
dim = 2 # dimension of Pauli matrix
s0_ = np.array([[1, 0], [0, 1]])
s1_ = np.array([[0, 1], [1, 0]])
s2_ = np.array([[0, -1j], [1j, 0]])
s3_ = np.array([[1, 0], [0, -1]])

s0 = s0_[ax, :, ax, :, ax]
s1 = s1_[ax, :, ax, :, ax]
s2 = s2_[ax, :, ax, :, ax]
s3 = s3_[ax, :, ax, :, ax]

# define the equilibrium Green functions (retarded, advanced, lesser, greater)
# as a function of (t,t2)
# They are arrays with indices [e,i,t,j,t2]
Sin = np.sin
Cos = np.cos
Exp = np.exp
grt = -thetatt * Exp(-eta*tt) * (d*Sin(Ek*tt)/Ek * s1 +
                                 e*Sin(Ek*tt)/Ek * s3 + 1j*Cos(Ek*tt) * s0)
gat = theta_ntt * Exp(eta*tt) * (d*Sin(Ek*tt)/Ek * s1 +
                                 e*Sin(Ek*tt)/Ek * s3 + 1j*Cos(Ek*tt) * s0)
glt = -1j/2*Exp(1j*Ek*tt - eta*np.abs(tt)) * (d/Ek * s1 + e/Ek * s3 - s0)
ggt = -1j/2*Exp(-1j*Ek*tt - eta*np.abs(tt)) * (d/Ek * s1 + e/Ek * s3 + s0)


def to_matrix(tensor):
    # convert tensor with indices [e,i,t,j,t2] to tensor with indices [e,i*t,j*t2]
    Ne = tensor.shape[0]
    return tensor.reshape((Ne, dim, N, dim*N)).reshape((Ne, dim*N, dim*N))

def to_tensor(matrix):
    # convert tensor with indices [e,i*t,j*t2] to tensor with indices [e,i,t,j,t2]
    Ne = matrix.shape[0]
    return matrix.reshape((Ne, dim, N, dim*N)).reshape((Ne, dim, N, dim, N))


gr = to_matrix(grt)
ga = to_matrix(gat)
gl = to_matrix(glt)
gg = to_matrix(ggt)

# perturbation (is diagonal in time indices)
Ut = np.diag(s(t_, tau=tau_pump))[ax, ax, :, ax, :] * s3 * a0 * cos(Omega_pump*t_)
U = to_matrix(Ut)
id = np.eye(dim*N)

GRt = np.copy(grt)
GAt = np.copy(gat)
GLt = np.copy(glt)
GGt = np.copy(ggt)


#######################################################################
###### self consistent calculation of the full Green function #########
#######################################################################

it = 4  # number of self consistent iterations
if vcoulomb == 0:
    # in case of Coulomb being turned off, the is exact and does not need to be self consistent
    it = 1

for i in range(it):
    # [e,i,t,j,t2]
    # computation of Coulomb self energy (only if vcoulomb != 0)
    if vcoulomb !=0:
        GRtloc = np.sum(GRt, axis=0) / Ne
        GAtloc = np.sum(GAt, axis=0) / Ne
        GGtloc = np.sum(GGt, axis=0) / Ne
        GLtloc = np.sum(GLt, axis=0) / Ne

        def selfsum(a, b, c):
            return np.einsum('ij,jtky,kl,ty->itly', s3_, a, s3_, np.einsum('itjy,jk,kylt,li->ty', b, s3_, c, s3_))[ax]

        SGt = vcoulomb * selfsum(GGtloc, GGtloc, GLtloc)
        SLt = vcoulomb * selfsum(GLtloc, GLtloc, GGtloc)
        SRt = vcoulomb * (selfsum(GLtloc + GRtloc, GLtloc + GRtloc,
                        GLtloc + GRtloc) - selfsum(GLtloc, GLtloc, GGtloc))
        SAt = vcoulomb * (selfsum(GGtloc + GAtloc, GGtloc + GAtloc,
                        GGtloc + GAtloc) - selfsum(GGtloc, GGtloc, GLtloc))

        # SRt = vcoulomb * (selfsum(GRtloc,GGtloc,GLtloc) + selfsum(GLtloc,GLtloc,GAtloc) + selfsum(GLtloc,GRtloc,GLtloc))
        # SAt = vcoulomb * (selfsum(GAtloc,GLtloc,GGtloc) + selfsum(GGtloc,GGtloc,GRtloc) + selfsum(GGtloc,GAtloc,GGtloc))

        print(f'consistency check SF at iteration {i} of gfs: 0 == ', np.max( np.abs(SGt-SLt-SRt+SAt)))

        o = np.ones(grt.shape)
        SL = to_matrix(SLt*o)
        SG = to_matrix(SGt*o)
        SR = to_matrix(SRt*o)
        SA = to_matrix(SAt*o)
    else:
        oo = to_matrix(np.ones(grt.shape))
        SL = 0*oo
        SG = 0*oo
        SR = 0*oo
        SA = 0*oo

    # Keldysh equations
    inv = np.linalg.inv
    GR = inv(inv(gr)-dt*U-dt**2*SR)
    GA = inv(inv(ga)-dt*U-dt**2*SA)

    a = (id/dt + dt*GR@(U/dt + SR))
    b = (id/dt + dt*(U/dt + SA)@GA)
    GL = (a@gl@b + GR@SL@GA) * dt**2
    GG = (a@gg@b + GR@SG@GA) * dt**2

    GLt = to_tensor(GL)
    GGt = to_tensor(GG)
    GRt = to_tensor(GR)
    GAt = to_tensor(GA)

    # symmetrize (make GL, GG, GR-GA anti hermitian and GR+GA hermitian)
    # GLt = 0.5 * (GLt - np.transpose(GLt,axes=(0,3,4,1,2)).conj())
    # GGt = 0.5 * (GGt - np.transpose(GGt,axes=(0,3,4,1,2)).conj())
    # GRplusAt = GRt + GAt
    # GRminusAt = GRt - GAt
    # GRplusAt = 0.5 * (GRplusAt + np.transpose(GRplusAt,axes=(0,3,4,1,2)).conj())
    # GRminusAt = 0.5 * (GRminusAt - np.transpose(GRminusAt,axes=(0,3,4,1,2)).conj())
    # GRt = 0.5 * (GRplusAt + GRminusAt)
    # GAt = 0.5 * (GRplusAt - GRminusAt)

    # consistenct check
    consist = np.abs(GGt-GLt-GRt+GAt)
    # plt.figure()
    # plt.pcolormesh(t_,t_,np.sum(consist, axis=(0,1,3)))
    # plt.pause(0.01)
    print(f'consistency check GF at iteration {i}: 0 == ', np.max(consist))


# The Green's functions have been computed. Now we will plot some quantities

#%%
#################### Plot bloch vectors ######################
b1 = 1j * np.einsum('ij,ejtip->tp', s1_, GLt) / Ne
b2 = 1j * np.einsum('ij,ejtip->tp', s2_, GLt) / Ne
b3 = 1j * np.einsum('ij,ejtip->tp', s3_, GLt) / Ne
# b1 = np.einsum('ij,ejtip->tp',s1_,GRt) / Ne
# b2 = np.einsum('ij,ejtip->tp',s2_,GRt) / Ne
# b3 = np.einsum('ij,ejtip->tp',s3_,GRt) / Ne

c1 = np.diag(b1)
c2 = np.diag(b2)
c3 = np.diag(b3)

plt.figure('bloch', figsize=(10, 7.5))
plt.clf()
plt.subplot(221)
plt.plot(t_, c2.real, label='y')
plt.plot(t_, c3.real, label='z')
plt.plot(t_, c1.real, label='x')

spump = s(t_, tau=tau_pump)
sprobe = s(t_, tau=tau, tpp=tpp)
plt.plot(t_, nm(spump), c='k', label='pump')
plt.plot(t_, nm(sprobe), c='k', label='pump')
plt.xlim((-6, 6))
# if Ne == 1:
# plt.plot(t_,np.sqrt(c1**2 + c2**2 + c3**2), label='magnitude')

plt.subplot(223)
plt.plot(t_, c2.imag, label='y')
plt.plot(t_, c3.imag, label='z')
plt.plot(t_, c1.imag, label='x')
plt.legend()

plt.subplot(222)
wpulse, swpump = fft(t_, spump)
wpulse, swprobe = fft(t_, sprobe)
plt.plot(wpulse, np.abs(swpump), label='pump')
plt.plot(wpulse, np.abs(swprobe), label='probe')
plt.legend()
plt.axvline(2*d, c='gray', alpha=0.8, lw=0.7)
plt.xlabel('$\omega$')
plt.tight_layout()

checkmax(c1.imag)
checkmax(c2.imag)
checkmax(c3.imag)

plt.subplot(224)
plt.plot(t_, np.imag(np.diagonal(SRt[0, 0, :, 0])))
plt.plot(t_, np.imag(np.diagonal(SRt[0, 1, :, 1])))
plt.plot(t_, np.imag(np.diagonal(SRt[0, 1, :, 0])))
plt.plot(t_, np.imag(np.diagonal(SRt[0, 0, :, 1])))
# plt.axhline(-vcoulomb, c='gray', lw=0.4)


# %%
#################### Plot Raman and THG signal ######################
#lesser response function?
r_ = de*np.einsum('ij,ejtkw,kl,elwit->wt', s3_,
                  to_tensor(GG), s3_, to_tensor(GL))
#Retarded quantity?
thg_ = 1j*de*np.einsum('ij,ejtkw,kl,elwit->wt', s3_, to_tensor(GR), s3_, to_tensor(GL)) + \
    1j*de*np.einsum('ij,ejtkw,kl,elwit->wt', s3_,
                    to_tensor(GL), s3_, to_tensor(GA))

print('real-thg: ', np.max(np.abs(thg_.real)))
print('imag-thg: ', np.max(np.abs(thg_.imag)))


def A(t, tau, Omega, tpp=0):
    return 1 / (tau * np.sqrt(2 * np.pi)) * np.exp(-(t-tpp)**2/(2 * tau**2)) * np.cos(Omega * (t-tpp))
    # return 1/ ( tau * np.sqrt(2 * np.pi) ) * np.exp(-(t-tpp)**2/(2 * tau**2)) * np.exp(1j * Omega * (t-tpp))


res_raman = []
res_thg = []
res_thg_t = []

Omega = 1.05

tpps = [-30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30]
for tpp in tpps:
    st_ = s(t_, tau=tau, tpp=tpp)
    st = st_[:, ax]
    st2 = st_[ax, :]

    at_ = A(t_, tau=tau, tpp=tpp, Omega=Omega)
    at = at_[:, ax]
    at2 = at_[ax, :]

    r = st*st2*r_
    # thg = np.sum((at*at2**2 + np.conj(at*at2**2))*thg_, axis=1)
    thg = np.sum(at*at2**2*thg_, axis=1)
    w, thgw = fft(t_, thg, axis=0)

    w, rw_ = fft(t_, r, axis=0)
    w, rww = ifft(t_, rw_, axis=1)

    raman = np.diag(rww)
    res_raman.append(raman)
    res_thg.append(thgw)
    res_thg_t.append(thg)

rmax = np.max(np.array(res_raman).real)
thgmax = np.max(np.abs(np.array(res_thg[0])))
thgtmax = np.max(np.abs(np.array(res_thg_t[0])))

# plot electrical field
plt.figure('at')
plt.clf()
plt.plot(t_, at_)


plt.figure('raman', figsize=(14, 19))
plt.clf()
plt.subplot(131)
offset = rmax/4
for i, raman in enumerate(res_raman[::-1]):
    plt.plot(w, raman.real - i*offset, label=f'tpp={tpps[::-1][i]}')
    plt.plot(w, raman.imag - i*offset, 'k--', lw=0.4)
plt.xlabel('$\omega/\Delta$')
plt.ylabel('Raman signal')
plt.title('Raman')
plt.legend()
plt.xlim((-5,5))
plt.axvline(0, c='gray', lw=0.4)
# plt.pause(0.01)

plt.subplot(132)
offset = thgtmax*1.2
for i, thgt in enumerate(res_thg_t[::-1]):
    tpp_ = tpps[::-1][i]
    o = 1
    plt.plot(t_-tpp_, s(t_, tau=tau_pump)-2*i*o, 'k', lw=1)
    plt.plot(t_-tpp_, thgt.real/thgtmax - 2*i*o, label=f'tpp={tpps[::-1][i]}')
    plt.plot(t_-tpp_, thgt.imag/thgmax - 2*i*o, 'k--', lw=0.4)
plt.xlabel('$t$')
plt.ylabel('$j^{(3)}(t)$')
plt.title('THG (time resolved)')
plt.axvline(0, c='gray', lw=0.4)
plt.xlim((-tmax, tmax))
plt.legend()

plt.subplot(133)
offset = thgmax
for i, thgw in enumerate(res_thg[::-1]):
    plt.plot(w/Omega, np.abs(thgw)/thgmax - 2*i, label=f'tpp={tpps[::-1][i]}')
    plt.plot(w, thgt.imag*0 - 2*i, 'k--', lw=0.4)
plt.xlabel('$\omega/\Omega$')
plt.ylabel('$j^{(3)}(\omega)$')
plt.title('THG (frequency domain)')
# plt.legend()
plt.xlim((0, 5))
# plt.pause(0.01)
plt.axvline(1, c='gray', lw=0.4)
plt.axvline(3, c='gray', lw=0.4)
plt.tight_layout()
plt.savefig('THG-pumped.pdf')



# %%
#################### Plot 2D spectrum ######################

t__ = t_[:, ax]
t2__ = t_[ax, :]
t2__ = t2__[t2__ > -3*tau]
t2__ = t2__[t2__ < 3*tau]


at__ = A(t__, tau=tau, tpp=t2__, Omega=Omega)

at = at__[:, ax, :]
at2 = at__[ax, :, :]

thg = np.sum(at*at2**2*thg_[:, :, ax], axis=1)

thg_shifted = np.zeros(thg.shape, dtype=complex)

for i in range(thg.shape[1]):
    thg_shifted[:, i] = np.roll(thg[:, i], -int(t2__[i]/dt))

# plt.pcolormesh(t2__,t__,thg.real)
plt.figure('2dt')
plt.pcolormesh(t2__, t__, thg_shifted.real, shading='auto')
plt.xlabel('$\Delta t$')
plt.ylabel('$t$')
plt.ylim((-4*tau, 4*tau))

figsize = (3.6, 2.5)
plt.figure('max', figsize=figsize)
maxi = np.argmax(thg_shifted[:, 0])
mini = np.argmin(thg_shifted[:, 0])
plt.plot(t2__, 1000*(thg_shifted[maxi] - thg_shifted[maxi][0]))
plt.plot(t2__, 1000*(thg_shifted[mini] - thg_shifted[mini][0]))
# plt.plot(t2__, np.max(thg_shifted, axis=0))
# plt.plot(t2__, np.min(thg_shifted, axis=0))
plt.xlabel('$\Delta t$')
plt.ylabel('$\delta E_{TH}$ (a.u.)')
plt.xlim((-13, 30))
plt.tight_layout()
plt.savefig('coh.png', dpi=400)

w, temp = fft(t__, thg_shifted, axis=0)
wpp, thg2d = fft(t2__, temp, axis=1)

plt.figure('2dw')
pc = np.abs(thg2d)
plt.pcolormesh(wpp, w, pc, cmap='inferno', vmax=np.max(pc)/10, shading='auto')
plt.axis('equal')
plt.xlim((-4, 4))
plt.ylim((-4, 4))
plt.xlabel('$\omega_{\Delta t}$')
plt.ylabel('$\omega$')


# %%
#################### Plot density of state ######################
# [e,i,t,j,t2]

u = np.sqrt(0.5*(1+e_/Ek_))
v = -np.sqrt(0.5*(1-e_/Ek_))
U = np.zeros((Ne, 2, 2))
U[:, 0, 0] = -u
U[:, 0, 1] = v
U[:, 1, 0] = v
U[:, 1, 1] = u

# DOS = -1j*U@to_tensor(GL)[:,:,ti,:,ti]@U
DOS = -1j * np.einsum('eij,ejtkt,ekl->etil', U, to_tensor(GL), U)
DOS0 = -np.trace(np.imag(to_tensor(GR)[:, :, 0, :, 0]), axis1=1, axis2=2)


# DOS = -1j*np.diagonal(, axis1=1,axis2=2)
# DOS += -1j*np.diagonal(to_tensor(GL)[:,1,:,1,:], axis1=1,axis2=2)
# DOS = -np.imag(np.diagonal(to_tensor(GR)[:,0,:,0,:], axis1=1,axis2=2))
en_ = np.copy(Ek_)


# [w,e,t]
def broaden(w, pos, dos, eta):
    # ww = w_[:,ax,ax]
    # pos2 = en_[ax,:,ax]
    occupation = np.real(np.sum(
        eta/((w_[:, ax, ax]-pos[ax, :, ax])**2 + eta**2) * dos[ax, :, :], axis=1)) / Ne
    # + np.sum(eta2/((ww+pos)**2+ eta2**2) * DOS[:,:,1,1][ax,:,:], axis=1))

    return occupation


w_ = np.linspace(-wd, wd, 1000)
eta2 = 0.08
occupation = broaden(w_, en_, DOS[:, :, 0, 0], eta2) + \
    broaden(w_, -en_, DOS[:, :, 1, 1], eta2)
occupation0 = broaden(w_, en_, DOS0[:, ax], eta2) + \
    broaden(w_, -en_, DOS0[:, ax], eta2)

cmap = matplotlib.cm.get_cmap('coolwarm')

figsize = (3.6, 2.5*2)
plt.figure('fermi-distr', figsize=figsize)
plt.clf()

plt.subplot(211, label='a')
times = [0, N//4, N//2, 3*N//4, -1]
times = np.arange(0, len(t_), N//10)
plt.fill_between(w_, occupation0[:, 0], alpha=0.4)
# plt.plot(w_,occupation0[:,0], c='k')
# for i, ti in enumerate(times):
for i, ti in enumerate(times[-5:-4]):
    plt.plot(w_, occupation[:, ti],
             label=f'{np.round(t_[ti],1)}', c=cmap(i/len(times)))
    plt.fill_between(w_, occupation[:, ti], alpha=0.5, color='blue')
    plt.xlim((-3.3, 3.3))
    plt.ylim((0, 2.3))
    plt.ylabel('State occupation')

plt.xlabel('$\omega/\Delta$')
plt.legend()

plt.subplot(212, label='2')
plt.axvspan(-1, 1, fc='gray', alpha=0.4)


for i, ti in enumerate(times):
    distribution = occupation[:, ti]/occupation0[:, 0]
    distribution = distribution[np.abs(w_) > 1]
    w__ = w_[np.abs(w_) > 1]
    plt.plot(w__[w__ < 0], distribution[w__ < 0], c=cmap(i/len(times)))
    plt.plot(w__[w__ > 0], distribution[w__ > 0],
             c=cmap(i/len(times)), label=f'${ti}$')
    plt.axhline(1, c='gray', lw=0.4)
    plt.xlabel('$\omega/\Delta$')
    plt.xlim((-4, 4))
    plt.ylabel('$n_F$')
    plt.title(ti)

    if i == len(times)-1:
        from scipy.optimize import curve_fit

        def fermi(e, T):
            return 1/(np.exp(e/T)+1)
        popt, pcov = curve_fit(fermi, w__, distribution,
                               p0=[0.05], bounds=(0.05, 10))
        Tfit = popt[0]
        plt.plot(w_, fermi(w_, *popt), '--', lw=0.5, c='g',
                 label=f'Fermi T={np.round(Tfit,2)}')
        # plt.legend()

plt.tight_layout()
plt.savefig('occupation.png', dpi=400)

plt.figure('particle-num')
plt.plot(t_, np.sum(occupation, axis=0), '.')
# plt.plot(t_, s(t_,tau=tau_pump))



# %%
#################### Plot electri field ######################
spump = s(t_, tau=tau_pump)
sprobe = A(t__, tau=tau, Omega=Omega)
plt.figure('spectra pump probe')
plt.subplot(121)
plt.plot(t_, nm(sprobe), label='probe')
plt.plot(t_, nm(spump), label='pump')
plt.xlabel('$t$')
plt.ylabel('$A(t)$')
plt.legend()

plt.subplot(122)
wpulse, swpump = fft(t_, spump)
wpulse, swprobe = fft(t_, sprobe)
plt.plot(wpulse, np.abs(swprobe), label='probe')
plt.plot(wpulse, np.abs(swpump), label='pump')
plt.xlim((0, 6))
plt.legend()
plt.xlabel('$\omega$')
plt.ylabel('$|A(\omega)|$')
plt.tight_layout()
plt.savefig('pulse shapes.pdf')


#%%
print(np.round(time.time()-start, 2), 's elasped')
# %%
# By Sida Tian
# Compute the response of the Higgs propagator
# Both Bare Higgs and Full Higgs are included

chi_R = 1j*de*np.einsum('ij,ejtkw,kl,elwit->tw', s1_,
                  to_tensor(GR), s1_, to_tensor(GL),optimize=True) +\
        1j*de*np.einsum('ij,ejtkw,kl,elwit->tw', s1_,
                  to_tensor(GL), s1_, to_tensor(GA),optimize=True)
        

bchi_R = 1j*de*np.einsum('ij,ejtkw,kl,elwit->tw', s1_,
                  to_tensor(gr), s1_, to_tensor(gl),optimize=True) +\
        1j*de*np.einsum('ij,ejtkw,kl,elwit->tw', s1_,
                  to_tensor(gl), s1_, to_tensor(ga),optimize=True)
        


U_binding = 1/np.sum(1/Ek_) 

def A(t, tau, Omega, tpp=0):
    return 1 / (tau * np.sqrt(2 * np.pi)) * np.exp(-(t-tpp)**2/(2 * tau**2)) * np.cos(Omega * (t-tpp))
    # return 1/ ( tau * np.sqrt(2 * np.pi) ) * np.exp(-(t-tpp)**2/(2 * tau**2)) * np.exp(1j * Omega * (t-tpp))


kron_time = np.eye(N)
H0_inv = kron_time/U_binding #Local in time, hence neither retarded nor advanced
#Note that H0 has no greater or lesser component

HR = inv(H0_inv - chi_R)


HR_bare = inv(H0_inv - bchi_R)
#############################
#Quasi particle
thg_ = 1j*de*np.einsum('ij,ejtkw,kl,elwit->wt', s3_, to_tensor(GR), s3_, to_tensor(GL)) + \
    1j*de*np.einsum('ij,ejtkw,kl,elwit->wt', s3_,
                    to_tensor(GL), s3_, to_tensor(GA))
    
Bthg_ = 1j*de*np.einsum('ij,ejtkw,kl,elwit->wt', s3_, to_tensor(gr), s3_, to_tensor(gl)) + \
    1j*de*np.einsum('ij,ejtkw,kl,elwit->wt', s3_,
                    to_tensor(gl), s3_, to_tensor(ga))
#############################
THG_R = 1j*de*np.einsum('ij,ejtkw,kl,elwit->tw', s1_,
                  to_tensor(GR), s3_, to_tensor(GL),optimize=True) +\
        1j*de*np.einsum('ij,ejtkw,kl,elwit->tw', s1_,
                  to_tensor(GL), s3_, to_tensor(GA),optimize=True)
        
        
THG_RT = 1j*de*np.einsum('ij,ejtkw,kl,elwit->tw', s3_,
                  to_tensor(GR), s1_, to_tensor(GL),optimize=True) +\
        1j*de*np.einsum('ij,ejtkw,kl,elwit->tw', s3_,
                  to_tensor(GL), s1_, to_tensor(GA),optimize=True)
        
        
THGR =  THG_R@HR@THG_RT
THGR_bare =  THG_R@HR_bare@THG_RT

# %%
#Tpp resolved figure
res_thg = []
res_thg_t = []
res_thg_quasi = []
res_thg_quasi_t = []
scale_ = 5

Omega = 1.4

tpps = [ -10, -7, -5, -2, -1,0, 1,2, 5, 7, 10, 20, 30]
for tpp in tpps:
    at_ = A(t_, tau=tau, tpp=tpp, Omega=Omega)
    at = at_[:, ax]
    at2 = at_[ax, :]

    r = st*st2*r_

    thg_quasi = np.sum(at*at2**2*thg_, axis=1)
    w, thgw_ = fft(t_, thg_quasi, axis=0)
    
    thg = np.sum(at*at2**2*THGR, axis=1)
    w, thgw = fft(t_, thg, axis=0)

    
    res_thg.append(thgw)
    res_thg_t.append(thg)
    res_thg_quasi.append(thgw_)
    res_thg_quasi_t.append(thg_quasi)
    
plt.figure('THG', figsize=(8, 22))
#thgmax = 4*np.max(np.abs(np.array(res_thg[3])))
#thgtmax = 4*np.max(np.abs(np.array(res_thg_t[3])))
offset = thgmax

plt.subplot(121)
for i, thgw in enumerate(res_thg_quasi_t[::-1]):
    plt.plot(w/Omega, (thgw)/thgmax - 2*i, label=f'tpp={tpps[::-1][i]}')
    plt.plot(w, thgw.imag*0 - 2*i, 'k--', lw=0.4)
plt.legend()
#plt.xlabel('$\omega/\Omega$')
plt.xlabel('$t$')
plt.ylabel('$j^{(3)}(\omega)$')
plt.title(r'$J^{(3)}_{quasi}(\omega)$')
# plt.legend()
plt.xlim((-3, 10))
#plt.xlim((0, 5))
# plt.pause(0.01)
#plt.axvline(1, c='gray', lw=0.4)
#plt.axvline(3, c='gray', lw=0.4)
#plt.axvline(2+d/Omega,c='b', lw=0.4)
#plt.axvline(2-d/Omega,c='b', lw=0.4)
#plt.axvline(d/Omega,c='red', lw=0.4)

#plt.axvline(2+2*d/Omega,c='red', lw=0.8)
#plt.axvline(2-2*d/Omega,c='red', lw=0.8)

plt.tight_layout()
#plt.savefig('THG-pumped.pdf')
########################################
plt.subplot(122)
for i, thgw in enumerate(res_thg_t[::-1]):
    plt.plot(w/Omega, scale_*(thgw)/thgmax - 2*i, label=f'tpp={tpps[::-1][i]}')
    plt.plot(w, thgw.imag*0 - 2*i, 'k--', lw=0.4)
#plt.legend()
#plt.xlabel('$\omega/\Omega$')
plt.xlabel('$t$')
plt.ylabel('$j^{(3)}(\omega)$')
plt.title(r'$J^{(3)}_{H}(\omega) \times 5$')
# plt.legend()
plt.xlim((-5, 10))
# plt.pause(0.01)
#plt.axvline(1, c='gray', lw=0.4)
#plt.axvline(3, c='gray', lw=0.4)
#plt.axvline(2+d/Omega,c='blue', lw=0.8)
#plt.axvline(2-d/Omega,c='blue', lw=0.8)

#plt.axvline(d/Omega,c='red', lw=0.4)

#plt.axvline(2+2*d/Omega,c='red', lw=0.8)
#plt.axvline(2-2*d/Omega,c='red', lw=0.8)
plt.tight_layout()
#plt.savefig('THG-pumped.pdf')
plt.savefig('THG_time_domain.pdf')


# %%
#Omega resolved figure
res_thg = []
res_thg_t = []
res_thg_quasi = []


#tpps = [ -10, -7, -5, -2, -1,0, 1,2, 5, 7, 10, 20, 30]
omega_list =[0.4, 0.5, 0.6, 0.7, 0.8,
             0.9, 1, 1.1, 1.2, 1.3, 
             1.4, 1.5, 1.6]

tpp = 5
for Omega in omega_list:
    at_ = A(t_, tau=tau, tpp=tpp, Omega=Omega)
    at = at_[:, ax]
    at2 = at_[ax, :]

    r = st*st2*r_

    thg_quasi = np.sum(at*at2**2*thg_, axis=1)
    w, thgw_ = fft(t_, thg_quasi, axis=0)
    
    thg = np.sum(at*at2**2*THGR, axis=1)
    w, thgw = fft(t_, thg, axis=0)

    
    res_thg.append(thgw)
    res_thg_t.append(thg)
    res_thg_quasi.append(thgw_)
    
    
plt.figure('THG', figsize=(8, 22))
#thgmax = np.max(np.abs(np.array(res_thg[0])))
#thgmax = np.max(np.abs(np.array(res_thg[0])))
#thgtmax = np.max(np.abs(np.array(res_thg_t[0])))
offset = thgmax

plt.subplot(121)
for i, thgw in enumerate(res_thg_quasi[::-1]):
    plt.plot(w/omega_list[-i-1], np.abs(thgw)/thgmax - 2*i, label=f'$\omega$={omega_list[::-1][i]}')
    plt.plot(w, thgw.imag*0 - 2*i, 'k--', lw=0.4)
plt.legend()
plt.xlabel('$\omega/\Omega$')
plt.ylabel('$j^{(3)}(\omega)$')
plt.title(r'$J^{(3)}_{quasi}(\omega)$')
# plt.legend()
plt.xlim((0, 5))
# plt.pause(0.01)
plt.axvline(1, c='gray', lw=0.4)
plt.axvline(3, c='gray', lw=0.4)
plt.axvline(Omega_pump/Omega,c='blue', lw=0.4)
plt.axvline(3*Omega_pump/Omega,c='blue', lw=0.4)
plt.tight_layout()
#plt.savefig('THG-pumped.pdf')
########################################
plt.subplot(122)
for i, thgw in enumerate(res_thg[::-1]):
    plt.plot(w/omega_list[-i-1], 5*np.abs(thgw)/thgmax - 2*i, label=f'tpp={omega_list[::-1][i]}')
    plt.plot(w, thgw.imag*0 - 2*i, 'k--', lw=0.4)
    plt.axvline(2+d/Omega,c='blue', lw=0.8)
    plt.axvline(2-d/Omega,c='blue', lw=0.8)
#plt.legend()
plt.xlabel('$\omega/\Omega$')
plt.ylabel('$j^{(3)}(\omega)$')
plt.title(r'$J^{(3)}_{H}(\omega) \times 5$')
# plt.legend()
plt.xlim((0, 5))
# plt.pause(0.01)
plt.axvline(1, c='gray', lw=0.4)
plt.axvline(3, c='gray', lw=0.4)



#plt.axvline(Omega_pump/Omega,c='blue', lw=0.4)
#plt.axvline(3*Omega_pump/Omega,c='blue', lw=0.4)
plt.tight_layout()
#plt.savefig('THG-pumped_frequency_tpp5.pdf')
#plt.savefig('THG_omega_1_0.pdf')

# %%

#plt.xlabel('')
plt.ylabel('Occupation')


#Omega resolved figure
res_thg = []
res_thg_t = []
res_thg_quasi = []

Omega = 1

#tpps = [ -10, -7, -5, -2, -1,0, 1,2, 5, 7, 10, 20, 30]
omega_list =[0.4, 0.5, 0.6, 0.7, 0.8,
             0.9, 1, 1.1, 1.2, 1.3, 
             1.4, 1.5, 1.6]

tpp = 5
for Omega in omega_list:
    at_ = A(t_, tau=tau, tpp=tpp, Omega=Omega)
    at = at_[:, ax]
    at2 = at_[ax, :]

    r = st*st2*r_

    thg_quasi = np.sum(at*at2**2*thg_, axis=1)
    w, thgw_ = fft(t_, thg_quasi, axis=0)
    
    thg = np.sum(at*at2**2*THGR, axis=1)
    w, thgw = fft(t_, thg, axis=0)

    
    res_thg.append(thgw)
    res_thg_t.append(thg)
    res_thg_quasi.append(thgw_)
    
    
plt.figure('THG', figsize=(8, 22))
#thgmax = np.max(np.abs(np.array(res_thg[0])))
thgmax = np.max(np.abs(np.array(res_thg[0])))
thgtmax = np.max(np.abs(np.array(res_thg_t[0])))
offset = thgmax

plt.subplot(121)
for i, thgw in enumerate(res_thg_quasi[::-1]):
    plt.plot(w/omega_list[-i-1], np.abs(thgw)/thgmax - 2*i, label=f'$\omega$={omega_list[::-1][i]}')
    plt.plot(w, thgw.imag*0 - 2*i, 'k--', lw=0.4)
plt.legend()
plt.xlabel('$\omega/\Omega$')
plt.ylabel('$j^{(3)}(\omega)$')
plt.title(r'$J^{(3)}_{quasi}(\omega)$')
# plt.legend()
plt.xlim((0, 5))
# plt.pause(0.01)
plt.axvline(1, c='gray', lw=0.4)
plt.axvline(3, c='gray', lw=0.4)
plt.axvline(Omega_pump/Omega,c='blue', lw=0.4)
plt.axvline(3*Omega_pump/Omega,c='blue', lw=0.4)
plt.tight_layout()
#plt.savefig('THG-pumped.pdf')
########################################
plt.subplot(122)
for i, thgw in enumerate(res_thg[::-1]):
    plt.plot(w/omega_list[-i-1], 5*np.abs(thgw)/thgmax - 2*i, label=f'tpp={omega_list[::-1][i]}')
    plt.plot(w, thgw.imag*0 - 2*i, 'k--', lw=0.4)
    plt.axvline(2+d/Omega,c='blue', lw=0.4)
    plt.axvline(2-d/Omega,c='blue', lw=0.4)
#plt.legend()
plt.xlabel('$\omega/\Omega$')
plt.ylabel('$j^{(3)}(\omega)$')
plt.title(r'$J^{(3)}_{H}(\omega) \times 5$')
# plt.legend()
plt.xlim((0, 5))
# plt.pause(0.01)
plt.axvline(1, c='gray', lw=0.4)
plt.axvline(3, c='gray', lw=0.4)



#plt.axvline(Omega_pump/Omega,c='blue', lw=0.4)
#plt.axvline(3*Omega_pump/Omega,c='blue', lw=0.4)
plt.tight_layout()
#plt.savefig('THG-pumped_frequency_tpp5.pdf')
#plt.savefig('THG_omega_1_0.pdf')
# %%
