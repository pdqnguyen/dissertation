# PMVP waveform generator
# van Putten-based model for SN GW radiation
# theta: colatitude
# phi: azimuth

#!/usr/bin/python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./custom-style.mplstyle')

DATA_DIR = os.path.join(os.path.dirname(__file__), '../scripts/data/')
OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/grb/')


# constants
msun = 1.98892e33
clite = 2.9979e10
ggrav = 6.6726e-8
factor = 1.0 * ggrav / clite**4
pi = 3.14159265358979e0
trd = 1./3.

# R_isco of a Kerr BH (Bardeen et al 72).
# Take direct (not retrograde) orbits
def Z1(J,M):
    z1 = 1. + ((1-J*clite/ggrav/M**2.)**trd + (1+J*clite/ggrav/M**2.)**trd) * (1-J**2.*clite**2./ggrav**2./M**4.)**trd
    return z1

def Z2(J,M):
    z1 = Z1(J,M)
    z2 = np.sqrt(z1**2. + 3.*(J*clite/ggrav/M**2)**2)
    return z2

def Risco(J,M):
    z1 = Z1(J,M)
    z2 = Z2(J,M)
    # Return Risco in cgs units
    risco = M*(3. + z2 - np.sqrt((3.-z1)*(3.+z1+2.*z2))) * ggrav/clite**2
    return risco

# Derivatives of above quantities (check Risco.nb)
def Z1dot(J,M,Jdot,Mdot):
    jm2 = clite/ggrav * J/M**2
    num = clite * (M*Jdot - 2.*J*Mdot)
    f1 = 3.*clite**2. * J**2. * ( (1+jm2)**(2./3.) - ((1-jm2)**(2./3.)) )
    f2 = -2.* clite * ggrav * J * M**2. * ( (1+jm2)**(2./3.) + ((1-jm2)**(2./3.)) )
    f3 = ggrav**2. * M**4. * ( (1-jm2)**(2./3.) - ((1+jm2)**(2./3.)) )
    den = 3*ggrav**3. * M**7. * (1 - jm2**2.)**(4./3.)
    z1dot = (f1 + f2 + f3) * num/den
    return z1dot

def Z2dot(J,M,Jdot,Mdot):
    jm2 = clite/ggrav * J/M**2
    num = 3.*clite**2.*J*M*Jdot - 6.*clite**2.*J**2 * Mdot + ggrav**2.*M**5 * Z1(J,M)*Z1dot(J,M,Jdot,Mdot)
    den = ggrav**2.*M**5 * np.sqrt(3.*jm2**2. + Z1(J,M)**2)
    z2dot = num / den
    return z2dot

def Riscodot(J,M,Jdot,Mdot):
    f1  = Mdot * (3. + Z2(J,M) - np.sqrt((3. - Z1(J,M)) * (3. + Z1(J,M) + 2.*Z2(J,M))))
    num = (Z1(J,M) + Z2(J,M))*Z1dot(J,M,Jdot,Mdot) - (3.-Z1(J,M))*Z2dot(J,M,Jdot,Mdot)
    den = np.sqrt((3. - Z1(J,M)) * (3. + Z1(J,M) + 2.*Z2(J,M)))
    riscodot = f1 + M * (Z2dot(J,M,Jdot,Mdot) + num/den)
    # Return Riscodot in cgs units
    riscodot *= ggrav/clite**2
    return riscodot

# Angular velocity courtesy of Kepler.
def Omega(M,dist):
    omega = np.sqrt(ggrav * M / dist**3.)
    return omega

# Loss of angular momentum and mass = P_GW/c^2
def Jdot(m,M,dist):
    jdot = -128./5. * m**2. * dist**4. * (Omega(M,dist))**5. * ggrav / clite**5.
    return jdot

def Mdot(m,M,dist):
    mdot = -128./5. * m**2. * dist**4. * (Omega(M,dist))**6. * ggrav / clite**7.
    return mdot

# Functions for the RK4 integration scheme
# We evolve Risco, J, M simultaneously
def diff(t,dat):
    r = dat[0]
    jbh = dat[1]
    mbh = dat[2]

    jdot = Jdot(mch, mbh, rct + r)
    mdot = Mdot(mch, mbh, rct + r)
    rdot = Riscodot(jbh, mbh, jdot, mdot)

    ret = np.array([rdot, jdot, mdot])
    return ret

def rk4(old,t,dt):
    k1 = diff(t,old)
    k2 = diff(t + 0.5*dt, old + 0.5*dt*k1)
    k3 = diff(t + 0.5*dt, old + 0.5*dt*k2)
    k4 = diff(t + dt, old + dt*k3)

    new = old + dt * (k1 + 2.*k2 + 2.*k3 + k4)/6.

    return new

# Second derivative of the reduced mass quadrupole moment
def Idotdot(m,dist,omega,time):
    fact = 4. * m * dist**2. * omega**2.
    Idotdot = np.zeros((3,3),float)
    Idotdot[0,0] = - np.cos(2.0*omega*time)
    Idotdot[0,1] = - np.sin(2.0*omega*time)
    Idotdot[0,2] = 0.0
    Idotdot[1,0] = - np.sin(2.0*omega*time)
    Idotdot[1,1] =   np.cos(2.0*omega*time)
    Idotdot[1,2] = 0.0
    Idotdot[2,0] = 0.0
    Idotdot[2,1] = 0.0
    Idotdot[2,2] = 0.0
    Idotdot = Idotdot * fact
    return Idotdot


# Estimate frequency evolution from zero crossings
def instfreq(hp, hx, dt):
    hp_zeros = np.where((hp[1:] < 0) & (hp[:-1] > 0))[0]
    hx_zeros = np.where((hx[1:] < 0) & (hx[:-1] > 0))[0]
    hp_period = np.diff(hp_zeros) * dt
    hx_period = np.diff(hx_zeros) * dt
    hp_freq = 1.0 / hp_period
    hx_freq = 1.0 / hx_period
    n = len(hp)
    t = np.linspace(0, dt * (n - 1), n)
    hp_freq_interp = np.interp(t, t[hp_zeros][:-1], hp_freq)
    hx_freq_interp = np.interp(t, t[hx_zeros][:-1], hx_freq)
    freq = np.mean([hp_freq_interp, hx_freq_interp], axis=0)
    return freq

 # Spherical harmonics
def re_ylm_2(l,m,theta,phi):
    if l != 2:
        print("l != 2 not implemented! Good Bye!")
        sys.exit()
    if m < -2 or m > 2:
        print("m must be in [-2,2]! Good Bye!")
        sys.exit()
    if m == 0:
        ret = np.sqrt(15.0/32.0/pi) * np.sin(theta)**2
    if m == 1:
        ret = np.sqrt(5.0/16.0/pi) * (np.sin(theta) + (1.0+np.cos(theta))) * np.cos(phi)
    if m == 2:
        ret = np.sqrt(5.0/64.0/pi) * (1.0+np.cos(theta))**2 * np.cos(2.0*phi)
    if m == -1:
        ret = np.sqrt(5.0/16.0/pi) * (np.sin(theta) + (1.0-np.cos(theta))) * np.cos(phi)
    if m == -2:
        ret = np.sqrt(5.0/64.0/pi) * (1.0-np.cos(theta))**2 * np.cos(2.0*phi)
    return ret

def im_ylm_2(l,m,theta,phi):
    if l != 2:
        print("l != 2 not implemented! Good Bye!")
        sys.exit()
    if m < -2 or m > 2:
        print("m must be in [-2,2]! Good Bye!")
        sys.exit()
    if m == 0:
        ret = 0.0e0
    if m == 1:
        ret = np.sqrt(5.0/16.0/pi) * (np.sin(theta) + (1.0+np.cos(theta))) * np.sin(phi)
    if m == 2:
        ret = np.sqrt(5.0/64.0/pi) * (1.0+np.cos(theta))**2 * np.sin(2.0*phi)
    if m == -1:
        ret = - np.sqrt(5.0/16.0/pi) * (np.sin(theta) + (1.0-np.cos(theta))) * np.sin(phi)
    if m == -2:
        ret = - np.sqrt(5.0/64.0/pi) * (1.0-np.cos(theta))**2 * np.sin(2.0*phi)
    return ret

# Expansion parameters to get the (l,m) modes
def re_Hlm(l,m,Idd):
    if l != 2:
        print("l != 2 not implemented! Good Bye!")
        sys.exit()
    if m < -2 or m > 2:
        print("m must be in [-2,2]! Good Bye!")
        sys.exit()
    if m == 0:
        ret = factor * np.sqrt(32*pi/15.0) * \
              (Idd[2,2] - 0.5e0*(Idd[0,0] + Idd[1,1]))
    if m == 1:
        ret =  - factor * np.sqrt(16*pi/5) * \
              Idd[0,2]
    if m == 2:
        ret = factor * np.sqrt(4*pi/5) * \
              (Idd[0,0] - Idd[1,1])
    if m == -1:
        ret =  factor * np.sqrt(16*pi/5) * \
              Idd[0,2]
    if m == -2:
        ret = factor * np.sqrt(4.0*pi/5) * \
               (Idd[0,0] - Idd[1,1])
    return ret

def im_Hlm(l,m,Idd):
    if l != 2:
        print("l != 2 not implemented! Good Bye!")
        sys.exit()
    if m < -2 or m > 2:
        print("m must be in [-2,2]! Good Bye!")
        sys.exit()
    if m == 0:
        ret = 0.0e0
    if m == 1:
        ret =  factor * np.sqrt(16*pi/5) * \
              Idd[1,2]
    if m == 2:
        ret = factor * np.sqrt(4*pi/5) * \
              -2 * Idd[0,1]
    if m == -1:
        ret = +factor * np.sqrt(16*pi/5) * \
              Idd[1,2]
    if m == -2:
        ret =  factor * np.sqrt(4.0*pi/5) * \
              +2 * Idd[0,1]
    return ret


# Let's have fun
# parameters
totaltime = 30 # seconds, duration
dt = 6.1035e-05  # sampling time

# physical parameters
Mbh = 10. * msun  # mass of the central BH (3-10 Msun)
astar = 0.95      # spin of the central BH; astar = Jbh/Mbh^2 (0.3-0.99)
Jbh = astar * Mbh**2. * ggrav/clite  # In cgs units. astar is dimensionless
Mdisk = 1.5 * msun    # mass of the accretion disk/torus (how big?)
epsilon = 0.2    # fraction of Mdisk that goes into the clumps (0.01-0.5 ?)
mch = epsilon * Mdisk  # mass of the chunks forming the 'binary'
rct = 10000000.    # radius of the thin disk. Binary is at rct + Risco
D = 3.08568e18*100*1e6  # 100 Mpc

# sky position
theta = 0.0
phi = 0.0

# Initial conditions
M0 = Mbh
J0 = Jbh
Risco0 = Risco(J0,M0)

olddata = Risco0, J0, M0   # In cgs units

print('.....................................')
print('Integrating system with central BH of')
print('mass (Msun) =', Mbh/msun, 'and spin a* =', astar)
print('mass of each of the clumps (Msun) =', mch/msun)
print('orbiting at (risco + ', rct/1e5, ' km)')
print('evolving the system for', totaltime, 'seconds')
print('.....................................')

# Output file
filename = 'M' + str(int(Mbh/msun)) + 'a' + str(astar) + 'eps' + str(epsilon) + '.dat'

f1 = open(os.path.join(DATA_DIR, 'vanputten.dat'), 'w')  # File for testing and fun
f2 = open(os.path.join(DATA_DIR, filename),'w')   # File for actual output for xpipeline

time = 0.0

# Variables to store Risco, J, M in case we want to plot them
t = [time]
r = [Risco0]
jbh = [J0]
mbh = [M0]
ast = [astar]
hplus = [0.]
hcross = [0.]

idotdot = Idotdot(mch,rct + Risco0,Omega(M0, rct + Risco0),time)

# Output file contains a lot of stuff
f1.write('Time \t Risco \t Jbh \t Mbh \t a* \t rad Energy \t h+ \t hx')
f1.write('\n')

# Write to file initial conditions before we start evolving
f1.write('\t'.join(str(col) for col in [time, Risco0, J0, M0, astar, hplus[0], hcross[0]]))
f1.write('\n')

f2.write('\t'.join(str(col) for col in [time, hplus[0], hcross[0]]))
f2.write('\n')

# Integrate
while (time < totaltime and Jbh >= 0.):
    # Evolve variables using Runge Kutta 4
    newdata = rk4(olddata,time,dt)
    time += dt

    # Store data in variables
    t.append(time)
    r.append(newdata[0])
    jbh.append(newdata[1])
    mbh.append(newdata[2])
    astar = (clite/ggrav) * newdata[1]/(newdata[2]*newdata[2])
    ast.append(astar)

    # Actualize variable to check complete spin-down
    Jbh = newdata[1]

    # Having evolved Risco, Jbh, Mbh, compute the mass quadrupole momentum
    idotdot = Idotdot(mch,rct + newdata[0],Omega(newdata[2],rct + newdata[0]),time)
    # Compute gravitational radiation!
    h = 0.0
    for m in [-2,-1,0,1,2]:
        cylm = complex(re_ylm_2(2,m,theta,phi), im_ylm_2(2,m,theta,phi))
        Hlm  = complex(re_Hlm(2,m,idotdot),im_Hlm(2,m,idotdot))
        h = h + cylm * Hlm
    hp = h.real/D
    hx = h.imag/D

    hplus.append(hp)
    hcross.append(hx)

    # print(to file)
    f1.write('\t'.join(str(col) for col in [time, newdata[0],
    newdata[1], newdata[2], astar, (M0 - newdata[2])*clite**2., hp, hx]))
    f1.write('\n')

    # print(to xpipeline file)
    f2.write('\t'.join(str(col) for col in [time, hp, hx]))
    f2.write('\n')

    olddata = newdata

# Check whether we ran out of time or whether the BH was completely spun down
if Jbh < 0:
    print("The BH was totally spun down at time t =", time)
    print("Evolution stopped before given total time =", totaltime)

f1.close()


# Plot time series

t = np.array(t)
hplus = np.array(hplus)
hcross = np.array(hcross)
hrss = np.sqrt(hplus**2 + hcross**2)
foft = instfreq(hplus, hcross, dt)

xlim = (dt, t.max() - dt)
plot_idx = (xlim[0] < t) & (t < xlim[1])

fig, ax = plt.subplots(2, 1, figsize=(6, 3.5))
fig.subplots_adjust(left=0.1, bottom=0.13, right=0.97, top=0.95, hspace=0.2)

ax[0].plot(t[plot_idx], hrss[plot_idx], color='black')
ax[0].set_xlim(xlim)
ax[0].set_ylim(0.75e-22, 1.5e-22)
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel(r'$h_{\mathrm{rss}}$')

ax[1].plot(t[plot_idx], foft[plot_idx], color='black')
ax[1].set_xlim(xlim)
ax[1].set_ylim(100, 250)
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Frequency [Hz]')

fig.savefig(os.path.join(OUT_DIR, 'vanputten.pdf'))

print("Done")
