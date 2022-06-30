import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('./custom-style.mplstyle')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/cf-example')
PSL_INJECTION_PATH = os.path.join(DATA_DIR, 'psl-injection.txt')

OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/noise-methods')


data = pd.read_csv(PSL_INJECTION_PATH, comment='#')[::3]
freqs = data.frequency.values
flags = data.flag.values
sens_bkg = data.sensBG.values
sens_inj = data.sensINJ.values
darm_bkg = data.darmBG.values
darm_inj = data.darmINJ.values

sens_thresh = 2
darm_thresh = 2
above_sens = (sens_inj > sens_thresh * sens_bkg)
above_darm = (darm_inj > darm_thresh * darm_bkg)
nonzero = ((flags == 'Measured') | (flags == 'Upper Limit'))
meas = (nonzero & above_sens & above_darm)
uppr = (nonzero & above_sens & ~meas)
cf = np.zeros(data.shape[0])
cf[meas] = np.sqrt((darm_inj[meas]**2 - darm_bkg[meas]**2) / (sens_inj[meas]**2 - sens_bkg[meas]**2))
cf[uppr] = darm_inj[uppr] / np.sqrt((sens_inj[uppr]**2 - sens_bkg[uppr]**2))

colors = ['#f5793a', '#85c0f9', '#056BA3', '#0f2080']

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 6))
plt.subplots_adjust(left=0.17, bottom=0.1, right=0.98, top=0.95, hspace=0.5)
ax1.plot(freqs, sens_inj, lw=2, color=colors[0])
ax1.plot(freqs, sens_bkg, lw=2, color='black')
ax2.plot(freqs, darm_inj, lw=2, color=colors[0])
ax2.plot(freqs, darm_bkg, lw=2, color='black')
ambient = sens_bkg * cf
ax2.plot(freqs[uppr], ambient[uppr], 'x', mew=1, color=colors[1])
ax2.plot(freqs[meas], ambient[meas], '.', color=colors[2])
ax3.plot(freqs[uppr], cf[uppr], 'x', mew=1, color=colors[1])
ax3.plot(freqs[meas], cf[meas], '.', color=colors[2])

# ax1.set_xscale('log')
# ax2.set_xscale('log')
# ax3.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_yscale('log')

# frange = (10, 130)
frange = (50, 250)
ax1.set_xlim(frange)
ax2.set_xlim(frange)
ax3.set_xlim(frange)

# ax1.set_ylim(3e-12, 2e-8)
# ax2.set_ylim(3e-22, 1e-18)
# ax3.set_ylim(4e-12, 1e-9)
ax1.set_ylim(1e-12, 1e-8)
ax2.set_ylim(1e-22, 1e-18)
ax3.set_ylim(4e-12, 1e-9)


# xticks = range(frange[0], frange[1] + 1, 20)
# xticks = range(20, frange[1] + 1, 20)
xticks = range(50, frange[1] + 1, 50)
ax1.set_xticks(xticks)
ax2.set_xticks(xticks)
ax3.set_xticks(xticks)

# ax1.grid(True, which='minor', axis='x')
# ax2.grid(True, which='minor', axis='x')
# ax3.grid(True, which='minor', axis='x')
ax1.grid(True, which='major', axis='both')
ax2.grid(True, which='major', axis='both')
ax3.grid(True, which='major', axis='both')

# ax1.set_xlabel('Frequency [Hz]')
# ax2.set_xlabel('Frequency [Hz]')
ax3.set_xlabel('Frequency [Hz]')

ax1.set_ylabel(r'Displacement $\left[\mathrm{m}/\mathrm{Hz}^{1/2}\right]$')
ax2.set_ylabel('DARM ASD ' + r'$\left[\mathrm{m}/\mathrm{Hz}^{1/2}\right]$')
ax3.set_ylabel(r'Vibrational coupling [m/m]')

# ax1.set_title('HAM6 accelerometer')
ax1.set_title('PSL accelerometer')
ax2.set_title('Detector response')
ax3.set_title('Coupling function')

plt.savefig(os.path.join(OUT_DIR, 'cf-example.pdf'))
