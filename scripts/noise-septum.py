import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_cf, logbin

plt.style.use('./custom-style.mplstyle')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/noise-septum')
OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/noise-studies')
COLORS = ['#f5793a', '#85c0f9', '#056BA3', '#0f2080']


def get_ccf(cfs):
    freqs = cfs[0].frequency.values
    flags = np.array([cf.flag.values for cf in cfs])
    factors = np.array([cf.factor.values for cf in cfs])
    injs = np.array([cf.sensINJ.values for cf in cfs])
    bkgs = np.array([cf.sensBG.values for cf in cfs])
    ratios = np.nan_to_num(injs / bkgs)
    ccf = np.zeros(len(freqs))
    cflags = np.zeros(len(freqs), dtype=flags.dtype)
    for i in range(len(freqs)):
        max_idx = np.argmax(ratios[:, i])
        ccf[i] = factors[max_idx, i]
        cflags[i] = flags[max_idx, i]
    return ccf, cflags


def main():
    cf_dict = {'ham5': [], 'ham6': []}
    for file in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, file)
        data = pd.read_csv(filepath, comment='#')
        if 'ham5' in file:
            cf_dict['ham5'].append(data)
        elif 'ham6' in file:
            cf_dict['ham6'].append(data)
    columns = cf_dict['ham5'][0].columns
    freqs = cf_dict['ham5'][0].frequency.values
    sens_bkg = np.array([cf.sensBG.values for cf in cf_dict['ham5'] + cf_dict['ham6']]).max(axis=0)
    darm_bkg = np.array([cf.darmBG.values for cf in cf_dict['ham5'] + cf_dict['ham6']]).max(axis=0)
    sens_injs_ham5 = np.array([cf.sensINJ.values for cf in cf_dict['ham5']])
    sens_injs_ham6 = np.array([cf.sensINJ.values for cf in cf_dict['ham6']])
    darm_injs_ham5 = np.array([cf.darmINJ.values for cf in cf_dict['ham5']])
    darm_injs_ham6 = np.array([cf.darmINJ.values for cf in cf_dict['ham6']])
    sens_inj_ham5 = sens_injs_ham5.max(axis=0)
    sens_inj_ham6 = sens_injs_ham6.max(axis=0)
    darm_inj_ham5 = darm_injs_ham5.max(axis=0)
    darm_inj_ham6 = darm_injs_ham6.max(axis=0)
    ccf_ham5, cflags_ham5 = get_ccf(cf_dict['ham5'])
    ccf_ham6, cflags_ham6 = get_ccf(cf_dict['ham6'])
    amb_ham5 = sens_bkg * ccf_ham5
    amb_ham6 = sens_bkg * ccf_ham6
    # Lower threshold to show some upper limits as measurements
    # b/c we're just showing two channels so we don't need to be so strict.
    meas_ham5 = (cflags_ham5 == 'Measured') | (darm_inj_ham5 / darm_bkg > 2)
    meas_ham6 = (cflags_ham6 == 'Measured') | (darm_inj_ham6 / darm_bkg > 2)

    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    fig.subplots_adjust(left=0.15, bottom=0.13, right=0.97, top=0.95, hspace=0.2)
    ax[0].plot(freqs, sens_inj_ham5, lw=2, color=COLORS[0])
    ax[0].plot(freqs, sens_inj_ham6, lw=2, color=COLORS[1])
    ax[0].plot(freqs, sens_bkg, lw=2, color='black')
    ax[1].plot(freqs, darm_inj_ham5, lw=2, color=COLORS[0], label="HAM5 injections")
    ax[1].plot(freqs, darm_inj_ham6, lw=2, color=COLORS[1], label="HAM6 injections")
    ax[1].plot(freqs[meas_ham5], amb_ham5[meas_ham5], '.', ms=3, color=COLORS[0])
    ax[1].plot(freqs[meas_ham6], amb_ham6[meas_ham6], '.', ms=3, color=COLORS[1])
    ax[1].plot(freqs, darm_bkg, lw=2, color='black', label="Background")
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_xlim(10, 150)
    ax[1].set_xlim(10, 150)
    ax[0].set_ylim(1e-10, 1e-6)
    ax[1].set_ylim(1e-22, 1e-16)
    ax[0].grid(True, which='major', axis='both')
    ax[1].grid(True, which='major', axis='both')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel(r'Displacement $\left[\mathrm{m}/\sqrt{\mathrm{Hz}}\right]$')
    ax[1].set_ylabel(r'DARM $\left[\mathrm{m}/\sqrt{\mathrm{Hz}}\right]$')
    ax[1].legend(loc='upper right', ncol=2, framealpha=0.6)
    fig.savefig(os.path.join(OUT_DIR, 'vib-septum-shaker.pdf'))
    return


if __name__ == '__main__':
    main()
