import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

plt.style.use('./custom-style.mplstyle')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/ambient')
ALIGO_SENS_PATH = os.path.join(DATA_DIR, 'aligo-sens.h5')
VIB_LHO_PATH = os.path.join(DATA_DIR, 'ambient-vib-lho.txt')
VIB_LLO_PATH = os.path.join(DATA_DIR, 'ambient-vib-llo.txt')
MAG_LHO_PATH = os.path.join(DATA_DIR, 'ambient-mag-lho.txt')
MAG_LLO_PATH = os.path.join(DATA_DIR, 'ambient-mag-llo.txt')

OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/')

with h5py.File(ALIGO_SENS_PATH, 'r') as f:
    aligo_sens = (f['Freq'][()], f['traces']['Total'][()])
data = dict(
    vib=dict(
        lho=pd.read_csv(VIB_LHO_PATH).astype(float),
        llo=pd.read_csv(VIB_LLO_PATH).astype(float)
    ),
    mag=dict(
        lho=pd.read_csv(MAG_LHO_PATH).astype(float),
        llo=pd.read_csv(MAG_LLO_PATH).astype(float)
    ),
)

colors = ['#015d8e', '#85c0f9', '#f5793a']

# figure for each coupling type, subplot for each observatory
markersizes = dict(
    vib=2,
    mag=5,
)
for coupling_type in data.keys():
    fig, ax = plt.subplots(2, 1, figsize=(5, 4))
    plt.subplots_adjust(left=0.2, bottom=0.12, right=0.99, top=0.95, hspace=0.2)
    path = os.path.join(OUT_DIR, f"ambient_{coupling_type}.pdf")
    for i, (observatory, df) in enumerate(data[coupling_type].items()):
        ax[i].loglog(df.frequency, df.darm, 'k-')
        ax[i].plot(
            df.frequency[df.flag > 0],
            df.amb[df.flag > 0],
            '.',
            color=colors[0],
            markersize=markersizes[coupling_type],
            zorder=3
        )
        ax[i].plot(
            df.frequency[df.flag == 0],
            df.amb[df.flag == 0],
            '.',
            color=colors[1],
            markersize=markersizes[coupling_type],
            zorder=2
        )
        ax[i].plot(aligo_sens[0], 4000 * np.sqrt(aligo_sens[1]), lw=3, color=colors[2], zorder=1)
        ax[i].set_xlim(5, 2048)
        ax[i].set_ylim(1e-23, 1e-16)
        ax[i].grid(which='major', axis='both', lw=1)
        ax[i].grid(which='minor', axis='both', lw=0.7, ls='--')
        ax[i].set_axisbelow(True)
        ax[i].set_ylabel(observatory.upper() + r' DARM $\left[\mathrm{m}/\mathrm{Hz}^{1/2}\right]$')
    ax[-1].set_xlabel('Frequency [Hz]')
    plt.savefig(path)

# all-in-one-figure plot
coupling_names = dict(
    vib='Vibrational',
    mag='Magnetic',
)
markersizes = dict(
    vib=2,
    mag=5,
)
fig, ax = plt.subplots(2, 2, figsize=(6, 4))
plt.subplots_adjust(left=0.14, bottom=0.12, right=0.99, top=0.93, wspace=0.05, hspace=0.2)
for j, coupling_type in enumerate(data.keys()):
    for i, (observatory, df) in enumerate(data[coupling_type].items()):
        ax[i, j].loglog(df.frequency, df.darm / 4000, 'k-')
        ax[i, j].plot(
            df.frequency[df.flag > 0],
            df.amb[df.flag > 0] / 4000,
            '.',
            color=colors[0],
            markersize=markersizes[coupling_type],
            zorder=3,
        )
        ax[i, j].plot(
            df.frequency[df.flag == 0],
            df.amb[df.flag == 0] / 4000,
            '.',
            color=colors[1],
            markersize=markersizes[coupling_type],
            zorder=2,
        )
        ax[i, j].plot(aligo_sens[0], np.sqrt(aligo_sens[1]), lw=3, color=colors[2], zorder=1)
        ax[i, j].set_xlim(5, 2048)
        ax[i, j].set_ylim(1e-26, 1e-21)
        ax[i, j].grid(which='major', axis='both', lw=1)
        ax[i, j].grid(which='minor', axis='both', lw=0.7, ls='--')
        ax[i, j].set_axisbelow(True)
        ax[-1, j].set_xlabel('Frequency [Hz]')
        ax[i, 0].set_ylabel(observatory.upper() + r' strain $\left[\mathrm{Hz}^{-1/2}\right]$')
        ax[0, j].set_xticklabels([])
        ax[i, -1].set_yticklabels([])
        ax[0, j].set_title(coupling_names[coupling_type])
    plt.savefig(os.path.join(OUT_DIR, 'ambient.pdf'))

