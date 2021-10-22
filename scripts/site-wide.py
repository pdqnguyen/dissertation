import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

plt.style.use('./custom-style.mplstyle')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/ambient')
ALIGO_SENS_PATH = os.path.join(DATA_DIR, 'aligo-sens.h5')
DARM_LHO_PATH = os.path.join(DATA_DIR, 'darm-lho.txt')
DARM_LLO_PATH = os.path.join(DATA_DIR, 'darm-llo.txt')
VIB_LHO_PATH = os.path.join(DATA_DIR, 'ambient-vib-lho-post-o3.txt')
VIB_LLO_PATH = os.path.join(DATA_DIR, 'ambient-vib-llo-post-o3.txt')
MAG_LHO_PATH = os.path.join(DATA_DIR, 'ambient-mag-lho-post-o3.txt')
MAG_LLO_PATH = os.path.join(DATA_DIR, 'ambient-mag-llo-post-o3.txt')

OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/')

with h5py.File(ALIGO_SENS_PATH, 'r') as f:
    aligo_sens = (f['Freq'][()], f['traces']['Total'][()])
darm_dict = dict(
    lho=pd.read_csv(DARM_LHO_PATH),
    llo=pd.read_csv(DARM_LLO_PATH),
)
ambient_dict = dict(
    vib=dict(
        lho=pd.read_csv(VIB_LHO_PATH),
        llo=pd.read_csv(VIB_LLO_PATH)
    ),
    mag=dict(
        lho=pd.read_csv(MAG_LHO_PATH),
        llo=pd.read_csv(MAG_LLO_PATH)
    ),
)

colors = ['#015d8e', '#85c0f9', '#f5793a']

# figure for each coupling type, subplot for each observatory
markersizes = dict(
    vib=2,
    mag=5,
)
for coupling_type in ambient_dict.keys():
    fig, ax = plt.subplots(2, 1, figsize=(5, 4))
    plt.subplots_adjust(left=0.2, bottom=0.12, right=0.99, top=0.95, hspace=0.2)
    path = os.path.join(OUT_DIR, f"ambient_{coupling_type}.pdf")
    for i, (observatory, df) in enumerate(ambient_dict[coupling_type].items()):
        darm = darm_dict[observatory]
        ax[i].loglog(darm.frequency, darm.darm, 'k-')
        ax[i].plot(
            df.frequency[df.flag == 'Measured'],
            df.amb[df.flag == 'Measured'],
            '.',
            color=colors[0],
            markersize=markersizes[coupling_type],
            zorder=3
        )
        ax[i].plot(
            df.frequency[df.flag == 'Upper Limit'],
            df.amb[df.flag == 'Upper Limit'],
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
for j, coupling_type in enumerate(ambient_dict.keys()):
    for i, (observatory, df) in enumerate(ambient_dict[coupling_type].items()):
        darm = darm_dict[observatory]
        ax[i, j].loglog(darm.frequency, darm.darm / 4000, 'k-')
        ax[i, j].plot(
            df.frequency[df.flag == 'Measured'],
            df.amb[df.flag == 'Measured'] / 4000,
            '.',
            color=colors[0],
            markersize=markersizes[coupling_type],
            zorder=3,
        )
        ax[i, j].plot(
            df.frequency[df.flag == 'Upper Limit'],
            df.amb[df.flag == 'Upper Limit'] / 4000,
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

