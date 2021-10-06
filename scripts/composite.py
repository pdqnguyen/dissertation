import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils import logbin, interp1d, load_cf


plt.style.use('./custom-style.mplstyle')


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/composite')
FILES = [
    'HAM5_shaker_20-100_Hz.txt',
    'HAM6_shaker_5-100_Hz.txt',
    'HAM6_shaker_120-200_Hz.txt',
    'HAM5_speaker_30-2000_Hz.txt',
    'HAM6_speaker_30-2000_Hz.txt',
]
OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures')
COLORS = ['#056BA3', '#f0e442', '#a2c8ec', '#c35e08', '#A1D99B']


def plot_composite():
    injections = []
    cf_list = []
    for file in FILES:
        injections.append(os.path.splitext(file)[0].replace('_', ' '))
        cf_list.append(load_cf(os.path.join(DATA_DIR, file)))

    freqs = cf_list[0]['frequency']
    darm = np.median(np.column_stack([cf['darmBG'] for cf in cf_list]), axis=1)
    ambient_all = np.column_stack([cf['factor'] * cf['sensBG'] for cf in cf_list])
    sensinj_all = np.column_stack([cf['sensBG'] for cf in cf_list])
    flag_all = np.column_stack([cf['flag'] for cf in cf_list])

    # Bin all data
    binwidth = 0.3
    bins = []
    i = 1
    while i < len(freqs):
        bins.append(int(i))
        i += 1 + i * (binwidth / 100.)
    freqs_b = logbin(freqs, bins)
    darm_b = logbin(darm, bins)
    ambient_all_b = np.zeros((len(freqs_b), len(injections)))
    sensinj_all_b = np.zeros_like(ambient_all_b)
    flag_all_b = np.zeros_like(ambient_all_b).astype(object)
    for j in range(len(injections)):
        ambient_all_b[:, j] = logbin(ambient_all[:, j], bins)
        sensinj_all_b[:, j] = logbin(sensinj_all[:, j], bins)
        flag_all_b[:, j] = interp1d(freqs_b, freqs, flag_all[:, j], kind='nearest')

    # Compute composite coupling function
    comp_ambient = np.zeros(len(freqs_b))
    comp_flag = np.zeros(len(freqs_b), dtype=int)
    comp_inj = np.array(['No data'] * len(freqs_b), dtype=object)
    for i, row in enumerate(ambient_all_b):
        meas_row = (flag_all_b[i] == 'Measured')
        uppr_row = (flag_all_b[i] == 'Upper Limit')
        if meas_row.sum() > 0:
            j = np.where(row == row[meas_row].min())[0][0]
            comp_flag[i] = 2
        elif uppr_row.sum() > 0:
            j = np.where(row == row[uppr_row].min())[0][0]
            comp_flag[i] = 1
        else:
            continue
        comp_ambient[i] = row[j]
        comp_inj[i] = injections[j]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    # plt.subplots_adjust(left=0.18, bottom=0.15, right=0.5, top=0.95)
    plt.subplots_adjust(left=0.18, bottom=0.12, right=0.99, top=0.78)
    # ax = fig.add_axes([0.11, 0.15, 0.55, 0.8])
    ax.plot(freqs_b, darm_b, 'k', lw=2, zorder=1)
    handles = [Patch(fc='k')]
    labels = ['DARM background']
    for color, inj in zip(COLORS, injections):
        uppr = (comp_inj == inj) & (comp_flag == 1)
        meas = (comp_inj == inj) & (comp_flag == 2)
        ax.plot(freqs_b[uppr], comp_ambient[uppr], 'x', color=color, zorder=1)
        ax.plot(freqs_b[meas], comp_ambient[meas], 'o', mfc=color, mec='black', mew=0.5, zorder=2)
        handles.append(Patch(fc=color))
        labels.append(inj)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(20, 2048)
    ax.set_ylim(5e-23, 1e-18)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'DARM ASD $\left[\mathrm{m}/\mathrm{Hz}^{1/2}\right]$')
    ax.set_xticks([50, 100, 500, 1000])
    ax.grid(True, which='both', axis='x')
    ax.grid(True, which='major', axis='y')
    # ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    ax.legend(handles, labels, ncol=2, loc='lower right', bbox_to_anchor=(1, 1.02))
    fig.savefig(os.path.join(OUT_DIR, 'composite.pdf'))
    return


if __name__ == '__main__':
    plot_composite()
