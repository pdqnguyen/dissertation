import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from utils import load_cf

plt.style.use('./custom-style.mplstyle')


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/injection-wallcoil')
OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/noise-methods/')


def plot_wallcoil():
    comb_files = glob.glob(os.path.join(DATA_DIR, 'magnetic_comb*.txt'))
    comb_cf_list = [load_cf(file) for file in comb_files]
    comb_freqs = comb_cf_list[0]['frequency']
    comb_bg = comb_cf_list[0]['darmBG']
    comb_injs = [cf['darmINJ'] for cf in comb_cf_list]

    broadband_files = glob.glob(os.path.join(DATA_DIR, 'magnetic_broadbd*.txt'))
    broadband_cf_list = [load_cf(file) for file in broadband_files]
    broadband_freqs = broadband_cf_list[0]['frequency']
    broadband_bg = broadband_cf_list[0]['darmBG']
    broadband_injs = [cf['darmINJ'] for cf in broadband_cf_list]

    xlim = (20, 300)
    ylim = (1e-20, 1e-17)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.95)
    ax.plot(broadband_freqs, np.maximum.reduce(broadband_injs), color='#015d8e', label=r'$5\times5$-m coil')
    ax.plot(comb_freqs, np.maximum.reduce(comb_injs), color='#85c0f9', label='1-m coil')
    ax.plot(comb_freqs, comb_bg, 'k-')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'DARM ASD $\left[\mathrm{m}/\sqrt{\mathrm{Hz}}\right]$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, which='major', axis='both')
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(OUT_DIR, 'injection-wallcoil.pdf'))
    return


if __name__ == '__main__':
    plot_wallcoil()
