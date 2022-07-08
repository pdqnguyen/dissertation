import os
import matplotlib.pyplot as plt

from utils import load_cf

plt.style.use('./custom-style.mplstyle')


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/noise-48hz')
OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/noise-studies/')

COLORS = ['#f5793a', '#056BA3']


def plot_48hz_injection():
    cf = load_cf(os.path.join(DATA_DIR, 'ham3.txt'))
    freqs = cf['frequency']
    flags = cf['flag']
    factor = cf['factor']
    sens_bkg = cf['sensBG']
    sens_inj = cf['sensINJ']
    darm_bkg = cf['darmBG']
    darm_inj = cf['darmINJ']
    ambient = sens_bkg * factor

    fig, ax = plt.subplots(2, 1, figsize=(6, 4))

    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.97, top=0.97)
    ax[0].plot(freqs, sens_inj, color=COLORS[0])
    ax[0].plot(freqs, sens_bkg, 'k-')
    ax[0].set_xlim(20, 300)
    ax[0].set_ylim(1e-12, 1e-8)
    # ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'Sensor ASD $\left[\mathrm{m}/\sqrt{\mathrm{Hz}}\right]$')
    ax[0].grid(True, which='major', axis='both')

    ax[1].plot(freqs, darm_inj, color=COLORS[0], label='Injection')
    ax[1].plot(freqs, darm_bkg, 'k-', label='Background')
    meas = (flags == 'Measured') & (freqs < 60)
    ax[1].plot(freqs[meas], ambient[meas], color=COLORS[1], marker='.', lw=0, label='Estimated\nambient')
    ax[1].set_xlim(20, 300)
    ax[1].set_ylim(1e-20, 1e-18)
    # ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel(r'DARM ASD $\left[\mathrm{m}/\sqrt{\mathrm{Hz}}\right]$')
    ax[1].grid(True, which='major', axis='both')
    ax[1].legend(loc='upper right')
    fig.savefig(os.path.join(OUT_DIR, 'vib-48hz-injection.pdf'))
    return


if __name__ == '__main__':
    plot_48hz_injection()
