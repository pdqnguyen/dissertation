import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils import load_cf

plt.style.use('./custom-style.mplstyle')


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/cf-locations-vib/')
OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/noise-methods/')


def plot_injection_locations():
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    cf_dict = {os.path.splitext(os.path.basename(file))[0]: load_cf(file) for file in files}
    injections = list(cf_dict.keys())

    freqs = cf_dict[injections[0]]['frequency']
    cf_all = np.zeros((len(freqs), len(injections)))
    for j, injection in enumerate(injections):
        cf = cf_dict[injection]
        meas = cf['flag'] == 'Measured'
        cf_all[meas, j] = cf['factor'][meas]

    injection_groups = {
        'HAM5 top': ('#f5793a', 'o', [j for j, inj in enumerate(injections) if 'HAM5Top' in inj]),
        'HAM5 door': ('#85c0f9', 'x', [j for j, inj in enumerate(injections) if 'HAM5Door' in inj]),
        'HAM6 top': ('#0f2080', '^', [j for j, inj in enumerate(injections) if 'HAM6Top' in inj]),
    }

    frange = (50, 100)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    plt.subplots_adjust(left=0.13, bottom=0.15, right=0.97, top=0.99)
    ymin, ymax = 1, 0
    labels = []
    for name, (color, marker, cols) in injection_groups.items():
        for j in cols:
            y = cf_all[:, j]
            meas = (y > 0) & (freqs >= frange[0]) & (freqs < frange[1])
            if any(meas):
                ymin = min(ymin, y[meas].min())
                ymax = max(ymax, y[meas].max())
            if name not in labels:
                label = name
                labels.append(label)
            else:
                label = None
            plt.plot(freqs[meas], y[meas], marker, color=color, label=label, markerfacecolor='none')
    plt.xlim(frange)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Vibrational coupling [m/m]')
    plt.yscale('log')
    plt.grid(which='both', axis='both')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(OUT_DIR, 'cf-locations-vib.pdf'))
    return


if __name__ == '__main__':
    plot_injection_locations()
