import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import load_cf

plt.style.use('./custom-style.mplstyle')


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/noise-septum-comparison')
OUT_DIR =  os.path.join(os.path.dirname(__file__), '../figures/noise-studies')

OLD_INJECTIONS = [
    '966CSBigShakerHAM5Door_5to20Hz_redo',
    '967CSBigShakerHAM5Door_20to100Hz_redo',
    '968CSBigShakerHAM5Door_100to500Hz_redo',
    '12EMshakeHAM5Top_40to80Hz',
    '01EMshakeHAM5Top_5to100Hz',
]
NEW_INJECTIONS = [
    '121CSBigShakerHAM5Door_5to20Hz',
    '120CSBigShakerHAM5Door_20to100Hz',
    '119CSBigShakerHAM5Door_100to500Hz',
    '124CSEMShakerHAM5Top_40to80Hz_louder',
    '125CSEMShakerHAM5Top_5to100Hz',
]


def plot_comparison(old_data, new_data):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.subplots_adjust(bottom=0.15, right=0.99, top=0.93)
    for i, (old, new) in enumerate(zip(old_data, new_data)):
        old_meas = old[old.flag == 'Measured']
        new_meas = new[new.flag == 'Measured']
        if i == 0:
            labels = ["Beginning of O3a", "Beginning of O3b"]
        else:
            labels = [None, None]
        ax.plot(old_meas.frequency, old_meas.factor, 'b.', label=labels[0])
        ax.plot(new_meas.frequency, new_meas.factor, 'r.', label=labels[1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(4, 900)
    ax.set_ylim(1e-13, 1e-10)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Vibrational coupling [m/m]")
    ax.set_title("Septum accelerometer coupling comparison")
    ax.grid()
    ax.grid(axis='both', which='minor', ls=':', alpha=0.5)
    ax.legend(loc='lower right', framealpha=0.5)
    plt.savefig(os.path.join(OUT_DIR, 'vib-septum-comparison.pdf'))
    return


def main():
    # Get data
    darm = pd.read_csv(os.path.join(DATA_DIR, 'darm.txt'), delim_whitespace=True, names=['frequency', 'value'])
    old_data = []
    for name in OLD_INJECTIONS:
        df = pd.read_csv(os.path.join(DATA_DIR, f'old/{name}.txt'), comment='#')
        old_data.append(df)
    new_data = []
    for name in NEW_INJECTIONS:
        df = pd.read_csv(os.path.join(DATA_DIR, f'new/{name}.txt'), comment='#')
        new_data.append(df)
    # Combine background spectra
    old_bkg = pd.concat([df.sensBG for df in old_data], 1).min(1)
    new_bkg = pd.concat([df.sensBG for df in new_data], 1).min(1)
    # Plot comparison
    plot_comparison(old_data, new_data)
    # # Combine data across injections for summary stats
    # old_factors = pd.concat([df.factor for df in old_data], 1)
    # old_flags = pd.concat([df.flag for df in old_data], 1)
    # old_factors_meas = old_factors.values.flatten()[old_flags.values.flatten() == 'Measured']
    # old_avg_cf = old_factors_meas.mean()
    # old_std_cf = old_factors_meas.std()
    # new_factors = pd.concat([df.factor for df in new_data], 1)
    # new_flags = pd.concat([df.flag for df in new_data], 1)
    # new_factors_meas = new_factors.values.flatten()[new_flags.values.flatten() == 'Measured']
    # new_avg_cf = new_factors_meas.mean()
    # new_std_cf = new_factors_meas.std()
    # print("{:.1e} +/- {:.1e}".format(old_avg_cf, old_std_cf))
    # print("{:.1e} +/- {:.1e}".format(new_avg_cf, new_std_cf))
    # print(new_avg_cf / old_avg_cf)
    # # Compute estimated ambient noise (not used)
    # for df in old_data:
    #     df['ambient'] = df.factor * old_bkg
    # for df in new_data:
    #     df['ambient'] = df.factor * new_bkg
    return


if __name__ == '__main__':
    main()
