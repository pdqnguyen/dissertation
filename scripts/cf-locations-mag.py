import glob as glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.style.use('./custom-style.mplstyle')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/cf-locations-mag/')
OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/noise-methods/')
CHANNELS = [
    'H1:PEM-CS_MAG_EBAY_LSCRACK_XYZ',
    'H1:PEM-EX_MAG_VEA_FLOOR_XYZ',
    'H1:PEM-EY_MAG_VEA_FLOOR_XYZ',
    'L1:PEM-CS_MAG_LVEA_VERTEX_XYZ',
    'L1:PEM-EX_MAG_VEA_FLOOR_XYZ',
    'L1:PEM-EY_MAG_VEA_FLOOR_XYZ',
]
LOCATIONS = [
    'H1 corner station',
    'H1 end station X',
    'H1 end station Y',
    'L1 corner station',
    'L1 end station X',
    'L1 end station Y',
]
DIRECTORIES = [
    'h1cs',
    'h1ex',
    'h1ey',
    'l1cs',
    'l1ex',
    'l1ey',
]
FRANGE = (5, 300)
FCOMB = 7.1


def get_data(channel, directory, freqs, method='geom_stdev'):
    files = glob.glob(directory + '/*.txt')
    df_dict = {}
    injection_dict = {}
    for file in files:
        try:
            location = re.search('\d+(\w+?)_', os.path.basename(file)).group(1)
        except AttributeError:
            continue
        injection = os.path.basename(file).replace('.txt', '')
        df = pd.read_csv(file, comment='#')
        df_dict[injection] = df
        if location in injection_dict.keys():
            injection_dict[location].append(injection)
        else:
            i = len(injection_dict)
            injection_dict[location] = [injection]
    injections = list(df_dict.keys())
    locations = list(injection_dict.keys())

    df_all = pd.DataFrame(index=range(len(freqs)), columns=['frequency'] + injections)
    df_all.frequency = freqs
    for injection in injections:
        meas = (df_dict[injection].flag == 'Measured')
        meas_freqs = df_dict[injection].frequency[meas]
        meas_factors = df_dict[injection].factor[meas]
        column = df_all[injection].copy()
        for i, freq in enumerate(freqs):
            column.iloc[i] = meas_factors[np.abs(meas_freqs - freq) < 1.0].min()
        df_all[injection] = column
    df_all = df_all.fillna(0)
    df_new = pd.DataFrame(index=range(len(freqs)), columns=['frequency'] + locations)
    df_new['frequency'] = df_all.frequency
    for location, injections in injection_dict.items():
        x = df_all[injections]
        df_new[location] = x[x > 0].min(axis=1)
    df_new = df_new.fillna(0)
    df_new = df_new[(df_new[locations] > 0).sum(1) > 1]
    meas = (df_new[df_new.columns[1:]] > 0).any(axis=1)
    cf_std = df_new[df_new.columns[1:]][meas].apply(eval(method), axis=1)
    out = pd.concat((df_new.frequency[meas], cf_std), axis=1)
    out.columns = ['frequency', 'stdev']
    return out


def geom_stdev(x):
    x = x[x > 0].astype(float).values
    n = len(x)
    if n > 1:
        x_mean = np.prod(x)**(1.0 / n)
        x_std = np.exp(np.sqrt(np.sum(np.log(x / x_mean)**2) / n))
        return x_std
    else:
        return 0

def norm_stdev(x):
    x = x[x > 0].astype(float).values
    if len(x) > 1:
        return x.std() / x.mean()
    else:
        return 0

def main():
    freqs = np.arange(FCOMB, max(FRANGE), FCOMB)
    data_list = []
    for channel, d in zip(CHANNELS, DIRECTORIES):
        print(channel)
        single = get_data(channel, os.path.join(DATA_DIR, d), freqs, method='geom_stdev')
        single.columns = ['frequency', channel]
        data_list.append(single)
    combined = pd.concat(data_list, axis=1).fillna(0)
    combined = pd.concat((combined['frequency'].max(1), combined[CHANNELS]), axis=1)
    combined.columns = ['frequency'] + CHANNELS
    # combined.to_csv('combined_geom_stdev.csv', float_format='%.2f')

    mean_by_freq = combined[CHANNELS].apply(lambda x: np.mean(x[x > 0]), axis=1)
    print(f"Average standard deviation per bin: {mean_by_freq.mean():.2f}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.97, top=0.97)
    # ax.set_position([0.12, 0.15, 0.86, 0.81])

    ax.plot(combined['frequency'], mean_by_freq, 'x', ms=10, label='Bin average')
    ax.axhline(mean_by_freq.mean(), label='Overall average')
    for channel in CHANNELS:
        show = combined[channel] > 0
        ax.plot(combined['frequency'][show], combined[channel][show], 'o', label=channel[7:14].replace('_',' '))
    ax.set_xlim(FRANGE)
    ax.set_ylim(0.5, 6)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('$\sigma_g$', rotation=0, labelpad=20)
    # ax.set_title('Geometric standard deviation of coupling\nfactors from multiple injections')
    ax.grid(which='minor', axis='both')
    ax.grid(which='major', axis='both')
    ax.legend(loc='upper right', ncol=2, framealpha=0.8)
    plt.savefig(os.path.join(OUT_DIR, 'cf-locations-mag.pdf'))
    return


if __name__ == '__main__':
    main()
