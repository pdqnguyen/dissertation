#! /usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

plt.style.use('./custom-style.mplstyle')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/noise-jitter/')
OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/noise-studies/')

OLD_FNAME = 'old.txt'
NEW_FNAME = 'new.txt'
BINWIDTH = 1


def logbin(x, bins):
    x_split = np.split(x, bins)
    x_b = np.zeros(len(x_split))
    for i, slice_ in enumerate(x_split):
        if any(slice_ > 0):
            x_b[i] = slice_[slice_ > 0].mean()
    return x_b


def logbin_flags(x, bins):
    x_split = np.split(x, bins)
    x_b = np.zeros(len(x_split), dtype=object)
    for i, slice_ in enumerate(x_split):
        if 'Measured' in slice_:
            x_b[i] = 'Measured'
        elif 'Upper Limit' in slice_:
            x_b[i] = 'Upper Limit'
        else:
            x_b[i] = 'No data'
    return x_b


def logbin_df(df, binwidth):
    raw_freqs = df.frequency
    bins = []
    i = 1
    while i < len(raw_freqs):
        bins.append(int(i))
        i += 1 + i * (binwidth / 100.)
    freqs = logbin(df.frequency.values, bins)
    flags = logbin_flags(df.flag.values, bins)
    ambient = logbin(df.ambient.values, bins)
    darm = logbin(df.darm.values, bins)
    out = pd.DataFrame(dict(
        frequency=freqs,
        flag=flags,
        ambient=ambient,
        darm=darm,
    ))
    return out


def main():
    script = os.path.basename(__file__)
    outfile = os.path.join(OUT_DIR, script.replace('.py', '.pdf'))
    old = pd.read_csv(os.path.join(DATA_DIR, OLD_FNAME), comment='#')
    new = pd.read_csv(os.path.join(DATA_DIR, NEW_FNAME), comment='#')

    old['ambient'] = old.sensBG * old.factor
    old['darm'] = old['darmBG']

    if BINWIDTH > 0:
        old = logbin_df(old, binwidth=BINWIDTH)
        new = logbin_df(new, binwidth=BINWIDTH)

    old_meas = old[old.flag == 'Measured']
    old_uppr = old[old.flag == 'Upper Limit']
    new_meas = new[new.flag == 'Measured']
    new_uppr = new[new.flag == 'Upper Limit']
    fmin = max(
        old.frequency[(old.flag == 'Measured') | (old.flag == 'Upper Limit')].min(),
        new.frequency[(new.flag == 'Measured') | (new.flag == 'Upper Limit')].min()
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.97, top=0.97)
    line_darm, = ax.plot(new.frequency.values, new.darm.values, 'k')
    line_old_uppr, = ax.plot(old_uppr.frequency, old_uppr.ambient, 'x', alpha=0.1, label=None)
    line_new_uppr, = ax.plot(new_uppr.frequency, new_uppr.ambient, 'x', alpha=0.1, label=None)
    color_old = line_old_uppr.get_color()
    color_new = line_new_uppr.get_color()
    line_old_meas, = ax.plot(old_meas.frequency, old_meas.ambient, '.', color=color_old)
    line_new_meas, = ax.plot(new_meas.frequency, new_meas.ambient, '.', color=color_new)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(fmin, 1024)
    ax.set_ylim(1e-22, 1e-18)
    ax.set_xlabel('Frequency [Hz]')
    # ax.set_ylabel('Jitter coupling\n[m/beam diameters]')
    ax.set_ylabel(r'DARM $\left[\mathrm{m}/\mathrm{Hz}^{1/2}\right]$')
    ax.grid(which='minor', axis='both', ls=':', alpha=0.6)
    ax.grid(which='major', axis='both', ls=':')
    ax.legend([line_darm, line_old_meas, line_new_meas], ["DARM", "Apr 2019", "May 2021"], loc='upper left')
    plt.savefig(outfile)
    return


if __name__ == '__main__':
    main()
