import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

plt.style.use('./custom-style.mplstyle')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/noise-ambient')
ALIGO_SENS_PATH = os.path.join(DATA_DIR, 'aligo-sens.h5')
DARM_LHO_PATH = os.path.join(DATA_DIR, 'darm-lho.txt')
DARM_LLO_PATH = os.path.join(DATA_DIR, 'darm-llo.txt')
VIB_LHO_PATH = os.path.join(DATA_DIR, 'ambient-vib-lho-post-o3.txt')
VIB_LLO_PATH = os.path.join(DATA_DIR, 'ambient-vib-llo-post-o3.txt')
MAG_LHO_PATH = os.path.join(DATA_DIR, 'ambient-mag-lho-post-o3.txt')
MAG_LLO_PATH = os.path.join(DATA_DIR, 'ambient-mag-llo-post-o3.txt')
JITTER_LHO_PATH = os.path.join(DATA_DIR, '../noise-jitter/new.txt')

OUT_DIR = os.path.join(os.path.dirname(__file__), '../figures/noise-studies')


def merge_jitter(full, jitter):
    jitter = jitter[['frequency', 'ambient', 'flag']]
    jitter.columns = ['frequency', 'jitter_amb', 'jitter_flag']
    jitter.loc[jitter.jitter_flag == 'No data', 'jitter_amb'] = np.nan
    full = full.merge(jitter, how='left', on='frequency')
    is_psl = full.channel.str.contains('_PSL_')
    which_lower = full.loc[:, ['amb', 'jitter_amb']].idxmin(axis=1)
    use_jitter = is_psl & (which_lower == 'jitter_amb')
    full.loc[use_jitter, ['amb', 'flag']] = full.loc[use_jitter, ['jitter_amb', 'jitter_flag']].values
    return full


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
jitter_df = pd.read_csv(JITTER_LHO_PATH, comment='#')
ambient_dict['vib']['lho'] = merge_jitter(ambient_dict['vib']['lho'], jitter_df)

colors = ['#015d8e', '#85c0f9', '#f5793a']

# figure for each coupling type, subplot for each observatory
markersizes = dict(
    vib=2,
    mag=5,
)
for coupling_type in ambient_dict.keys():
    fig, ax = plt.subplots(2, 1, figsize=(6, 7))
    plt.subplots_adjust(left=0.15, bottom=0.07, right=0.98, top=0.98, hspace=0.13)
    path = os.path.join(OUT_DIR, f"{coupling_type}-ambient.pdf")
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
        if coupling_type == 'vib':
            ax[i].set_xlim(10, 2048)
            ax[i].set_ylim(1e-22, 1e-17)
        else:
            ax[i].set_xlim(5, 2048)
            ax[i].set_ylim(1e-22, 1e-16)
        ax[i].grid(which='major', axis='both', lw=1, alpha=0.6)
        ax[i].grid(which='minor', axis='both', lw=0.5, ls='--', alpha=0.6)
        ax[i].set_axisbelow(True)
        ax[i].set_ylabel(observatory.upper() + r' DARM $\left[\mathrm{m}/\sqrt{\mathrm{Hz}}\right]$')
    ax[-1].set_xlabel('Frequency [Hz]')
    plt.savefig(path)

# # all-in-one-figure plot
# coupling_names = dict(
#     vib='Vibrational',
#     mag='Magnetic',
# )
# markersizes = dict(
#     vib=2,
#     mag=5,
# )
# fig, ax = plt.subplots(2, 2, figsize=(6, 4))
# plt.subplots_adjust(left=0.14, bottom=0.12, right=0.99, top=0.93, wspace=0.05, hspace=0.2)
# for j, coupling_type in enumerate(ambient_dict.keys()):
#     for i, (observatory, df) in enumerate(ambient_dict[coupling_type].items()):
#         darm = darm_dict[observatory]
#         ax[i, j].loglog(darm.frequency, darm.darm / 4000, 'k-')
#         ax[i, j].plot(
#             df.frequency[df.flag == 'Measured'],
#             df.amb[df.flag == 'Measured'] / 4000,
#             '.',
#             color=colors[0],
#             markersize=markersizes[coupling_type],
#             zorder=3,
#         )
#         ax[i, j].plot(
#             df.frequency[df.flag == 'Upper Limit'],
#             df.amb[df.flag == 'Upper Limit'] / 4000,
#             '.',
#             color=colors[1],
#             markersize=markersizes[coupling_type],
#             zorder=2,
#         )
#         ax[i, j].plot(aligo_sens[0], np.sqrt(aligo_sens[1]), lw=3, color=colors[2], zorder=1)
#         ax[i, j].set_xlim(5, 2048)
#         ax[i, j].set_ylim(1e-26, 1e-21)
#         ax[i, j].grid(which='major', axis='both', lw=1)
#         ax[i, j].grid(which='minor', axis='both', lw=0.7, ls='--')
#         ax[i, j].set_axisbelow(True)
#         ax[-1, j].set_xlabel('Frequency [Hz]')
#         ax[i, 0].set_ylabel(observatory.upper() + r' strain $\left[\mathrm{Hz}^{-1/2}\right]$')
#         ax[0, j].set_xticklabels([])
#         ax[i, -1].set_yticklabels([])
#         ax[0, j].set_title(coupling_names[coupling_type])
#     plt.savefig(os.path.join(OUT_DIR, 'noise-ambient.pdf'))
