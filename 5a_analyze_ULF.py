import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_mlt_rose(
        df,
        mlt_col="best_mlt",
        bin_width_hours=1.0,
        color="#7fa8c9",
        edgecolor="black",
        title="ULF Wave Events by MLT",
        ax=None,
):
    mlt = df[mlt_col].dropna().to_numpy()
    # Wrap into [0, 24) just in case of stray 24.0 / negative values
    mlt = np.mod(mlt, 24)

    bins = np.arange(0, 24 + bin_width_hours, bin_width_hours)
    counts, edges = np.histogram(mlt, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Map MLT hour -> angle (radians), with 12 at top and clockwise
    # direction corresponding to DEcreasing hour (12, 11, 10, ... 1, 24, 23...)
    theta = np.deg2rad(((12 - centers) % 24) * 15)
    width = np.deg2rad(bin_width_hours * 15) * 0.95  # small gap between wedges

    # If no axis was passed in, make a standalone figure (original behavior).
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, projection="polar")
    else:
        fig = ax.figure

    ax.set_theta_zero_location("N")  # 0 hours-from-noon at the top
    ax.set_theta_direction(-1)  # angle increases clockwise

    ax.bar(
        theta, counts, width=width, bottom=0.0,
        color=color, edgecolor=edgecolor, linewidth=0.4, alpha=0.9,
    )

    # Hour labels around the rim: 12 at top, going clockwise 11,10,...,1,24,...,13
    hour_ticks = np.arange(1, 25)
    hour_angles = np.deg2rad(((12 - hour_ticks) % 24) * 15)
    ax.set_xticks(hour_angles)
    ax.set_xticklabels(hour_ticks)

    ax.set_rlabel_position(135)  # move the radial count labels out of the way
    ax.tick_params(axis='y', labelsize=8)  # keep radial labels from cluttering the wedges
    ax.set_title(title, va="bottom", pad=20)

    if standalone:
        plt.tight_layout()
        plt.show()

    return fig, ax


g13_ulf = pd.read_csv('ULF_shocks/ulf_shock_summary_GOES13.csv')
g15_ulf = pd.read_csv('ULF_shocks/ulf_shock_summary_GOES15.csv')
g16_ulf = pd.read_csv('ULF_shocks/ulf_shock_summary_GOES16.csv')
g17_ulf = pd.read_csv('ULF_shocks/ulf_shock_summary_GOES17.csv')
g18_ulf = pd.read_csv('ULF_shocks/ulf_shock_summary_GOES18.csv')

all_events = pd.concat([g13_ulf, g15_ulf, g16_ulf, g17_ulf, g18_ulf])
all_events.reset_index(drop=True, inplace=True)
print(all_events.shape)
print(all_events.keys())

# convert sub-Pc5 periods to seconds
all_events['period'] = all_events['period'] * 60
# joblib.dump(all_events, 'ULF_shocks/all_ulf_shock_events.pkl')

pc5 = all_events.loc[all_events['best_is_pc5'] == True]
lower_freq = all_events.dropna(subset=['comp'])

all_sig = all_events.loc[all_events['n_detections'] > 0]
none = all_events.loc[all_events['n_detections'] == 0]

print('Total events:', len(all_events))
print('Events with oscillations:', len(all_sig))
print('Events with no detected oscillations:', len(none))
print('Pc5 events:', len(pc5))
print('Sub-pc5 events from Clean code:', len(lower_freq))

both_count = 0
for i in range(len(all_events)):
    event = all_events.iloc[i]
    if (event['best_is_pc5'] == True) & (not pd.isna(event['comp'])):
        both_count += 1
print('Events with both Pc5 and sub-pc5 events:', both_count)

# 2x2 grid of polar histograms: all significant events, then each field component
fig, axes = plt.subplots(
    2, 2, figsize=(9, 9),
    subplot_kw={'projection': 'polar'},
    constrained_layout=True,
)
fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, wspace=0.02, hspace=0.02)

plot_mlt_rose(all_events.loc[all_events['n_detections'] > 0], title='All significant events', ax=axes[0, 0])
plot_mlt_rose(all_events.loc[all_events['best_component'] == 'b_radial'], title='Radial', ax=axes[0, 1])
plot_mlt_rose(all_events.loc[all_events['best_component'] == 'b_azimuthal'], title='Azimuthal', ax=axes[1, 0])
plot_mlt_rose(all_events.loc[all_events['best_component'] == 'b_parallel'], title='Parallel', ax=axes[1, 1])

plt.savefig('paper figures/ulf_dists.png', dpi=300, format='png', bbox_inches='tight')

plt.show()

# Pc5 vs sub-Pc5, plotted separately (unchanged behavior)
plot_mlt_rose(pc5, title='Pc5 events')
plot_mlt_rose(lower_freq, title='Sub-pc5 events')