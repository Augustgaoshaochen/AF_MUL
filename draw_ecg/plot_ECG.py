import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np

lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def plot_new(
        ecg,
        sample_rate=100,
        title='Non-Atrial Fibrillation: ECG 12',
        # lead_index=lead_index,
        lead_order=None,
        style=None,
        columns=1,
        row_height=6,
        show_lead_name=True,
        show_grid=True,
        show_separate_line=True,
        speed=50,
        voltage=20
):
    if not lead_order:
        lead_order = list(range(0, len(ecg)))
    secs = len(ecg[0]) / sample_rate
    leads = len(lead_order)
    rows = 1
    display_factor = 1
    line_width = 0.5
    fig, ax = plt.subplots(figsize=(10, 1.2))
    display_factor = display_factor ** 0.25
    fig.subplots_adjust(
        hspace=0,
        wspace=0,
        left=0,  # the left side of the subplots of the figure
        right=1,  # the right side of the subplots of the figure
        bottom=0,  # the bottom of the subplots of the figure
        top=1
    )
    fig.suptitle(title)
    x_min = 0
    x_max = columns * secs
    y_min = row_height / 4 - (rows / 2) * row_height
    y_max = row_height / 4

    if (style == 'bw'):
        color_major = (0.4, 0.4, 0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line = (0, 0, 0)
    else:
        color_major = (1, 0, 0)
        color_minor = (1, 0.7, 0.7)
        color_line = (0, 0, 0.7)

    if (show_grid):
        ax.set_yticks(np.arange(y_min, y_max, 0.5))
        x_ticks_major = np.arange(0, 10, 0.2)
        x_ticks_minor = np.arange(0, 1, 0.04)  # Subdivide each major interval into five parts

        ax.set_xticks(x_ticks_major)
        ax.set_xticks(x_ticks_minor, minor=True)

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # Set minor ticks

        x_tick_labels_major = [f"{i}" if idx % 5 == 0 else "" for idx, i in enumerate(x_ticks_major)]
        ax.set_xticklabels(x_tick_labels_major, fontsize=10)
        y = np.arange(y_min, y_max, 0.5)
        ax.set_yticklabels(y, fontsize=10)

        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    t_lead = lead_order[1]

    step = 1.0 / sample_rate
    if (show_lead_name):
        ax.text(0.07, -0.5, lead_index[t_lead], fontsize=9 * display_factor)
    ax.plot(
        np.arange(0, len(ecg[t_lead]) * step, step),
        ecg[t_lead],
        linewidth=1,
        color='black'
    )
    ax.set_xlabel('Time(s)')  # Label for x-axis
    ax.set_ylabel('Value(mV)')  # Label for y-axis
    plt.savefig("{}/{}".format('/home/gaoshaochen/data', 'test') + '.svg',
                bbox_inches='tight')


def show():
    plt.show()


data_Atrain = np.load('/home/chenpeng/workspace/dataset/CSPC2021_fanc/cpsc_fangchan.npy')
data = data_Atrain[0]
plot_new(np.array(data))

show()
