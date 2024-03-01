from math import ceil
from matplotlib import pyplot as plt
import numpy as np
from scipy.fftpack import fft
from common.utils import is_debug_environment
from matplotlib.ticker import AutoMinorLocator
import os

figure_number = 1

###########################################################################

def plot_time_signals(t, sig):
    if not is_debug_environment():
        return
    global figure_number
    plt.figure(str(figure_number))
    figure_number += 1
    plt.subplot(1, 1, 1)
    plt.plot(t, sig["data"])
    plt.title(sig["title"])
    plt.xlabel(sig["xlabel"])
    plt.ylabel(sig["ylabel"])
    plt.grid(True)

def plot_time_signals_multiplot(t, *args):
    if not is_debug_environment():
        return
    global figure_number
    plt.figure(str(figure_number))
    figure_number += 1
    for i in range(len(args)):
        plt.subplot(len(args), 1, i + 1)
        plt.plot(t, args[i]["data"])
        plt.title(args[i]["title"])
        plt.xlabel(args[i]["xlabel"])
        plt.ylabel(args[i]["ylabel"])
        plt.grid(True)

def plot_time_signals_multiline(t, *args):
    if not is_debug_environment():
        return
    global figure_number
    plt.figure(str(figure_number))
    figure_number += 1
    plt.subplot(1, 1, 1)
    for i in range(len(args)):
        plt.plot(t, args[i]["data"], label=args[i]["title"])
    plt.grid(True)
    plt.legend()

def plot_time_signals_array_multiplot(t, signals):
    if not is_debug_environment():
        return
    global figure_number
    plt.figure(str(figure_number))
    figure_number += 1
    for i in range(len(signals)):
        plt.subplot(len(signals), 1, i + 1)
        plt.plot(t, signals[i]["data"])
        plt.title(signals[i]["title"])
        plt.xlabel(signals[i]["xlabel"])
        plt.ylabel(signals[i]["ylabel"])
        plt.grid(True)

def plot_time_signals_array_multiline(t, signals):
    if not is_debug_environment():
        return
    global figure_number
    plt.figure(str(figure_number))
    figure_number += 1
    plt.subplot(1, 1, 1)
    for i in range(len(signals)):
        plt.plot(t, signals[i]["data"], label=signals[i]["title"])
    plt.grid(True)
    plt.legend()

# Amplitude spectrum
def plot_amplitude_freq_spectrum(signal):
    if not is_debug_environment():
        return
    global figure_number
    plt.figure(str(figure_number))
    figure_number += 1
    nfft = len(signal["data"])
    A = fft(signal["data"], nfft)
    A = A[:nfft // 2]
    m = np.abs(A)
    f = np.arange(nfft // 2) * signal["sampling_freq"] / nfft
    plt.subplot(1, 1, 1)
    plt.plot(f, m)
    plt.title(signal["title"])
    plt.xlabel(signal["xlabel"])
    plt.ylabel(signal["ylabel"])
    plt.grid(True)

def plot_custom_ecg_qrs(ecg, title, r_peak_locations, q_locations, s_locations, p_locations=None):
    if not is_debug_environment():
        return
    global figure_number
    plt.figure(str(figure_number))
    figure_number += 1
    plt.subplot(1, 1, 1)
    plt.plot(ecg)
    plt.title(title)
    plt.title(title)
    plt.plot(r_peak_locations, ecg[r_peak_locations], '*k')
    plt.plot(q_locations, ecg[q_locations], '^k')
    plt.plot(s_locations, ecg[s_locations], 'o')
    # if t_locations is not None:
    #     plt.plot(t_locations, ecg[t_locations], 'v')
    if p_locations is not None:
        plt.plot(p_locations, ecg[p_locations], 'p')
    plt.grid(True)


def plot_custom_ecg_pqrst(ecg, title, p_locations, q_locations, r_locations, s_locations, t_locations):
    if not is_debug_environment():
        return
    global figure_number
    plt.figure(str(figure_number))
    figure_number += 1
    plt.subplot(1, 1, 1)
    plt.plot(ecg)
    plt.title(title)
    plt.title(title)
    plt.plot(r_locations, ecg[r_locations], '*k')
    plt.plot(q_locations, ecg[q_locations], '^k')
    plt.plot(s_locations, ecg[s_locations], 'o')
    plt.plot(t_locations, ecg[t_locations], '*b')
    plt.plot(p_locations, ecg[p_locations], '^y')
    plt.grid(True)

def plot_ppg_waveforms(
        bp_waveform,
        waveform_dd,
        bp_integral,
        threshold,
        zone_of_interest,
        foot_index,
        systolic_index,
        dicrotic_index,
        notch_index,
        time):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(time, bp_waveform)
    axs[0].plot(time[foot_index], bp_waveform[foot_index], '<', color='b', markerfacecolor='b')
    axs[0].plot(time[systolic_index], bp_waveform[systolic_index], '^', color='g', markerfacecolor='g')
    axs[0].plot(time[notch_index], bp_waveform[notch_index], '^', color='r', markerfacecolor='r')
    axs[0].plot(time[dicrotic_index], bp_waveform[dicrotic_index], '^', color='c', markerfacecolor='c')
    axs[0].legend(['Filtered', 'Foot', 'Systole', 'Notch', 'Dicrotic Peak'], loc='upper right')
    axs[0].set_ylabel('Arterial Pressure')

    axs[1].plot(time, waveform_dd)
    axs[1].plot(time, bp_integral)
    axs[1].plot(time, threshold)
    axs[1].plot(time, zone_of_interest * 0.1)
    axs[1].legend(['2nd Derivative', 'Integral', 'Threshold', 'ZOI'], loc='upper right')
    axs[1].set_xlabel('Time (s)')
    plt.grid(True)

def show_plots():
    if not is_debug_environment():
        return
    plt.show()

def close_plots():
    if not is_debug_environment():
        return
    plt.close('all')

def _ax_plot(ax, x, y, secs=10, lwidth=0.5, amplitude_ecg = 1.8, time_ticks =0.2):
    ax.set_xticks(np.arange(0,11,time_ticks))
    ax.set_yticks(np.arange(-ceil(amplitude_ecg),ceil(amplitude_ecg),1.0))

    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    ax.minorticks_on()

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_ylim(-amplitude_ecg, amplitude_ecg)
    ax.set_xlim(0, secs)

    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))

    ax.plot(x,y, linewidth=lwidth)

lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# def plot_12(
#         ecg,
#         sample_rate = 100,
#         title       = 'ECG 12',
#         lead_index  = lead_index,
#         lead_order  = None,
#         columns     = 1,
#         speed = 50,
#         voltage = 20,
#         line_width = 0.6
#         ):
#     """Plot multi lead ECG chart.
#     # Arguments
#         ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
#         sample_rate: Sample rate of the signal.
#         title      : Title which will be shown on top off chart
#         lead_index : Lead name array in the same order of ecg, will be shown on
#             left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#         lead_order : Lead display order
#         columns    : display columns, defaults to 2
#         speed      : signal speed on display, defaults to 50 mm / sec
#         voltage    : signal voltage on display, defaults to 20 mm / mV
#         line_width : line width, default to 0.6
#     """
#     if not lead_order:
#         lead_order = list(range(0,len(ecg)))

#     leads = len(lead_order)
#     seconds = len(ecg[0])/sample_rate

#     plt.rcParams.update({'font.size': 8})
#     fig, ax = plt.subplots(
#         ceil(len(lead_order)/columns),columns,
#         sharex=True,
#         sharey=True,
#         figsize=((speed/25)*seconds*columns,    # 1 inch= 25,4 mm. Rounded to 25 for simplicity
#             (4.1*voltage/25)*leads/columns)     # 1 subplot usually contains values in range of (-2,2) mV
#         )
#     fig.subplots_adjust(
#         hspace = 0,
#         wspace = 0.04,
#         left   = 0.04,  # the left side of the subplots of the figure
#         right  = 0.98,  # the right side of the subplots of the figure
#         bottom = 0.06,  # the bottom of the subplots of the figure
#         top    = 0.95
#         )
#     fig.suptitle(title)

#     step = 1.0/sample_rate

#     for i in range(0, len(lead_order)):
#         if(columns == 1):
#             t_ax = ax[i]
#         else:
#             t_ax = ax[i//columns,i%columns]
#         t_lead = lead_order[i]
#         t_ax.set_ylabel(lead_index[t_lead])
#         t_ax.tick_params(axis='x',rotation=90)

#         _ax_plot(t_ax, np.arange(0, len(ecg[t_lead])*step, step), ecg[t_lead], seconds)

# def plot_12(
#         ecg,
#         sample_rate    = 100,
#         title          = None,
#         lead_index     = lead_index,
#         lead_order     = None,
#         style          = None,
#         columns        = 1,
#         row_height     = 12,
#         show_lead_name = True,
#         show_grid      = True,
#         show_separate_line  = True,
#         ):
#     """Plot multi lead ECG chart.
#     # Arguments
#         ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
#         sample_rate: Sample rate of the signal.
#         title      : Title which will be shown on top off chart
#         lead_index : Lead name array in the same order of ecg, will be shown on
#             left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#         lead_order : Lead display order
#         columns    : display columns, defaults to 2
#         style      : display style, defaults to None, can be 'bw' which means black white
#         row_height :   how many grid should a lead signal have,
#         show_lead_name : show lead name
#         show_grid      : show grid
#         show_separate_line  : show separate line
#     """

#     if not lead_order:
#         lead_order = list(range(0,len(ecg)))
#     secs  = len(ecg[0])/sample_rate
#     leads = len(lead_order)
#     rows  = int(ceil(leads/columns))
#     # display_factor = 2.5
#     display_factor = 1
#     line_width = 1.0
#     fig, ax = plt.subplots(figsize=(secs*columns * display_factor, rows * row_height / 10 * display_factor))
#     display_factor = display_factor ** 0.5
#     fig.subplots_adjust(
#         hspace = 0,
#         wspace = 0,
#         left   = 0,  # the left side of the subplots of the figure
#         right  = 1,  # the right side of the subplots of the figure
#         bottom = 0,  # the bottom of the subplots of the figure
#         top    = 1
#         )

#     fig.suptitle(title)

#     x_min = 0
#     x_max = columns*secs
#     y_min = row_height/4 - (rows/2)*row_height
#     y_max = row_height/4

#     if (style == 'bw'):
#         color_major = (0.4,0.4,0.4)
#         color_minor = (0.75, 0.75, 0.75)
#         color_line  = (0,0,0)
#     else:
#         color_major = (1,0,0)
#         color_minor = (1, 0.7, 0.7)
#         color_line  = (0,0,0)

#     if(show_grid):
#         ax.set_xticks(np.arange(x_min,x_max,0.2))
#         ax.set_yticks(np.arange(y_min,y_max,0.5))

#         ax.set_xticklabels([])
#         ax.set_yticklabels([])

#         ax.minorticks_on()

#         ax.xaxis.set_minor_locator(AutoMinorLocator(5))

#         ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
#         ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

#     ax.set_ylim(-30.5, 1.5)
#     ax.set_xlim(x_min,x_max)


#     for c in range(0, columns):
#         for i in range(0, rows):
#             if (c * rows + i < leads):
#                 y_offset = -(row_height/4) * ceil(i%rows)
#                 if (y_offset < 0):
#                     y_offset = y_offset + i/2


#                 x_offset = 0
#                 if(c > 0):
#                     x_offset = secs * c
#                     if(show_separate_line):
#                         ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line)


#                 t_lead = lead_order[c * rows + i]

#                 step = 1.0/sample_rate
#                 # print(ecg[t_lead])
#                 # print((ecg[t_lead]*10000000.0) + y_offset)
#                 if(show_lead_name):
#                     ax.text(x_offset + 0.07, y_offset + 0.5, lead_index[t_lead], fontsize=10 * display_factor, fontweight='bold', fontfamily='serif')
#                 ax.plot(
#                     np.arange(0, len(ecg[t_lead])*step, step) + x_offset,
#                     (ecg[t_lead]*10000000.0) + y_offset,
#                     linewidth=line_width * display_factor,
#                     color=color_line
#                     )

def plot_12(
        ecg,
        sample_rate    = 100,
        title          = None,
        lead_index     = lead_index,
        lead_order     = None,
        style          = None,
        columns        = 4,
        row_height     = 3,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order
        columns    : display columns, defaults to 2
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """

    if not lead_order:
        lead_order = list(range(0,len(ecg)))
    secs  = len(ecg[0])/sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    # display_factor = 2.5
    display_factor = 1
    line_width = 1.0
    fig, ax = plt.subplots(figsize=(secs*columns * display_factor, rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0,
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title)

    x_min = 0
    x_max = columns*secs
    y_min = row_height/4 - (rows/2)*row_height
    y_max = row_height/1.5

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (1,0,0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0,0,0)

    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.2))
        ax.set_yticks(np.arange(y_min,y_max,0.5))

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    # ax.set_ylim(-30.5, 1.5)
    # ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)

    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows)
                if (y_offset == 0):
                    y_offset = y_offset + 0.15
                elif (y_offset < 0):
                    y_offset = y_offset + 0.05

                x_offset = 0
                if(c > 0):
                    x_offset = secs * c
                    if(show_separate_line):
                        ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line)

                t_lead = lead_order[c * rows + i]

                step = 1.0/sample_rate
                if(show_lead_name):
                    ax.text(x_offset + 0.07, y_offset - 0.3, lead_index[t_lead], fontsize=9 * display_factor)
                ax.plot(
                    np.arange(0, len(ecg[t_lead])*step, step) + x_offset,
                    ecg[t_lead] + y_offset,
                    linewidth=line_width * display_factor,
                    color=color_line
                    )

    # for c in range(0, columns):
    #     for i in range(0, rows):
    #         if (c * rows + i < leads):
    #             y_offset = -(row_height/4) * ceil(i%rows)
    #             if (y_offset < 0):
    #                 y_offset = y_offset + i/2

    #             x_offset = 0
    #             if(c > 0):
    #                 x_offset = secs * c
    #                 if(show_separate_line):
    #                     ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line)

    #             t_lead = lead_order[c * rows + i]

    #             step = 1.0/sample_rate
    #             # print(ecg[t_lead])
    #             # print((ecg[t_lead]*10000000.0) + y_offset)
    #             if(show_lead_name):
    #                 ax.text(x_offset + 0.07, y_offset + 0.5, lead_index[t_lead], fontsize=10 * display_factor, fontweight='bold', fontfamily='serif')
    #             ax.plot(
    #                 np.arange(0, len(ecg[t_lead])*step, step) + x_offset,
    #                 (ecg[t_lead]*10000000.0) + y_offset,
    #                 linewidth=line_width * display_factor,
    #                 color=color_line
    #                 )

def plot_6(
        ecg,
        sample_rate    = 100,
        title          = None,
        lead_index     = lead_index,
        lead_order     = None,
        style          = None,
        columns        = 2,
        row_height     = 3,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order
        columns    : display columns, defaults to 2
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """

    if not lead_order:
        lead_order = list(range(0,len(ecg)))
    secs  = len(ecg[0])/sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    # display_factor = 2.5
    display_factor = 1
    line_width = 1.0
    fig, ax = plt.subplots(figsize=(secs*columns * display_factor, rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0,
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title)

    x_min = 0
    x_max = columns*secs
    y_min = row_height/4 - (rows/2)*row_height
    y_max = row_height/2

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (1,0,0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0,0,0)

    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.24))
        ax.set_yticks(np.arange(y_min,y_max,1.0))

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    # ax.set_ylim(-16.5, 1.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min,x_max)


    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows)
                if (y_offset == 0):
                    y_offset = y_offset + 0.15
                elif (y_offset < 0):
                    y_offset = y_offset + 0.05
                # y_offset = -(row_height/2) * ceil(i%rows)
                # # if (y_offset < 0):
                # #     y_offset = y_offset + i/2


                x_offset = 0
                if(c > 0):
                    x_offset = secs * c
                    if(show_separate_line):
                        ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line)


                t_lead = lead_order[c * rows + i]

                step = 1.0/sample_rate
                # print(ecg[t_lead])
                # print((ecg[t_lead]*10000000.0) + y_offset)
                if(show_lead_name):
                    ax.text(x_offset + 0.07, y_offset - 0.3, lead_index[t_lead], fontsize=9 * display_factor, fontfamily='serif')
                ax.plot(
                    np.arange(0, len(ecg[t_lead])*step, step) + x_offset,
                    (ecg[t_lead]) + y_offset,
                    linewidth=line_width * display_factor,
                    color=color_line
                    )

def plot_1(ecg,
        sample_rate    = 104,
        title          = None,
        lead_index     = lead_index,
        lead_order     = None,
        style          = None,
        columns        = 1,
        row_height     = 1,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        fig_width  : The width of the plot
        fig_height : The height of the plot
    """
    if not lead_order:
        lead_order = list(range(0,len(ecg)))
    secs  = len(ecg[0])/sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    # display_factor = 2.5
    display_factor = 1
    line_width = 1.0
    fig, ax = plt.subplots(figsize=(secs*2 * display_factor, 3 * 3 / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0,
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title)

    x_min = 0
    x_max = columns*secs
    y_min = row_height/2 - (rows)*row_height
    y_max = row_height/2

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (1,0,0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0,0,0)

    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.12))
        ax.set_yticks(np.arange(y_min,y_max,0.1))

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min,x_max)


    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):

                t_lead = lead_order[c * rows + i]

                step = 1.0/sample_rate
                # print(ecg[t_lead])
                # print((ecg[t_lead]*10000000.0) + y_offset)
                if(show_lead_name):
                    ax.text(0.05, 0.1, lead_index[t_lead], fontsize=9 * display_factor, fontfamily='serif')
                ax.plot(
                    np.arange(0, len(ecg[t_lead])*step, step),
                    (ecg[t_lead]*2000000.0),
                    linewidth=line_width * display_factor,
                    color=color_line
                    )
    # _ax_plot(ax,np.arange(0,len(ecg)*step,step), ecg, seconds, line_w, ecg_amp,timetick)

default_path = './images/'
# os.path.join(os.getcwd(), "images")
# if not os.path.exists(default_path):
#     os.mkdir(default_path)

def save_as_png(file_name, path = default_path, dpi = 100, layout='tight'):
    """Plot multi lead ECG chart.
    # Arguments
        file_name: file_name
        path     : path to save image, defaults to current folder
        dpi      : set dots per inch (dpi) for the saved image
        layout   : Set equal to "tight" to include ax labels on saved image
    """
    plt.ioff()
    plt.savefig(path + file_name + '.png', dpi = dpi, bbox_inches=layout)
    plt.close()
