import math
import numpy as np
from scipy.signal import find_peaks
from app.algos.utilities.plotter import plot_custom_ecg_pqrst
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis

###########################################################################

def find_t_peaks(sig, r_locs, s_locs, fs):

    # Initialize T with zeros
    t_locs_temp = np.copy(s_locs)
    search_offset_t = int(np.mean(np.diff(r_locs)) / 2) if len(r_locs) > 1 else int(0.1 * fs)
    t_locs = np.zeros(t_locs_temp.shape, dtype=int)
    sig_length = len(sig)
    for i, t1 in enumerate(t_locs_temp):
        if t1 >= sig_length or t1 + search_offset_t >= sig_length:
            t_locs[i] = t1 + np.argmax(sig[t1:])
        else:
            t_locs[i] = t1 + np.argmax(sig[t1:t1 + search_offset_t])

    t_locs = np.unique(t_locs)
    t_locs_int = t_locs.astype(int)

    return t_locs_int

###########################################################################

def find_p_peaks(sig, q_locs, r_locs, fs):

    pf_locs_temp = np.copy(q_locs)
    pf_locs = np.zeros(pf_locs_temp.shape, dtype=int)
    search_offset_p = int(np.mean(np.diff(r_locs)) / 3) if len(r_locs) > 1 else int(0.1 * fs)

    for i, pf1 in enumerate(pf_locs_temp):
        if pf1 - search_offset_p <= 0:
            pf_locs[i] = pf1 - np.argmax(sig[:pf1])
        else:
            pf_locs[i] = pf1 - np.argmax(sig[pf1 - search_offset_p:pf1])

    # Adaptive Thresholding
    # t1 = sig[r_locs] / 10
    # t2 = sig[r_locs] / 4

    # for i in range(min(len(pf_locs), len(r_locs))):
    #     if sig[pf_locs[i]] > t2[i]:
    #         t2[i] = t2[i] * 3.3
    #     elif sig[pf_locs[i]] < t1[i]:
    #         t1[i] = t1[i] / 2

    # flag_p_final = np.logical_or(sig[pf_locs] < t1, sig[pf_locs] > t2)
    # pf_locs = np.delete(pf_locs, np.where(flag_p_final))

    P_loc = np.unique(pf_locs)
    p_locs_int = P_loc.astype(int)

    return p_locs_int

###########################################################################

def get_qrst_cycles(r_locs, q_locs, s_locs, fs):
    if len(r_locs) > len(s_locs) + 1:
        print("Not enough S or T peaks found")
        return None, None
    RIGHT_WINDOW = 12
    cycles = []
    rr_diifs = np.diff(r_locs)
    for i in range(len(rr_diifs)):
        start = r_locs[i]
        end = r_locs[i + 1]
        if (s_locs[i] > end):
            s_locs = np.insert(s_locs, 0, 0)
            continue;
        rs_diff = s_locs[i] - r_locs[i]
        if (rs_diff > 0.3 * rr_diifs[i]):
            continue;
        if rs_diff > 2 * RIGHT_WINDOW:
            continue;
        if (s_locs[i] < start):
            s_locs = s_locs[1:] # Pop the first one
        if len(s_locs) <= i:
            break

        x = {
            'q' : int(q_locs[i]),
            'r' : int(r_locs[i]),
            's' : int(s_locs[i]),
        }
        cycles.append(x)

    dur_qr = np.zeros(len(cycles))
    dur_rs = np.zeros(len(cycles))
    dur_qrs = np.zeros(len(cycles))

    for i in range(len(cycles)):
        x = cycles[i]
        dur_qr[i]  = x['r'] - x['q']
        dur_rs[i]  = x['s'] - x['r']
        dur_qrs[i] = x['s'] - x['q']

    duration_qr  = np.mean(dur_qr) * (1 / fs)
    duration_rs  = np.mean(dur_rs) * (1 / fs)
    duration_qrs = np.mean(dur_qrs) * (1 / fs)

    duration = {
        'qr' : duration_qr,
        'rs' : duration_rs,
        'qrs': duration_qrs
    }
    return duration, cycles

###########################################################################

def find_qs_locations(sig, r_locs):
    # Initialize Q and S with zeros
    q_locs = np.zeros(len(r_locs))
    s_locs = np.zeros(len(r_locs))

    # Find minima on the left and right of R peaks
    #KK: Please check these windows,..Original values are commented out
    LEFT_WINDOW = 16  # 8   # Changed windows values for smoothened signal
    RIGHT_WINDOW = 12 # 10  # Original values commented out

    for i in range(len(r_locs)):
        q_locs[i] = r_locs[i] - np.argmin(sig[r_locs[i]::-1][:LEFT_WINDOW]) + 1
        s_locs[i] = r_locs[i] + np.argmin(sig[r_locs[i]:r_locs[i] + RIGHT_WINDOW]) - 1

    # Convert to int
    # Q and S locations are returned as floats, they need to be indices
    q_locs_int = q_locs.astype(int)
    s_locs_int = s_locs.astype(int)

    return q_locs_int, s_locs_int

###########################################################################

def find_rr_intervals_heart_rate(sig, fs):
    mean_ecg = np.mean(sig)
    std_ecg = np.std(sig)
    peak_height = mean_ecg + 2.5 * std_ecg
    min_peak_distance = math.ceil(0.4 * fs)

    #ref : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    r_locs, _ = find_peaks(sig, height=peak_height, distance=min_peak_distance)

    return r_locs

###########################################################################

def insert_missing(p_locs, q_locs, r_locs, s_locs, t_locs):

    if len(p_locs) == len(q_locs) and len(q_locs) == len(r_locs) and len(r_locs) == len(s_locs) and len(s_locs) == len(t_locs):
        return p_locs, q_locs, r_locs, s_locs, t_locs

    maximum = max(len(p_locs), len(q_locs), len(r_locs), len(s_locs), len(t_locs))
    minimum = min(len(p_locs), len(q_locs), len(r_locs), len(s_locs), len(t_locs))

    print('max = ', maximum)
    print('min = ', minimum)

    for i in range(maximum):
        pi = p_locs[i] if i < len(p_locs) else -1
        qi = q_locs[i] if i < len(q_locs) else -1
        ri = r_locs[i] if i < len(r_locs) else -1
        si = s_locs[i] if i < len(s_locs) else -1
        ti = t_locs[i] if i < len(t_locs) else -1

        pi_nx = p_locs[i+1] if i+1 < len(p_locs) else -1

        if pi > qi and pi != -1 and qi != -1:
            p_locs = np.insert(p_locs, i, -1)
        elif qi == -1 and ri != -1 and pi > ri :
            p_locs = np.insert(p_locs, i, -1)

        if qi > ri and qi != -1 and ri != -1:
            q_locs = np.insert(q_locs, i, -1)
        elif ri == -1 and si != -1 and qi > si :
            q_locs = np.insert(q_locs, i, -1)

        if ri > si and ri != -1 and si != -1:
            r_locs = np.insert(r_locs, i, -1)
        elif si == -1 and ti != -1 and ri > ti :
            r_locs = np.insert(r_locs, i, -1)

        if si > ti and si != -1 and ti != -1:
            s_locs = np.insert(s_locs, i, -1)
        elif ti == -1 and pi_nx != -1 and si > pi_nx :
            s_locs = np.insert(s_locs, i, -1)

        if ti > pi_nx and ti != -1 and pi_nx != -1:
            t_locs = np.insert(t_locs, i+1, -1)

    print(p_locs)
    print(q_locs)
    print(r_locs)
    print(s_locs)
    print(t_locs)

    return p_locs,q_locs,r_locs,s_locs,t_locs

############################################################################################################

def identify_clusters(p_locs, q_locs, r_locs, s_locs, t_locs):
    sum = len(p_locs) + len(q_locs) + len(r_locs) + len(s_locs) + len(t_locs)
    sum_new = 0
    while sum_new != sum:
        sum = sum_new
        p_locs, q_locs, r_locs, s_locs, t_locs = insert_missing(p_locs, q_locs, r_locs, s_locs, t_locs)
        sum_new = min(len(p_locs), len(q_locs), len(r_locs), len(s_locs), len(t_locs))

    maximum = max(len(p_locs), len(q_locs), len(r_locs), len(s_locs), len(t_locs))
    if len(p_locs) < maximum:
        p_locs = np.insert(p_locs, len(p_locs), -1)
    if len(q_locs) < maximum:
        q_locs = np.insert(q_locs, len(q_locs), -1)
    if len(r_locs) < maximum:
        r_locs = np.insert(r_locs, len(r_locs), -1)
    if len(s_locs) < maximum:
        s_locs = np.insert(s_locs, len(s_locs), -1)
    if len(t_locs) < maximum:
        t_locs = np.insert(t_locs, len(t_locs), -1)

    print('-------------------')
    print(p_locs)
    print(q_locs)
    print(r_locs)
    print(s_locs)
    print(t_locs)

    maximum = max(len(p_locs), len(q_locs), len(r_locs), len(s_locs), len(t_locs))

    pqrst = []
    qrs = []

    for i in range(maximum):
        pi = p_locs[i]
        qi = q_locs[i]
        ri = r_locs[i]
        si = s_locs[i]
        ti = t_locs[i]

        if pi != -1 and qi != -1 and ri != -1 and si != -1 and ti != -1:
            pqrst.append([pi, qi, ri, si, ti])

        if qi != -1 and ri != -1 and si != -1:
            qrs.append([qi, ri, si])

    return pqrst, qrs

############################################################################################################

def get_intervals(sig, r_locs, pqrst_locs, qrs_locs, fs):

    # RR Interval
    rr_period = np.diff(r_locs) / fs
    mean_rr   = np.mean(rr_period)
    std_rr    = np.std(rr_period)
    skew_rr   = skew(rr_period)
    kurt_rr   = kurtosis(rr_period)

    # QRS Complex Period
    qrs_period = np.zeros(len(qrs_locs))
    if len(qrs_locs) > 0:
        for i in range(len(qrs_locs)):
            qrs_period[i] = qrs_locs[i][2] - qrs_locs[i][0] / fs
        mean_qrs = np.mean(qrs_period)
        std_qrs  = np.std(qrs_period)
        skew_qrs = skew(qrs_period)
        kurt_qrs = kurtosis(qrs_period)
    else:
        mean_qrs = 0
        std_qrs  = 0
        skew_qrs = 0
        kurt_qrs = 0

    # PR Interval
    pr_period = np.zeros(len(pqrst_locs))
    if len(pqrst_locs) > 0:
        for i in range(len(pqrst_locs)):
            pr_period[i] = pqrst_locs[i][2] - pqrst_locs[i][0] / fs
        mean_pr = np.mean(pr_period)
        std_pr  = np.std(pr_period)
        skew_pr = skew(pr_period)
        kurt_pr = kurtosis(pr_period)
    else:
        mean_pr = 0
        std_pr  = 0
        skew_pr = 0
        kurt_pr = 0

    return {
        'mean_rr' : round(mean_rr),
        'mean_qrs': round(mean_qrs),
        'mean_pr' : round(mean_pr),
        # 'std_rr'  : std_rr,
        # 'skew_rr' : skew_rr,
        # 'kurt_rr' : kurt_rr,
        # 'std_qrs' : std_qrs,
        # 'skew_qrs': skew_qrs,
        # 'kurt_qrs': kurt_qrs,
        # 'std_pr'  : std_pr,
        # 'skew_pr' : skew_pr,
        # 'kurt_pr' : kurt_pr
    }

############################################################################################################

def find_ecg_intervals(ecg, fs):
    mean_ecg = np.mean(ecg)
    std_ecg = np.std(ecg)
    peak_height = mean_ecg + 2.5 * std_ecg
    min_peak_distance = math.ceil(0.4 * fs)

    r_locs, _ = find_peaks(ecg, height=peak_height, distance=min_peak_distance)
    if len(r_locs) < 2:
        print("Not enough R peaks found")
        return None, None

    # Q locations
    q_locs, s_locs = find_qs_locations(ecg, r_locs)
    t_locs = find_t_peaks(ecg, r_locs, s_locs, fs)
    p_locs = find_p_peaks(ecg, q_locs, r_locs, fs)

    plot_custom_ecg_pqrst(ecg, 'ECG', p_locs, q_locs, r_locs, s_locs, t_locs)

    # Insert missing peaks
    pqrst_locs, qrs_locs = identify_clusters(p_locs, q_locs, r_locs, s_locs, t_locs)
    intervals = get_intervals(ecg, r_locs, pqrst_locs, qrs_locs, fs)

    print("PQRST locations = ", pqrst_locs)
    print("QRS locations = ", qrs_locs)

    print("R locations = ", r_locs)
    print("Q locations = ", q_locs)
    print("S locations = ", s_locs)
    print("T locations = ", t_locs)
    print("P locations = ", p_locs)

    print("Intervals = ", intervals)

    # duration, cycles = get_qrst_cycles(r_locs, q_locs, s_locs, fs)
    # print("QRST cycles = ", cycles)
    # print("Duration = ", duration)

    return intervals, pqrst_locs

###########################################################################

# def compute_mean_ecg_intervals(ecg, p_locs, q_locs, r_locs, s_locs, t_locs, fs):
#     # RR Interval
#     rr_period = np.diff(r_locs) / fs
#     mean_rr   = np.mean(rr_period)
#     std_rr    = np.std(rr_period)
#     skew_rr   = skew(rr_period)
#     kurt_rr   = kurtosis(rr_period)
#     mean_rr   = 0 if np.isnan(mean_rr) else mean_rr
#     std_rr    = 0 if np.isnan(std_rr) else std_rr
#     skew_rr   = 0 if np.isnan(skew_rr) else skew_rr
#     kurt_rr   = 0 if np.isnan(kurt_rr) else kurt_rr

#     q_locs_unique = np.unique(q_locs)
#     s_locs_unique = np.unique(s_locs)

#     # QRS Complex Period
#     if len(q_locs_unique) > 2 and len(s_locs_unique) > 2 and len(q_locs_unique) != len(s_locs_unique):
#         s_locs = s_locs[~np.isnan(interp1d(q_locs_unique, q_locs_unique, kind='nearest')(s_locs_unique))]
#         q_locs = q_locs[~np.isnan(interp1d(s_locs_unique, s_locs_unique, kind='nearest')(q_locs_unique))]

#     if len(q_locs) > len(s_locs):
#         q_locs1 = q_locs[:len(s_locs)]
#         s_locs1 = s_locs
#     else:
#         s_locs1 = s_locs[:len(q_locs)]
#         q_locs1 = q_locs

#     qrs_period = (s_locs1 - q_locs1) / fs
#     mean_qrs   = np.mean(qrs_period)
#     std_qrs    = np.std(qrs_period)
#     kurt_qrs   = kurtosis(qrs_period)
#     skew_qrs   = skew(qrs_period)
#     mean_qrs   = 0 if np.isnan(mean_qrs) else mean_qrs
#     std_qrs    = 0 if np.isnan(std_qrs) else std_qrs
#     skew_qrs   = 0 if np.isnan(skew_qrs) else skew_qrs
#     kurt_qrs   = 0 if np.isnan(kurt_qrs) else kurt_qrs

#     # PR Interval
#     r_locs = r_locs[~np.isnan(interp1d(np.unique(p_locs), np.unique(p_locs), kind='nearest')(np.unique(r_locs)))]
#     p_locs2 = p_locs[~np.isnan(interp1d(np.unique(r_locs), np.unique(r_locs), kind='nearest')(np.unique(p_locs)))]

#     if len(r_locs) > len(p_locs2):
#         r_locs3 = r_locs[:len(p_locs2)]
#         p_locs2 = p_locs2
#     else:
#         p_locs2 = p_locs2[:len(r_locs)]
#         r_locs3 = r_locs

#     pr_interval = (r_locs3 - p_locs2) / fs
#     mean_pr     = np.mean(pr_interval) if len(pr_interval) > 0 else 0
#     std_pr      = np.std(pr_interval)
#     skew_pr     = skew(pr_interval)
#     kurt_pr     = kurtosis(pr_interval)
#     mean_pr     = 0 if np.isnan(mean_pr) else abs(mean_pr)
#     std_pr      = 0 if np.isnan(std_pr) else std_pr
#     kurt_pr     = 0 if np.isnan(kurt_pr) else kurt_pr
#     skew_pr     = 0 if np.isnan(skew_pr) else skew_pr

#     # QT Interval
#     q_locs = q_locs[~np.isnan(interp1d(np.unique(t_locs), np.unique(t_locs), kind='nearest')(np.unique(q_locs)))]
#     t_locs = t_locs[~np.isnan(interp1d(np.unique(q_locs), np.unique(q_locs), kind='nearest')(np.unique(t_locs)))]

#     if len(q_locs) > len(t_locs):
#         q_locs2 = q_locs[:len(t_locs)]
#         t_locs2 = t_locs
#     else:
#         t_locs2 = t_locs[:len(q_locs)]
#         q_locs2 = q_locs

#     qt_interval = (t_locs2 - q_locs2) / fs
#     mean_qt     = np.mean(qt_interval) if len(qt_interval) > 0 else 0
#     std_qt      = np.std(qt_interval)
#     skew_qt     = skew(qt_interval)
#     kurt_qt     = kurtosis(qt_interval)
#     mean_qt     = 0 if np.isnan(mean_qt) else abs(mean_qt)
#     std_qt      = 0 if np.isnan(std_qt) else std_qt
#     kurt_qt     = 0 if np.isnan(kurt_qt) else kurt_qt
#     skew_qt     = 0 if np.isnan(skew_qt) else skew_qt

#     intervals = {
#         'mean_rr' : mean_rr,
#         'mean_qrs': mean_qrs,
#         'mean_qt' : mean_qt,
#         'mean_pr' : mean_pr,
#         'std_rr'  : std_rr,
#         'skew_rr' : skew_rr,
#         'kurt_rr' : kurt_rr,
#         'std_qrs' : std_qrs,
#         'skew_qrs': skew_qrs,
#         'kurt_qrs': kurt_qrs,
#         'std_pr'  : std_pr,
#         'skew_pr' : skew_pr,
#         'kurt_pr' : kurt_pr,
#         'std_qt'  : std_qt,
#         'skew_qt' : skew_qt,
#         'kurt_qt' : kurt_qt
#     }
#     return intervals
#     # return mean_rr, mean_qrs, mean_qt, mean_pr, std_rr, skew_rr, kurt_rr, std_qrs, skew_qrs, kurt_qrs, std_pr, skew_pr, kurt_pr, std_qt, skew_qt, kurt_qt

# ###########################################################################

# def compute_mean_ecg_intervals_c(q_locs, r_locs, s_locs, t_locs, fs):

#     # SS Interval Code
#     ss_int  = np.diff(s_locs) / fs
#     mean_ss = np.mean(ss_int)
#     std_ss  = np.std(ss_int)
#     skew_ss = skew(ss_int)
#     kurt_ss = kurtosis(ss_int)

#     # QR Interval Code
#     r_locs = r_locs[~np.isnan(interp1d(np.unique(q_locs), np.unique(q_locs), kind='nearest')(np.unique(r_locs)))]
#     q_locs = q_locs[~np.isnan(interp1d(np.unique(r_locs), np.unique(r_locs), kind='nearest')(np.unique(q_locs)))]

#     if len(q_locs) > len(r_locs):
#         q_locs = q_locs[:len(r_locs)]
#     elif len(r_locs) > len(q_locs):
#         r_locs = r_locs[:len(q_locs)]

#     qr_int = (r_locs - q_locs) / fs
#     mean_qr = np.mean(qr_int)
#     std_qr = np.std(qr_int)
#     skew_qr = skew(qr_int)
#     kurt_qr = kurtosis(qr_int)

#     # ST Interval Code
#     s_locs = s_locs[~np.isnan(interp1d(np.unique(t_locs), np.unique(t_locs), kind='nearest')(np.unique(s_locs)))]
#     t_locs = t_locs[~np.isnan(interp1d(np.unique(s_locs), np.unique(s_locs), kind='nearest')(np.unique(t_locs)))]

#     if len(t_locs) < len(s_locs):
#         s_locs_f = s_locs[:len(t_locs)]
#         t_locs_f = t_locs
#     elif len(t_locs) > len(s_locs):
#         t_locs_f = t_locs[:len(s_locs)]
#         s_locs_f = s_locs
#     else:
#         t_locs_f = t_locs
#         s_locs_f = s_locs

#     st_int  = (t_locs_f - s_locs_f) / fs
#     mean_st = np.mean(st_int)
#     std_st  = np.std(st_int)
#     skew_st = skew(st_int)
#     kurt_st = kurtosis(st_int)

#     # RS Interval Code
#     r_locs = r_locs[~np.isnan(interp1d(np.unique(s_locs), np.unique(s_locs), kind='nearest')(np.unique(r_locs)))]
#     s_locs = s_locs[~np.isnan(interp1d(np.unique(r_locs), np.unique(r_locs), kind='nearest')(np.unique(s_locs)))]

#     if len(s_locs) < len(r_locs):
#         r_locs = r_locs[:len(s_locs)]
#     elif len(s_locs) > len(r_locs):
#         s_locs = s_locs[:len(r_locs)]

#     rs_int  = (s_locs - r_locs) / fs
#     mean_rs = np.mean(rs_int)
#     std_rs  = np.std(rs_int)
#     skew_rs = skew(rs_int)
#     kurt_rs = kurtosis(rs_int)

#     # Eliminating unwanted possibilities
#     mean_qr = 0 if np.isnan(mean_qr) else mean_qr
#     mean_ss = 0 if np.isnan(mean_ss) else mean_ss
#     mean_st = 0 if np.isnan(mean_st) else mean_st
#     mean_rs = 0 if np.isnan(mean_rs) else mean_rs
#     std_qr  = 0 if np.isnan(std_qr)  else std_qr
#     std_ss  = 0 if np.isnan(std_ss)  else std_ss
#     std_st  = 0 if np.isnan(std_st)  else std_st
#     std_rs  = 0 if np.isnan(std_rs)  else std_rs
#     skew_qr = 0 if np.isnan(skew_qr) else skew_qr
#     skew_ss = 0 if np.isnan(skew_ss) else skew_ss
#     skew_st = 0 if np.isnan(skew_st) else skew_st
#     skew_rs = 0 if np.isnan(skew_rs) else skew_rs
#     kurt_qr = 0 if np.isnan(kurt_qr) else kurt_qr
#     kurt_ss = 0 if np.isnan(kurt_ss) else kurt_ss
#     kurt_st = 0 if np.isnan(kurt_st) else kurt_st
#     kurt_rs = 0 if np.isnan(kurt_rs) else kurt_rs

#     intervals = {
#         'mean_qr': mean_qr,
#         'mean_ss': mean_ss,
#         'mean_st': mean_st,
#         'mean_rs': mean_rs,
#         'std_qr' : std_qr,
#         'std_ss' : std_ss,
#         'std_st' : std_st,
#         'std_rs' : std_rs,
#         'skew_qr': skew_qr,
#         'skew_ss': skew_ss,
#         'skew_st': skew_st,
#         'skew_rs': skew_rs,
#         'kurt_qr': kurt_qr,
#         'kurt_ss': kurt_ss,
#         'kurt_st': kurt_st,
#         'kurt_rs': kurt_rs
#     }
#     return intervals
#     # return (mean_qr, mean_ss, mean_st, mean_rs, std_qr, std_ss, std_st, skew_qr, skew_ss, skew_st, skew_rs, kurt_qr, kurt_ss, kurt_st, kurt_rs, std_rs)

# ###########################################################################
