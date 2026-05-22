import numpy as np
import pandas as pd


def get_isi(spikes, t_start, t_end):
    # restrict to time window 
    spikes = [sk for sk in spikes if t_start <= sk <= t_end]

    if len(spikes) < 2:
        # do something 
        return np.nan

    # compute ISIs
    spikes = np.sort(spikes) # ensure sorted
    isi = np.diff(spikes)
    return isi

def _safe_stat(x, fn):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    arr = np.asarray(x)
    return fn(arr) if arr.size > 0 else np.nan

def compute_isi_stats(spike_df, t_start, t_end, sample_stats=True):
    """
    spike_df: DataFrame indexed by condition (rows) with columns=skid.
             Each cell contains a list/array of spike times.

    Returns:
      stats_df: long df with columns [cond, skid, isi_mean, isi_var, isi_std, isi_cv, isi_skew, n_isis]
      isis: DataFrame of arrays (same shape as spike_df) containing ISIs per cell
    """
    ddof = 1 if sample_stats else 0

    # Compute ISI arrays per (cond, skid)
    isis = spike_df.map(lambda spks: get_isi(spks, t_start, t_end))

    # Metric tables (wide: cond x skid)
    isi_mean = isis.map(lambda x: _safe_stat(x, np.mean))
    isi_var  = isis.map(lambda x: _safe_stat(x, lambda a: np.var(a, ddof=ddof) if a.size > ddof else np.nan))
    isi_std  = isis.map(lambda x: _safe_stat(x, lambda a: np.std(a, ddof=ddof) if a.size > ddof else np.nan))

    def _cv(a):
        mu = np.mean(a)
        if not np.isfinite(mu) or mu <= 0:
            return np.nan
        sd = np.std(a, ddof=ddof) if a.size > ddof else np.nan
        return sd / mu if np.isfinite(sd) else np.nan

    isi_cv   = isis.map(lambda x: _safe_stat(x, _cv))
    isi_skew = isis.map(lambda x: _safe_stat(x, lambda a: pd.Series(a).skew() if a.size >= 3 else np.nan))
    n_isis   = isis.map(lambda x: _safe_stat(x, lambda a: int(a.size)))

    # Stack into ONE long df: (cond, skid) index, metric columns
    stats_df = pd.concat(
        {
            "isi_mean": isi_mean,
            "isi_var": isi_var,
            "isi_std": isi_std,
            "isi_cv": isi_cv,
            "isi_skew": isi_skew,
            "n_isis": n_isis,
        },
        axis=1,
    )

    # Move metrics from first level of columns into real columns
    stats_df = (
        stats_df.stack()                 # index: (cond, skid), columns: metrics
                .rename_axis(["cond", "skid"])
                .reset_index()
    )

    return stats_df, isis


def get_fano_factor(spikes, t_start, t_end, window_size_ms):
    # restrict to time window 
    spikes = [sk for sk in spikes if t_start <= sk <= t_end]

    if len(spikes) == 0:
        return np.nan

    # compute spike counts in windows
    bins = np.arange(t_start, t_end + window_size_ms, window_size_ms)
    counts, _ = np.histogram(spikes, bins=bins)

    if counts.size < 2:
        return np.nan

    mean_count = np.mean(counts)
    var_count = np.var(counts, ddof=1) # sample variance
    fano_factor = var_count / mean_count if mean_count > 0 else np.nan
    return fano_factor

def get_fano_factor_for_df(spike_df, t_start, t_end, window_size_ms):
    return spike_df.map(lambda spks: get_fano_factor(spks, t_start, t_end, window_size_ms))
