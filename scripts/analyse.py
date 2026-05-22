from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import time
import ast
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import seaborn as sns

from scripts.plot_networks import build_connectivity_graph, plot_connectivity_graph
from scripts.plotting_activity import plot_spikes
from scripts.isi_metrics_functions import compute_isi_stats, get_fano_factor

@dataclass
class SimRes:
    sim_id: str          # "{file_idx}-{sim_id}-{0=flat,1=full}"
    arch: str            # "flat" or "full"
    spikes: pd.DataFrame
    global_syn_weight: float
    pr: float
    input_frequency_hz: float
    seed: int
    spike_array: Any = field(default=None, init=False)


class SimCollection:
    def __init__(self):
        self._store: Dict[str, SimRes] = {}

    def _add(self, sim: SimRes):
        self._store[sim.sim_id] = sim

    def __getitem__(self, sim_id: str) -> SimRes:
        return self._store[sim_id]

    def __iter__(self):
        return iter(self._store.values())

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"SimCollection({len(self._store)} simulations)"

    @property
    def ids(self) -> List[str]:
        return list(self._store.keys())

    def apply_on_spikes(self, func: Callable[[pd.DataFrame], Any]) -> None:
        for sim in self._store.values():
            sim.spike_array = func(sim.spikes)
    
    def edit_spike_arrays(self, func: Callable[[Any], Any]) -> None:
        for sim in self._store.values():
            sim.spike_array = func(sim.spike_array)

    def summary_spike_array(self) -> pd.DataFrame:
        records = []
        for sim in self._store.values():
            for _, row in sim.spike_array.iterrows():
                records.append({
                    'sim_id': sim.sim_id,
                    'arch': sim.arch,
                    'global_syn_weight': sim.global_syn_weight,
                    'pr': sim.pr,
                    'input_frequency_hz': sim.input_frequency_hz,
                    'master_seed': sim.seed,
                    'neuron': row['neuron'],
                    'neuron_idx': row['neuron_idx'],
                    'spike_times': row['spike_times'],
                    'n_spikes': row['n_spikes'],
                    'average_rate_hz': row['average_rate_hz'],
                    'first_spike': row['first_spike'],
                    'name': row.get('name', np.nan),
                    'celltype': row.get('celltype', np.nan),

                })
        return pd.DataFrame(records)


def load_results(folder: str) -> SimCollection:
    collection = SimCollection()
    fps = sorted(f for f in os.listdir(folder) if f.endswith('.json'))
    for file_idx, fname in enumerate(fps):
        with open(os.path.join(folder, fname)) as f:
            raw = json.load(f)
        params = {
            'global_syn_weight': raw['global_syn_weight'],
            'pr': raw['pr'],
            'input_frequency_hz': raw['input_frequency_hz'],
            'seed': raw['seed'],
        }
        df = pd.DataFrame(raw['data'])
        for (sid, arch), group in df.groupby(['sim_id', 'arch']):
            arch_flag = 1 if arch == 'full' else 0
            collection._add(SimRes(
                sim_id=f"{file_idx}-{sid}-{arch_flag}",
                arch=arch,
                spikes=group.drop(columns=['sim_id', 'arch']).reset_index(drop=True),
                **params,
            ))
    return collection

sims = load_results('out')
all_ids = sims.ids
print(f"Loaded data from {len(sims.ids)} simulations")

#%% meta data 
neus = pd.read_csv("data/skid_to_name_wMB.csv")
neuron_details = pd.read_csv('data/neuron_details.csv')
neus['skid'] = neus['skid'].astype(int)
skid_to_name = dict(zip(neus['skid'], neus['name']))
skid_to_celltype = dict(zip(neuron_details['skeleton_id'], neuron_details['celltype'])) 

#%%
def spikes_to_spike_array(spikes: pd.DataFrame) -> np.ndarray:
    return (
        spikes.groupby(["neuron", "neuron_idx"])["time_s"]
        .apply(list)
        .reset_index()
        .rename(columns={"time_s": "spike_times"})
        [["neuron", "neuron_idx", "spike_times"]]
    )

def spike_array_params(spike_array: pd.DataFrame) -> pd.DataFrame:
    spike_array = spike_array.copy()
    spike_array['n_spikes'] = spike_array['spike_times'].apply(len)
    sim_length = spike_array['spike_times'].apply(lambda x: max(x) if len(x) > 0 else 0).max()
    spike_array['average_rate_hz'] = spike_array['n_spikes'] / sim_length if sim_length > 0 else 0
    spike_array['first_spike'] = spike_array['spike_times'].apply(lambda x: min(x) if len(x) > 0 else np.nan)
    spike_array.sort_values('first_spike', inplace=True)
    spike_array.reset_index(drop=True, inplace=True)
    return spike_array

def spike_array_meta(spike_array: pd.DataFrame) -> pd.DataFrame:
    spike_array = spike_array.copy()
    spike_array['name'] = spike_array['neuron'].map(skid_to_name)
    spike_array['celltype'] = spike_array['neuron'].map(skid_to_celltype)
    return spike_array


sims.apply_on_spikes(spikes_to_spike_array)
sims.edit_spike_arrays(spike_array_params)
sims.edit_spike_arrays(spike_array_meta)
sim_summary = sims.summary_spike_array()


#%% sks summmary 

def to_simXneurons(spike_array: pd.DataFrame) -> pd.DataFrame:
    simX = (spike_array
            .pivot(index='sim_id', columns='neuron', values='spike_times')
            .map(lambda x: x if isinstance(x, list) else []))
    simid_to_arch = {sim.sim_id: sim.arch for sim in sims}
    simid_to_pr = {sim.sim_id: sim.pr for sim in sims}
    simid_to_weight = {sim.sim_id: sim.global_syn_weight for sim in sims}
    simX['arch'] = simX.index.map(simid_to_arch)
    simX['pr'] = simX.index.map(simid_to_pr)
    simX['global_syn_weight'] = simX.index.map(simid_to_weight)
    neus_in_data = set(simX.columns) - {'arch', 'pr', 'global_syn_weight'}
    return simX, simid_to_arch, simid_to_pr, simid_to_weight, neus_in_data


simX, simid_to_arch, simid_to_pr, simid_to_weight, neus_in_data = to_simXneurons(sim_summary)


#%% summary style analysis 

def get_firing_info_from_spk_catalog(spk_catalog, skid_to_celltype, skid_groups_of_interest, skid_to_name, all_neurons, time_wins=[(0, 0.1), (0, 0.2), (0, 0.5), (0, 1)], fano_window_ms=[0.1, 0.2, 0.5, 1], t_start=0, t_end=5):
    ''' Skid groups of interest should be dicts in the format group_name: [skids in group]'''
    spkc = spk_catalog.copy()
    
    all_celltypes = set(skid_to_celltype.values())
    values = {}
    values['simulation'] = spkc.index.tolist() # simulation names
    
    # define cell masks to detect activity
    neu_cols = [sk for sk in spkc.columns if sk in all_neurons]
    # adjust skid_groups_of_interest to only include skids that are in the spk catalog
    skid_groups_of_interest = {g: [sk for sk in skids if sk in neu_cols] for g, skids in skid_groups_of_interest.items()}

    ct_to_skids = {ct: [skid for skid, ct_ in skid_to_celltype.items() if ct_ == ct] for ct in set(skid_to_celltype.values())}
    ct_masks = {ct: [sk for sk in spkc.columns if sk in ct_to_skids[ct]] for ct in ct_to_skids}

    # overall responders and activity 
    values['responders_1'] = spkc[neu_cols].map(lambda x: len(x)> 0 ).sum(axis=1)
    values['responders_10'] = spkc[neu_cols].map(lambda x: len(x)> 10 ).sum(axis=1)
    values.update({f'responders_1_{c}': spkc[ct_masks[c]].map(lambda x: len(x)> 0 ).sum(axis=1) for c in ct_masks.keys()})
    values.update({f'responders_10_{c}': spkc[ct_masks[c]].map(lambda x: len(x)> 10 ).sum(axis=1) for c in ct_masks.keys()})
    values.update({f'responders_1_{g}': spkc[skid_groups_of_interest[g]].map(lambda x: len(x)> 0 ).sum(axis=1) for g in skid_groups_of_interest.keys()})
    values.update({f'responders_10_{g}': spkc[skid_groups_of_interest[g]].map(lambda x: len(x)> 10 ).sum(axis=1) for g in skid_groups_of_interest.keys()}) 

    values['total_activity'] = spkc[neu_cols].map(lambda x: len(x) if isinstance(x, list) else 0).sum(axis=1)
    values.update({f'total_activity_{c}': spkc[ct_masks[c]].map(lambda x: len(x) if isinstance(x, list) else 0).sum(axis=1) for c in ct_masks.keys()})
    values.update({f'total_activity_{g}': spkc[skid_groups_of_interest[g]].map(lambda x: len(x) if isinstance(x, list) else 0).sum(axis=1) for g in skid_groups_of_interest.keys()})

    # smaller time windows at start 
    for start, end in time_wins:
        values.update({f'total_activity_{start}_{end}ms': spkc[neu_cols].map(lambda x: len([t for t in x if isinstance(x, list) and start <= t <= end]) if isinstance(x, list) else 0).sum(axis=1)})
        values.update({f'total_activity_{start}_{end}ms_{c}': spkc[ct_masks[c]].map(lambda x: len([t for t in x if isinstance(x, list) and start <= t <= end]) if isinstance(x, list) else 0).sum(axis=1) for c in ct_masks.keys()})
        values.update({f'total_activity_{start}_{end}ms_{g}': spkc[skid_groups_of_interest[g]].map(lambda x: len([t for t in x if isinstance(x, list) and start <= t <= end]) if isinstance(x, list) else 0).sum(axis=1) for g in skid_groups_of_interest.keys()})

    # compute avg latency of each cell type and group of interest - only for cells that respond at least once
    first_spks = spkc[neu_cols].map(lambda x: min(x) if isinstance(x, list) and len(x) > 0 else np.nan)
    values.update({f'avg_latency': first_spks.mean(axis=1)})
    values.update({f'avg_latency_{c}': first_spks[ct_masks[c]].map(lambda x: np.nanmean(x) if np.sum(~np.isnan(x)) > 0 else np.nan).mean(axis=1) for c in ct_masks.keys()})
    values.update({f'avg_latency_{g}': first_spks[skid_groups_of_interest[g]].map(lambda x: np.nanmean(x) if np.sum(~np.isnan(x)) > 0 else np.nan).mean(axis=1) for g in skid_groups_of_interest.keys()})

    # very first spikes 
    values.update({f'first_spike': first_spks.min(axis=1)})
    values.update({f'first_spike_{c}': first_spks[ct_masks[c]].min(axis=1) for c in ct_masks.keys()})
    values.update({f'first_spike_{g}': first_spks[skid_groups_of_interest[g]].min(axis=1) for g in skid_groups_of_interest.keys()})

    # compute average fano factor of each cell type and group of interest - only for cells that respond at least once
    # trying different time scales here to better understand whats going on 
    for window_size_ms in fano_window_ms:
        values.update({f'avg_fano_{window_size_ms}ms': spkc[neu_cols].map(lambda x: get_fano_factor(x, t_start, t_end, window_size_ms) if isinstance(x, list) and len(x) > 0 else np.nan).mean(axis=1)})
        values.update({f'avg_fano_{window_size_ms}ms_{c}': spkc[ct_masks[c]].map(lambda x: get_fano_factor(x, t_start, t_end, window_size_ms) if isinstance(x, list) and len(x) > 0 else np.nan).mean(axis=1) for c in ct_masks.keys()})
        values.update({f'avg_fano_{window_size_ms}ms_{g}': spkc[skid_groups_of_interest[g]].map(lambda x: get_fano_factor(x, t_start, t_end, window_size_ms) if isinstance(x, list) and len(x) > 0 else np.nan).mean(axis=1) for g in skid_groups_of_interest.keys()})
    # and isi cv 
    for st, en in [(t_start, t_end)]+time_wins:
        isi_all, _ = compute_isi_stats(spkc[neu_cols], st, en)
        vals = isi_all.groupby('cond')['isi_cv'].mean()
        vals_full = pd.Series(index=spkc.index, dtype=float)
        vals_full[vals.index] = vals.values 
        values[f'avg_isi_cv_{st}_{en}ms'] = list(vals_full.values)

        for g in ct_masks.keys():
            isi_group, _ = compute_isi_stats(spkc[ct_masks[g]], st, en)
            if isi_group.empty:
                values[f'avg_isi_cv_{st}_{en}ms{g}'] = [np.nan] * len(spkc)
            else:   
                vals = isi_group.groupby('cond')['isi_cv'].mean()
                vals_full = pd.Series(index=spkc.index, dtype=float) # create full series with all conditions
                vals_full[vals.index] = vals.values # fill in values for conditions that have data
                values[f'avg_isi_cv_{st}_{en}ms{g}'] = list(vals_full.values) # ensure it's a 1D array
        for g in skid_groups_of_interest.keys():
            isi_group, _ = compute_isi_stats(spkc[skid_groups_of_interest[g]], st, en)
            if isi_group.empty:
                values[f'avg_isi_cv_{st}_{en}ms{g}'] = [np.nan] * len(spkc)
            else:
                vals = isi_group.groupby('cond')['isi_cv'].mean()
                vals_full = pd.Series(index=spkc.index,  dtype=float) # create full series with all conditions
                vals_full[vals.index] = vals.values # fill in values for conditions that have data
                values[f'avg_isi_cv_{st}_{en}ms{g}'] = list(vals_full.values) # ensure it's a 1D array
    
    for k, v in values.items():
        if isinstance(v, pd.Series):
            values[k] = list(v.values)
        if isinstance(v, np.ndarray):
            print(f'{k} is an array')
        if isinstance(v, list) != True:
            print(f'{k} is not a list, converting to list')
            print(v.shape)
        print(f'{k}: {len(v)}') # check values as we go
    df = pd.DataFrame.from_dict(values)
    #df = df.T
    return df

skid_groups_of_interest = {'input': [3827211, 3464773,3410499,2608843], 'mbon-m1': [17016974,4022539]}
firing_info_df = get_firing_info_from_spk_catalog(simX, skid_to_celltype, skid_groups_of_interest, skid_to_name, neus_in_data)
firing_info_df['arch'] = firing_info_df['simulation'].map(lambda x: simid_to_arch[x])



# %%
