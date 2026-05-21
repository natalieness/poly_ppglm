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
print(f"Loaded {len(sims)} simulations")

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


# %%
