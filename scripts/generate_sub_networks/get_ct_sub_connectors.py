from itertools import chain
import ast

import pandas as pd
import numpy as np

neurons = pd.read_csv('data/neuron_details.csv')
kc_axo_axonic = pd.read_csv('data/kc_axo_axonic.csv')
neurons_nt = pd.read_csv('data/20260205_corrected_nts.csv')
connectors = pd.read_csv('data/polyadic_connectors.csv', dtype={'presynaptic_id': 'Int64', 'postsynaptic_id': 'string'})


restrict_network = ['KCs', 'MBINs', 'MBONs', 'MB-FBNs'] # set to None to include all neurons with connectors

all_neurons = neurons['skeleton_id'].to_list()
u_celltypes = neurons['celltype'].unique()
celltype_pops = {ct: neurons[neurons['celltype']==ct]['skeleton_id'].to_list() for ct in u_celltypes}
#connectors
if restrict_network is not None:
    pop_skids = []
    for pop in restrict_network:
        pop_skids += celltype_pops[pop]
        if isinstance(pop_skids[0], list):
            pop_skids = list(chain.from_iterable(pop_skids))
        all_neurons = pop_skids

# only get connectors from neurons in the network 
connectors['presynaptic_id'] = connectors['presynaptic_id'].astype(int)
connectors = connectors[connectors['presynaptic_id'].isin(all_neurons)]
connectors['postsynaptic_id'] = connectors['postsynaptic_id'].apply(ast.literal_eval)
connectors['postsynaptic_id'] = connectors['postsynaptic_id'].apply(lambda x: [skid for skid in x if skid in all_neurons])
connectors = connectors[connectors['postsynaptic_id'].apply(len) > 0] # drop connectors with no postsynaptic partners in the network



# process kc axo-axonic connections 
kc_axo_axonic['postsynaptic_to'] = kc_axo_axonic['postsynaptic_to'].apply(ast.literal_eval)
kc_axo_axonic['connector_id'] = kc_axo_axonic['connector_id'].astype(int)
kc_aa = dict(zip(kc_axo_axonic['connector_id'], kc_axo_axonic['postsynaptic_to']))
kc_aa_skids = set(kc_axo_axonic['presynaptic_to'].tolist()) # to quickly look up 


# inhibitory neurons include GABA and Glutamate
inhibitory_nts = ['GABA', 'Glutamate']

# first filter by confidence threshold
threshold = 0.0
neurons_nt_filtered = neurons_nt[neurons_nt['neurotransmitter_confidence'] >= threshold]
inhibitory_neurons = neurons_nt_filtered[neurons_nt_filtered['predicted_neurotransmitter'].isin(inhibitory_nts)]['skeleton_id'].to_list()
print(f'Found {len(inhibitory_neurons)} inhibitory neurons with confidence >= {threshold}')


# save for sim 
connectors.to_csv('data/ct_sub_connectors_mb.csv', index=False)
inhibitory_neurons_df = pd.DataFrame({'skid': inhibitory_neurons})
inhibitory_neurons_df.to_csv('data/2026_02_inhibitory_neurons.csv', index=False)


