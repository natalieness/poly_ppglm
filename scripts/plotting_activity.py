import matplotlib.pyplot as plt

def plot_spikes(result, title='Simulated Spikes', plot_max=10) -> None:

    fig, ax = plt.subplots(figsize=(10, 6))
    for neuron_idx, neuron in enumerate(result.neuron_names):
        if neuron_idx >= plot_max:
            break
        neuron_spikes = result.spikes[result.spikes["neuron_idx"] == neuron_idx]
        ax.scatter(
            neuron_spikes["time_s"],
            [neuron_idx] * len(neuron_spikes),
            label=neuron,
            s=10,
        )

    ax.set_xlabel("Time (s)")
    ax.set_yticks(range(min(plot_max, len(result.neuron_names))))
    ax.set_yticklabels(result.neuron_names[:plot_max])
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

    return fig, ax