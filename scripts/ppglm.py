from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import time
import ast 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import seaborn as sns

from scripts.plot_networks import build_connectivity_graph, plot_connectivity_graph
from scripts.plotting_activity import plot_spikes

@dataclass
class NeuronParams:
    """
    Parameters controlling each neuron's point-process GLM spike process.
    """

    # Baseline firing rate in Hz.
    base_rate_hz: float = 2.0

    # Self-history term added to log-intensity after a spike.
    # Negative values create refractoriness.
    self_history_weight: float = -3.0
    self_history_tau_s: float = 0.020

    # Used by p_mode="neuron_constant".
    p_release: float = 0.6

    # Used by p_mode="neuron_beta_static".
    beta_alpha: float = 6.0
    beta_beta: float = 4.0

    # Used by p_mode="neuron_gamma_cox_per_spike".
    # rho ~ Gamma(shape, scale)
    # p = 1 - exp(-rho * gamma_cox_scale)
    gamma_shape: float = 2.0
    gamma_scale: float = 0.5
    gamma_cox_scale: float = 1.0


@dataclass
class ReleaseSite:
    """
    One presynaptic synapse/release site.

    User-facing connectivity is:

        {
            presynaptic_neuron: {
                synapse_id: list_of_postsynaptic_partners
            }
        }

    Example:

        {
            "A": {
                "s1": ["B", "C"],
                "s2": ["B"]
            }
        }

    If partners = ["B", "C"], then one release event from this site
    can be broadcast to both B and C.
    """

    pre: str
    synapse_id: str
    partners: List[str]

    # Same weight and delay apply to every postsynaptic partner at this site.
    weight: float = 1.0
    delay_s: float = 0.001

    # Probability mode:
    #
    #   "site_constant"
    #       Use this ReleaseSite.p_release.
    #       If p_release is None, fall back to NeuronParams.p_release.
    #
    #   "neuron_constant"
    #       Use NeuronParams.p_release.
    #
    #   "site_beta_static"
    #       Sample one fixed p for this site:
    #           p_site ~ Beta(beta_alpha, beta_beta)
    #
    #   "neuron_beta_static"
    #       Sample one fixed p for each presynaptic neuron:
    #           p_A ~ Beta(alpha_A, beta_A)
    #
    #   "site_gamma_cox_per_spike"
    #       On every parent spike, draw a site-specific release propensity:
    #           rho ~ Gamma(shape, scale)
    #           p = 1 - exp(-rho * gamma_cox_scale)
    #
    #   "neuron_gamma_cox_per_spike"
    #       On every parent spike, draw one neuron-wide release propensity
    #       shared across that neuron's release sites.
    p_mode: Optional[str] = None
    p_release: Optional[float] = None

    beta_alpha: float = 6.0
    beta_beta: float = 4.0

    gamma_shape: float = 2.0
    gamma_scale: float = 0.5
    gamma_cox_scale: float = 1.0

    # Amplitude model:
    #
    #   "fixed"
    #   "gamma"
    #   "lognormal"
    q_mode: str = "fixed"
    q_mean: float = 1.0
    q_gamma_shape: float = 10.0
    q_lognormal_sigma: float = 0.20

    # Linking behavior across postsynaptic partners at the same synapse.
    share_release_across_partners: bool = True
    share_amplitude_across_partners: bool = True
    share_jitter_across_partners: bool = True

    # Latency jitter added to delay_s.
    jitter_sd_s: float = 0.0

    # If True, divide Q by number of partners.
    conserve_amplitude_across_partners: bool = False

    # Optional simple short-term depression.
    # Effective p is multiplied by stp_x in [0, 1].
    stp_enabled: bool = False
    stp_tau_recovery_s: float = 0.500
    stp_depletion_per_success: float = 0.25

    @property
    def uid(self) -> str:
        return f"{self.pre}:{self.synapse_id}"


@dataclass
class SimulationResult:
    """
    Container for simulation outputs.
    """

    spikes: pd.DataFrame
    release_events: pd.DataFrame
    lambda_hz: np.ndarray
    syn_drive: np.ndarray
    time_s: np.ndarray
    neuron_names: List[str]


# ---------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------


class LinkedReleasePPGLM:
    """
    Recurrent point-process GLM with linked stochastic presynaptic release sites.

    Neuron intensity:

        lambda_i(t) = exp(
            log(base_rate_i)
            + external_i(t)
            + syn_drive_i(t)
            + self_history_i(t)
        )

    A presynaptic spike from neuron A triggers release attempts at each
    release site of A.

    If release site A:s1 has partners ["B", "C"], then one release draw can
    be broadcast to both postsynaptic neurons:

        Z_{A,s1->B}^k = Z_{A,s1->C}^k = Z_{A,s1}^k

    The synapse has one weight and one delay shared across all partners.
    """

    ALLOWED_SYNAPSE_FIELDS = {
        "weight",
        "delay_s",
        "p_mode",
        "p_release",
        "beta_alpha",
        "beta_beta",
        "gamma_shape",
        "gamma_scale",
        "gamma_cox_scale",
        "q_mode",
        "q_mean",
        "q_gamma_shape",
        "q_lognormal_sigma",
        "share_release_across_partners",
        "share_amplitude_across_partners",
        "share_jitter_across_partners",
        "jitter_sd_s",
        "conserve_amplitude_across_partners",
        "stp_enabled",
        "stp_tau_recovery_s",
        "stp_depletion_per_success",
    }

    def __init__(
        self,
        neurons: Sequence[str],
        release_sites: Sequence[ReleaseSite],
        neuron_params: Optional[Mapping[str, NeuronParams | Mapping[str, Any]]] = None,
        *,
        dt_s: float = 0.001,
        t_stop_s: float = 1.0,
        default_p_mode: str = "site_constant",
        syn_tau_s: float = 0.010,
        max_rate_hz: float = 500.0,
        seed: Optional[int] = None,
        record_failed_release_attempts: bool = True,
    ):
        self.neurons = list(neurons)
        self.name_to_idx = {name: i for i, name in enumerate(self.neurons)}
        self.n_neurons = len(self.neurons)

        self.release_sites = list(release_sites)
        self.n_sites = len(self.release_sites)

        self.neuron_params = self._normalize_neuron_params(neuron_params)

        self.dt_s = float(dt_s)
        self.t_stop_s = float(t_stop_s)
        self.default_p_mode = default_p_mode
        self.syn_tau_s = float(syn_tau_s)
        self.max_rate_hz = float(max_rate_hz)
        self.record_failed_release_attempts = record_failed_release_attempts

        self.rng = np.random.default_rng(seed)

        self._validate()
        self.sites_by_pre_idx = self._build_sites_by_pre_idx()

        # Static beta probabilities are sampled once at initialization.
        self._neuron_beta_static_p = {
            name: self.rng.beta(
                self.neuron_params[name].beta_alpha,
                self.neuron_params[name].beta_beta,
            )
            for name in self.neurons
        }

        self._site_beta_static_p = {
            site.uid: self.rng.beta(site.beta_alpha, site.beta_beta)
            for site in self.release_sites
        }

    # -----------------------------------------------------------------
    # Builders
    # -----------------------------------------------------------------

    @classmethod
    def from_simple_connectivity(
        cls,
        neurons: Sequence[str],
        connectivity: Mapping[str, Mapping[str, Sequence[str]]],
        *,
        neuron_params: Optional[Mapping[str, NeuronParams | Mapping[str, Any]]] = None,
        global_synapse_defaults: Optional[Mapping[str, Any]] = None,
        synapse_defaults_by_pre: Optional[Mapping[str, Mapping[str, Any]]] = None,
        synapse_params: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
        **kwargs,
    ) -> "LinkedReleasePPGLM":
        """
        Build model from simplified connectivity.

        connectivity format:

            {
                "A": {
                    "s1": ["B", "C"],
                    "s2": ["B"],
                    "s3": ["C"],
                },
                "B": {
                    "s1": ["C"],
                }
            }

        global_synapse_defaults:
            Parameters applied to every synapse.

        synapse_defaults_by_pre:
            Parameters applied to all synapses from a given presynaptic neuron.

            {
                "A": {"p_release": 0.7, "delay_s": 0.003}
            }

        synapse_params:
            Optional per-synapse overrides.

            {
                "A": {
                    "s1": {"weight": 1.4, "delay_s": 0.002}
                }
            }

        Merge order:

            global_synapse_defaults
                -> synapse_defaults_by_pre[pre]
                    -> synapse_params[pre][synapse_id]
        """
        global_synapse_defaults = dict(global_synapse_defaults or {})
        synapse_defaults_by_pre = dict(synapse_defaults_by_pre or {})
        synapse_params = dict(synapse_params or {})

        cls._raise_on_unknown_fields(global_synapse_defaults, "global_synapse_defaults")

        for pre, defaults in synapse_defaults_by_pre.items():
            cls._raise_on_unknown_fields(defaults, f"synapse_defaults_by_pre[{pre!r}]")

        for pre, site_dict in synapse_params.items():
            for synapse_id, params in site_dict.items():
                cls._raise_on_unknown_fields(
                    params,
                    f"synapse_params[{pre!r}][{synapse_id!r}]",
                )

        release_sites: List[ReleaseSite] = []

        for pre, synapse_dict in connectivity.items():
            for synapse_id, partners in synapse_dict.items():
                params: Dict[str, Any] = {}
                params.update(global_synapse_defaults)
                params.update(synapse_defaults_by_pre.get(pre, {}))
                params.update(synapse_params.get(pre, {}).get(synapse_id, {}))

                release_sites.append(
                    ReleaseSite(
                        pre=pre,
                        synapse_id=synapse_id,
                        partners=list(partners),
                        **params,
                    )
                )

        return cls(
            neurons=neurons,
            release_sites=release_sites,
            neuron_params=neuron_params,
            **kwargs,
        )

    @staticmethod
    def _raise_on_unknown_fields(params: Mapping[str, Any], label: str) -> None:
        unknown = set(params) - LinkedReleasePPGLM.ALLOWED_SYNAPSE_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown synapse parameter(s) in {label}: {sorted(unknown)}"
            )

    def _normalize_neuron_params(
        self,
        neuron_params: Optional[Mapping[str, NeuronParams | Mapping[str, Any]]],
    ) -> Dict[str, NeuronParams]:
        out = {name: NeuronParams() for name in self.neurons}

        if neuron_params is None:
            return out

        for name, params in neuron_params.items():
            if name not in out:
                raise ValueError(f"NeuronParams provided for unknown neuron {name!r}.")

            if isinstance(params, NeuronParams):
                out[name] = params
            elif isinstance(params, Mapping):
                out[name] = NeuronParams(**params)
            else:
                raise TypeError(
                    f"neuron_params[{name!r}] must be NeuronParams or a dict."
                )

        return out

    # -----------------------------------------------------------------
    # Public simulation method
    # -----------------------------------------------------------------

    def simulate(
        self,
        external_log_drive: Optional[
            np.ndarray | Callable[[float, List[str]], np.ndarray]
        ] = None,
        clamped_spikes: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> SimulationResult:
        """
        Run simulation.

        external_log_drive:
            None, or an array of shape [n_neurons, n_time_bins], or a function:

                f(t_s, neuron_names) -> np.ndarray of shape [n_neurons]

            This is additive in log-intensity.

        clamped_spikes:
            Optional dict mapping neuron name to exact spike times in seconds.

            Example:

                {
                    "A": [0.050, 0.100, 0.150]
                }

            A clamped neuron only spikes at the provided times.
        """
        n_bins = int(np.ceil(self.t_stop_s / self.dt_s))
        time_s = np.arange(n_bins) * self.dt_s

        max_delay_s = self._max_delay_s()
        padding_bins = int(np.ceil(max_delay_s / self.dt_s)) + 10

        pending_input = np.zeros(
            (self.n_neurons, n_bins + padding_bins),
            dtype=float,
        )

        lambda_hz = np.zeros((self.n_neurons, n_bins), dtype=float)
        syn_drive_trace = np.zeros((self.n_neurons, n_bins), dtype=float)

        syn_drive = np.zeros(self.n_neurons, dtype=float)
        self_history = np.zeros(self.n_neurons, dtype=float)

        stp_x = np.ones(self.n_sites, dtype=float)

        base_log_rates = np.array(
            [
                np.log(max(self.neuron_params[name].base_rate_hz, 1e-12))
                for name in self.neurons
            ],
            dtype=float,
        )

        self_weights = np.array(
            [
                self.neuron_params[name].self_history_weight
                for name in self.neurons
            ],
            dtype=float,
        )

        self_taus = np.array(
            [
                max(self.neuron_params[name].self_history_tau_s, 1e-12)
                for name in self.neurons
            ],
            dtype=float,
        )

        forced = self._make_forced_spike_matrix(clamped_spikes, n_bins)

        spike_records: List[Dict[str, Any]] = []
        release_records: List[Dict[str, Any]] = []

        syn_decay = np.exp(-self.dt_s / max(self.syn_tau_s, 1e-12))
        self_decay = np.exp(-self.dt_s / self_taus)

        for b, t in enumerate(time_s):
            # Update synaptic drive.
            syn_drive *= syn_decay
            syn_drive += pending_input[:, b]

            # Update self-history.
            self_history *= self_decay

            # Recover STP variables.
            self._recover_stp(stp_x)

            # External input.
            ext = self._external_at_bin(external_log_drive, b, t, n_bins)

            # Point-process GLM intensity in Hz.
            eta = base_log_rates + ext + syn_drive + self_history
            lam = np.exp(eta)
            lam = np.clip(lam, 0.0, self.max_rate_hz)

            lambda_hz[:, b] = lam
            syn_drive_trace[:, b] = syn_drive

            # Bernoulli approximation to point process in small dt.
            spike_prob = 1.0 - np.exp(-lam * self.dt_s)
            spiked = self.rng.random(self.n_neurons) < spike_prob

            # Override generated spikes for clamped neurons.
            if forced is not None:
                clamped_mask = forced["clamped_neuron_mask"]
                forced_spikes = forced["forced_spikes"]
                spiked[clamped_mask] = forced_spikes[clamped_mask, b]

            spiking_indices = np.flatnonzero(spiked)

            for pre_idx in spiking_indices:
                pre = self.neurons[pre_idx]

                spike_records.append(
                    {
                        "time_s": t,
                        "neuron": pre,
                        "neuron_idx": pre_idx,
                        "lambda_hz": lam[pre_idx],
                    }
                )

                self_history[pre_idx] += self_weights[pre_idx]

                # Stores neuron-wide stochastic draws for this spike.
                # Example: neuron_gamma_cox_per_spike should be shared across
                # all release sites from this presynaptic spike.
                spike_context: Dict[str, float] = {}

                for site_idx in self.sites_by_pre_idx.get(pre_idx, []):
                    site = self.release_sites[site_idx]

                    p_eff = self._draw_release_probability(
                        pre=pre,
                        site=site,
                        stp_x_value=stp_x[site_idx],
                        spike_context=spike_context,
                    )

                    self._process_release_site(
                        pre=pre,
                        pre_idx=pre_idx,
                        site_idx=site_idx,
                        site=site,
                        parent_spike_time_s=t,
                        current_bin=b,
                        p_eff=p_eff,
                        stp_x=stp_x,
                        pending_input=pending_input,
                        release_records=release_records,
                    )

        spikes_df = pd.DataFrame(spike_records)
        release_df = pd.DataFrame(release_records)

        return SimulationResult(
            spikes=spikes_df,
            release_events=release_df,
            lambda_hz=lambda_hz,
            syn_drive=syn_drive_trace,
            time_s=time_s,
            neuron_names=self.neurons,
        )

    # -----------------------------------------------------------------
    # Release-site processing
    # -----------------------------------------------------------------

    def _process_release_site(
        self,
        *,
        pre: str,
        pre_idx: int,
        site_idx: int,
        site: ReleaseSite,
        parent_spike_time_s: float,
        current_bin: int,
        p_eff: float,
        stp_x: np.ndarray,
        pending_input: np.ndarray,
        release_records: List[Dict[str, Any]],
    ) -> None:
        n_partners = len(site.partners)

        # Release success/failure.
        if site.share_release_across_partners:
            released_shared = self.rng.random() < p_eff
            release_by_partner = [released_shared] * n_partners
        else:
            release_by_partner = [
                self.rng.random() < p_eff
                for _ in range(n_partners)
            ]

        # Amplitude Q.
        if site.share_amplitude_across_partners:
            q_shared = self._draw_amplitude(site)
            q_by_partner = [q_shared] * n_partners
        else:
            q_by_partner = [
                self._draw_amplitude(site)
                for _ in range(n_partners)
            ]

        if site.conserve_amplitude_across_partners and n_partners > 0:
            q_by_partner = [q / n_partners for q in q_by_partner]

        # Latency jitter.
        if site.share_jitter_across_partners:
            jitter_shared = self.rng.normal(0.0, site.jitter_sd_s)
            jitter_by_partner = [jitter_shared] * n_partners
        else:
            jitter_by_partner = [
                self.rng.normal(0.0, site.jitter_sd_s)
                for _ in range(n_partners)
            ]

        any_success = any(release_by_partner)

        # Simple short-term depression.
        if site.stp_enabled and any_success:
            stp_x[site_idx] -= site.stp_depletion_per_success
            stp_x[site_idx] = np.clip(stp_x[site_idx], 0.0, 1.0)

        for post, released, q, jitter in zip(
            site.partners,
            release_by_partner,
            q_by_partner,
            jitter_by_partner,
        ):
            post_idx = self.name_to_idx[post]

            arrival_time_s = (
                parent_spike_time_s
                + site.delay_s
                + jitter
            )

            arrival_bin = int(np.round(arrival_time_s / self.dt_s))

            # Prevent zero-delay causal loop within the same bin.
            arrival_bin = max(arrival_bin, current_bin + 1)

            amplitude = float(released) * q * site.weight

            if released and arrival_bin < pending_input.shape[1]:
                pending_input[post_idx, arrival_bin] += amplitude

            if released or self.record_failed_release_attempts:
                release_records.append(
                    {
                        "parent_spike_time_s": parent_spike_time_s,
                        "arrival_time_s": arrival_bin * self.dt_s,
                        "pre": pre,
                        "pre_idx": pre_idx,
                        "synapse_id": site.synapse_id,
                        "site_uid": site.uid,
                        "post": post,
                        "post_idx": post_idx,
                        "released": bool(released),
                        "p_release_eff": p_eff,
                        "q": q,
                        "weight": site.weight,
                        "amplitude": amplitude,
                        "delay_s": site.delay_s,
                        "jitter_s": jitter,
                        "n_partners_at_site": n_partners,
                        "share_release_across_partners": site.share_release_across_partners,
                        "share_amplitude_across_partners": site.share_amplitude_across_partners,
                        "share_jitter_across_partners": site.share_jitter_across_partners,
                        "stp_x_after": stp_x[site_idx],
                    }
                )

    def _draw_release_probability(
        self,
        *,
        pre: str,
        site: ReleaseSite,
        stp_x_value: float,
        spike_context: Dict[str, float],
    ) -> float:
        mode = site.p_mode or self.default_p_mode
        npar = self.neuron_params[pre]

        if mode == "site_constant":
            p = npar.p_release if site.p_release is None else site.p_release

        elif mode == "neuron_constant":
            p = npar.p_release

        elif mode == "site_beta_static":
            p = self._site_beta_static_p[site.uid]

        elif mode == "neuron_beta_static":
            p = self._neuron_beta_static_p[pre]

        elif mode == "site_gamma_cox_per_spike":
            rho = self.rng.gamma(site.gamma_shape, site.gamma_scale)
            p = 1.0 - np.exp(-rho * site.gamma_cox_scale)

        elif mode == "neuron_gamma_cox_per_spike":
            key = f"neuron_gamma_cox:{pre}"
            if key not in spike_context:
                rho = self.rng.gamma(npar.gamma_shape, npar.gamma_scale)
                spike_context[key] = 1.0 - np.exp(-rho * npar.gamma_cox_scale)
            p = spike_context[key]

        else:
            raise ValueError(f"Unknown p_mode: {mode!r}")

        p = float(np.clip(p, 0.0, 1.0))

        if site.stp_enabled:
            p *= float(np.clip(stp_x_value, 0.0, 1.0))

        return float(np.clip(p, 0.0, 1.0))

    def _draw_amplitude(self, site: ReleaseSite) -> float:
        if site.q_mode == "fixed":
            return float(site.q_mean)

        if site.q_mode == "gamma":
            shape = max(site.q_gamma_shape, 1e-12)
            scale = site.q_mean / shape
            return float(self.rng.gamma(shape, scale))

        if site.q_mode == "lognormal":
            sigma = site.q_lognormal_sigma
            mu = np.log(max(site.q_mean, 1e-12)) - 0.5 * sigma**2
            return float(self.rng.lognormal(mean=mu, sigma=sigma))

        raise ValueError(f"Unknown q_mode: {site.q_mode!r}")

    def _recover_stp(self, stp_x: np.ndarray) -> None:
        for site_idx, site in enumerate(self.release_sites):
            if not site.stp_enabled:
                continue

            tau = max(site.stp_tau_recovery_s, 1e-12)
            recovery_fraction = 1.0 - np.exp(-self.dt_s / tau)
            stp_x[site_idx] += (1.0 - stp_x[site_idx]) * recovery_fraction
            stp_x[site_idx] = np.clip(stp_x[site_idx], 0.0, 1.0)

    # -----------------------------------------------------------------
    # Input helpers
    # -----------------------------------------------------------------

    def _external_at_bin(
        self,
        external_log_drive: Optional[
            np.ndarray | Callable[[float, List[str]], np.ndarray]
        ],
        b: int,
        t: float,
        n_bins: int,
    ) -> np.ndarray:
        if external_log_drive is None:
            return np.zeros(self.n_neurons, dtype=float)

        if callable(external_log_drive):
            value = np.asarray(external_log_drive(t, self.neurons), dtype=float)
            if value.shape != (self.n_neurons,):
                raise ValueError(
                    "Callable external_log_drive must return shape [n_neurons]."
                )
            return value

        arr = np.asarray(external_log_drive, dtype=float)
        if arr.shape != (self.n_neurons, n_bins):
            raise ValueError(
                f"external_log_drive must have shape "
                f"[{self.n_neurons}, {n_bins}], got {arr.shape}."
            )
        return arr[:, b]

    def _make_forced_spike_matrix(
        self,
        clamped_spikes: Optional[Mapping[str, Sequence[float]]],
        n_bins: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        if clamped_spikes is None:
            return None

        forced_spikes = np.zeros((self.n_neurons, n_bins), dtype=bool)
        clamped_neuron_mask = np.zeros(self.n_neurons, dtype=bool)

        for neuron, times in clamped_spikes.items():
            if neuron not in self.name_to_idx:
                raise ValueError(f"clamped_spikes contains unknown neuron {neuron!r}.")

            idx = self.name_to_idx[neuron]
            clamped_neuron_mask[idx] = True

            for t in times:
                b = int(np.round(float(t) / self.dt_s))
                if 0 <= b < n_bins:
                    forced_spikes[idx, b] = True

        return {
            "forced_spikes": forced_spikes,
            "clamped_neuron_mask": clamped_neuron_mask,
        }

    # -----------------------------------------------------------------
    # Internal structure helpers
    # -----------------------------------------------------------------

    def _build_sites_by_pre_idx(self) -> Dict[int, List[int]]:
        sites_by_pre_idx: Dict[int, List[int]] = {}

        for site_idx, site in enumerate(self.release_sites):
            pre_idx = self.name_to_idx[site.pre]
            sites_by_pre_idx.setdefault(pre_idx, []).append(site_idx)

        return sites_by_pre_idx

    def _max_delay_s(self) -> float:
        max_delay = 0.0

        for site in self.release_sites:
            max_delay = max(
                max_delay,
                site.delay_s + 5.0 * max(site.jitter_sd_s, 0.0),
            )

        return max_delay

    def _validate(self) -> None:
        if len(set(self.neurons)) != len(self.neurons):
            raise ValueError("Neuron names must be unique.")

        seen_sites = set()

        for site in self.release_sites:
            if site.pre not in self.name_to_idx:
                raise ValueError(f"Unknown presynaptic neuron: {site.pre!r}")

            if not site.synapse_id:
                raise ValueError(f"Synapse id cannot be empty for pre={site.pre!r}.")

            key = (site.pre, site.synapse_id)
            if key in seen_sites:
                raise ValueError(
                    f"Duplicate synapse id {site.synapse_id!r} for pre={site.pre!r}."
                )
            seen_sites.add(key)

            if len(site.partners) == 0:
                raise ValueError(f"Release site {site.uid!r} has no partners.")

            for post in site.partners:
                if post not in self.name_to_idx:
                    raise ValueError(
                        f"Release site {site.uid!r} has unknown post neuron {post!r}."
                    )

            if site.delay_s < 0:
                raise ValueError(f"Release site {site.uid!r} has negative delay_s.")

            if site.jitter_sd_s < 0:
                raise ValueError(f"Release site {site.uid!r} has negative jitter_sd_s.")

            if site.q_mean < 0:
                raise ValueError(f"Release site {site.uid!r} has negative q_mean.")

            if site.p_release is not None and not (0.0 <= site.p_release <= 1.0):
                raise ValueError(
                    f"Release site {site.uid!r} has p_release outside [0, 1]."
                )

#%% creating fake networks 

def load_connectome_from_csv(con_path: str, inhibitory_neurons_path: str, axo_axonic_path: str,
                             postsyn_col: str = 'postsynaptic_to', presyn_col: str = 'presynaptic_to', 
                             connector_col: str = 'connector_id'):
    con = pd.read_csv(con_path)
    con[postsyn_col] = con[postsyn_col].apply(ast.literal_eval)
    con[presyn_col] = con[presyn_col].astype(int)
    con[connector_col] = con[connector_col].astype(str)

    # convert to connectivity dict format: {pre: {synapse_id: [postsynaptic partners]}}
    connectivity: Dict[str, Dict[str, List[str]]] = {}
    con.groupby(presyn_col).apply(lambda g: connectivity.setdefault(str(g.name), {}).update(
        {connector_id: postsyn_list for connector_id, postsyn_list in zip(g[connector_col], g[postsyn_col])}
    ))

    # inhibitory neurons 
    inh = pd.read_csv(inhibitory_neurons_path)
    inhibitory_neurons = set(inh['skid'].tolist())

    # get neurons for sim 
    neurons = list(con[presyn_col].append(con[postsyn_col].explode()).unique())

    inhibitory_neurons = set([n for n in inhibitory_neurons if n in neurons])

    # synapse defaults aby pre 
    raise NotImplementedError("synapse defaults by pre not implemented yet, need to load from csv if desired")
    # kc aco axonic synapses with special synapse ones





def create_connectome(n_neurons: int = 10, n_inhibitory: int = 2, \
                        mean_outgoing: int = 3, sd_outgoing: int = 1, \
                        mean_polyadic_fraction: float = 0.5, sd_polyadic_fraction: float = 0.3,
                        mean_polyadic_size: float = 3.0,   # mean # partners when site is polyadic
                        sd_polyadic_size: float = 1.0, syn_weight: float = 1.0, \
                        seed: Optional[int] = None):
    # make sure at least 1 excitatory neuron 
    n_inhibitory = min(n_inhibitory, n_neurons - 1)
    n_excitatory = n_neurons - n_inhibitory
    rng = np.random.default_rng(seed)

    neurons = [f"N{i}" for i in range(n_neurons)]
    inhibitory_neurons = set(rng.choice(neurons, size=n_inhibitory, replace=False))

    # make inhibitory neurons have negative weights
    synapse_defaults_by_pre = {neuron: {"weight": -syn_weight if neuron in inhibitory_neurons else syn_weight} for neuron in neurons}
    
    # per-neuron polyadic fraction (clipped to [0, 1])
    poly_frac_by_neuron = {
        n: float(np.clip(
            rng.normal(mean_polyadic_fraction, sd_polyadic_fraction),
            0.0, 1.0,
        ))
        for n in neurons
    }

    connectivity: Dict[str, Dict[str, List[str]]] = {}

    for pre in neurons:
        # number of outgoing release sites for this neuron
        n_sites = int(round(rng.normal(mean_outgoing, sd_outgoing)))
        n_sites = max(0, min(n_sites, n_neurons - 1))
        if n_sites == 0:
            continue

        possible_posts = [n for n in neurons if n != pre]
        site_dict: Dict[str, List[str]] = {}

        for s in range(n_sites):
            # decide polyadic or monadic
            is_polyadic = rng.random() < poly_frac_by_neuron[pre]

            if is_polyadic:
                k = int(round(rng.normal(mean_polyadic_size, sd_polyadic_size)))
                k = max(2, min(k, len(possible_posts)))
            else:
                k = 1

            partners = rng.choice(possible_posts, size=k, replace=False).tolist()
            site_dict[f"s{s+1}"] = partners

        connectivity[pre] = site_dict

    return neurons, connectivity, synapse_defaults_by_pre, inhibitory_neurons

def flatten_connectome(
    connectivity: Mapping[str, Mapping[str, Sequence[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Expand polyadic synapses into monadic ones, keeping the same
    nested-dict format as `connectivity`.

    Each (pre, synapse_id) with N>1 partners becomes N synapses
    (synapse_id__0, synapse_id__1, ...), each with a single partner.
    """
    flat: Dict[str, Dict[str, List[str]]] = {}
    for pre, synapse_dict in connectivity.items():
        flat[pre] = {}
        for synapse_id, partners in synapse_dict.items():
            partners = list(partners)
            if len(partners) <= 1:
                flat[pre][synapse_id] = list(partners)
            else:
                for i, post in enumerate(partners):
                    flat[pre][f"{synapse_id}_{i}"] = [post]
    return flat


def create_spikes(start: float, end: float, Hz: float) -> List[float]:
    interval = 1.0 / Hz
    return list(np.arange(start, end, interval))

def create_spikes_with_endpoint_jitter(
    start: float,
    end: float,
    Hz: float,
    jitter: float,
) -> List[float]:
    """
    Like create_spikes, but the start and end times are independently
    jittered by Uniform(-jitter, +jitter). Inter-spike intervals stay
    perfectly regular within the resulting [start', end'] window.
    """
    start_j = start + np.random.uniform(-jitter, jitter)
    end_j = end + np.random.uniform(-jitter, jitter)
    start_j = max(0.0, start_j)
    if end_j <= start_j:
        return []
    interval = 1.0 / Hz
    return list(np.arange(start_j, end_j, interval))

#%% main 

'''
Note: to simulate, have 3 options currently: 
1. just result = model.simulate() and let it run freely.
2. clamp A to a specific spike train, e.g. clamped_spikes={"A": [0.050, 0.100, 0.150, 0.200, 0.250]}
3. provide external drive to A via external_log_drive argument, e.g. external_log_drive=lambda t, names: np.array([2.0 if name == "A" and 0.2 <= t <= 0.6 else 0.0 for name in names])  
'''

global_syn_weight = 8.0
global_synapse_defaults = {
    "weight": global_syn_weight,
    "delay_s": 0.0018, # 1.8ms from shiu et al 2025
    "p_mode": "neuron_constant",
    "q_mode": "fixed", # fixed = every successful release uses q_mean (release amplitude)
    "q_mean": 1.0,
    "jitter_sd_s": 0.0005, # 0.5ms latency jitter
    "share_release_across_partners": True,
    "share_amplitude_across_partners": True,
    "share_jitter_across_partners": True,
    "conserve_amplitude_across_partners": False, # if true divides Q by the number of partners at this site
    "stp_enabled": False,
}

synapse_defaults_by_pre = None
synapse_params = None # could be used for exact synapse-by-synapse params 
global_base_rate_hz = 0.0
global_p_release = 0.5
self_history_weight = -3.0 # negative values create refractoriness, positive values create self-excitation
self_history_tau_s = 0.020 # decay time constant of the self-history effect
mean_polyadic_fraction = 0.4
mean_polyadic_size = 4.0
mean_outgoing = 20
sd_outgoing = 5

# input params 
input_frequency_hz = 80.0
input_start = 0.0
input_end = 5.0
jitter_input = 0.002 # add some jitter to input spike times to avoid perfect synchrony across partners
n_input_neurons = 2

# get network 
neurons, connectivity, synapse_defaults_by_pre, inhibitory_neurons = create_connectome(n_neurons = 10, n_inhibitory= 2, \
                        mean_outgoing = mean_outgoing, sd_outgoing = sd_outgoing, \
                        mean_polyadic_fraction = mean_polyadic_fraction, sd_polyadic_fraction = 0.3,
                        mean_polyadic_size = mean_polyadic_size,   # mean # partners when site is polyadic
                        sd_polyadic_size = 1.0, syn_weight = global_syn_weight)
flat_connectivity = flatten_connectome(connectivity)

neuron_params = { neu: {"base_rate_hz": global_base_rate_hz, "p_release": global_p_release, 
          "self_history_weight": self_history_weight, "self_history_tau_s": self_history_tau_s} for neu in neurons}


n_sims = 10
spikes = []
flat_spikes = []
for sim_id in range(n_sims):
    print(f"Running simulation {sim_id+1}/{n_sims}")
    model = LinkedReleasePPGLM.from_simple_connectivity(
        neurons=neurons,
        connectivity=connectivity,
        neuron_params=neuron_params,
        global_synapse_defaults=global_synapse_defaults,
        synapse_defaults_by_pre=synapse_defaults_by_pre,
        synapse_params=synapse_params,
        dt_s=0.001,
        t_stop_s=5.0,
        seed=sim_id, # different seed for each sim
    )


    flat_model = LinkedReleasePPGLM.from_simple_connectivity(
        neurons=neurons,
        connectivity=flat_connectivity,
        neuron_params=neuron_params,
        global_synapse_defaults=global_synapse_defaults,
        synapse_defaults_by_pre=synapse_defaults_by_pre,
        synapse_params=synapse_params,
        dt_s=0.001,
        t_stop_s=5.0,
        seed=1,
    )

    excitatory_neurons = [n for n in neurons if n not in inhibitory_neurons]
    input_neuron = excitatory_neurons[:n_input_neurons]

    clamped_spikes = {inp: create_spikes_with_endpoint_jitter(input_start, input_end, input_frequency_hz, jitter_input) for inp in input_neuron}

    stime = time.time()
    result = model.simulate(
        clamped_spikes=clamped_spikes
    )
    print(f"Simulation {sim_id+1} took {time.time() - stime:.4f} seconds")
    flat_result = flat_model.simulate(
        clamped_spikes=clamped_spikes
    )

    spikes.append(result.spikes.assign(sim_id=sim_id))
    flat_spikes.append(flat_result.spikes.assign(sim_id=sim_id))


# plotting helpers to visualise results 
mpl.rcParams.update({
    'font.size': 16, 
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# result has fields: spikes, release_events, lambda_hz, syn_drive, time_s, neuron_names
# just plot example :) 
fig, ax = plot_spikes(result, title='Polyadic')
fig, ax = plot_spikes(flat_result, title='Flattened')

# examine spikes 
spikes_df = pd.concat(spikes, ignore_index=True)
flat_spikes_df = pd.concat(flat_spikes, ignore_index=True)

combined_spikes = (
    pd.concat([spikes_df, flat_spikes_df], keys=["polyadic", "flat"], names=["arch"])
      .reset_index(level="arch")
      .reset_index(drop=True)
)
combined_spikes["arch"] = pd.Categorical(
    combined_spikes["arch"], categories=["polyadic", "flat"]
)
fig, ax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [2, 1]}, sharey=False)
s= 5
sns.stripplot(data=combined_spikes.groupby(['neuron', 'arch', 'sim_id']).size().reset_index(name='count'), x='neuron', y='count', hue='arch', dodge=True, palette="muted",s=s, ax=ax[0])
sns.stripplot(data=combined_spikes.groupby(['arch', 'sim_id']).size().reset_index(name='count'), x='arch', y='count', palette="muted", s=s,ax=ax[1])
ax[0].set_ylabel("Spike count")
ax[1].set_ylabel("Total spike count")
fig.tight_layout()





# %% visualise network

G = build_connectivity_graph(model)
fig, ax = plot_connectivity_graph(G, inhibitory_neurons=inhibitory_neurons)

# %%
