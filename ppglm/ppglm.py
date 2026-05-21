from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import time
import ast
import json

import numpy as np
import pandas as pd
import argparse


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

    # adding this pretty complicated extra thing here to allow different exc.inh effect dependent on partner identity 
    # to enable inhibition for postsyn. neurons that are known to express receptors that cause inhibition in response to acetylcholine only on their axons
    partner_weight_multipliers: Optional[Dict[str, float]] = None

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
        "partner_weight_multipliers",
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
        record_failed_release_attempts: bool = False,
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

        # Pre-computed arrays for the simulation hot path.
        self._neuron_p_release = np.array(
            [self.neuron_params[n].p_release for n in self.neurons], dtype=float
        )
        self._stp_site_indices: List[int] = [
            i for i, s in enumerate(self.release_sites) if s.stp_enabled
        ]
        self._build_site_cache()
        self._build_global_csr()

    # -----------------------------------------------------------------
    # Builders
    # -----------------------------------------------------------------

    def _build_site_cache(self) -> None:
        """Pre-compute per-site arrays used in the simulation hot path."""
        self._cache_post_idx: List[np.ndarray] = []
        self._cache_eff_weight: List[np.ndarray] = []
        self._cache_delay_bin: List[int] = []

        for site in self.release_sites:
            post_idxs = np.array(
                [self.name_to_idx[p] for p in site.partners], dtype=np.intp
            )
            if site.partner_weight_multipliers:
                mults = np.array(
                    [site.partner_weight_multipliers.get(p, 1.0) for p in site.partners],
                    dtype=float,
                )
            else:
                mults = np.ones(len(site.partners), dtype=float)

            self._cache_post_idx.append(post_idxs)
            self._cache_eff_weight.append((site.weight * mults))
            self._cache_delay_bin.append(int(round(site.delay_s / self.dt_s)))

    def _build_global_csr(self) -> None:
        """
        Build a single global CSR layout: neuron → site → partner.

        Eligible neurons (constant p, fixed q, shared release/jitter, no STP)
        are processed in one vectorised batch per time bin.  Others fall back
        to the Python site loop.
        """
        _ok_p = {"neuron_constant", "site_constant", "site_beta_static", "neuron_beta_static"}

        # ---- neuron → site CSR ----
        neuron_site_start = np.zeros(self.n_neurons + 1, dtype=np.int64)
        is_eligible = np.ones(self.n_neurons, dtype=bool)

        for pre_idx in range(self.n_neurons):
            site_indices = self.sites_by_pre_idx.get(pre_idx, [])
            neuron_site_start[pre_idx + 1] = len(site_indices)
            if not all(
                (self.release_sites[si].p_mode or self.default_p_mode) in _ok_p
                and self.release_sites[si].q_mode == "fixed"
                and self.release_sites[si].share_release_across_partners
                and self.release_sites[si].share_jitter_across_partners
                and not self.release_sites[si].stp_enabled
                for si in site_indices
            ):
                is_eligible[pre_idx] = False

        np.cumsum(neuron_site_start, out=neuron_site_start)
        n_total_sites = int(neuron_site_start[-1])

        site_p_release    = np.empty(n_total_sites, dtype=np.float64)
        site_q_mean       = np.empty(n_total_sites, dtype=np.float64)
        site_delay_bin    = np.empty(n_total_sites, dtype=np.int64)
        site_jitter_sd    = np.empty(n_total_sites, dtype=np.float64)
        site_partner_start = np.zeros(n_total_sites + 1, dtype=np.int64)

        flat_si = 0
        for pre_idx in range(self.n_neurons):
            pre = self.neurons[pre_idx]
            for site_idx in self.sites_by_pre_idx.get(pre_idx, []):
                site = self.release_sites[site_idx]
                mode = site.p_mode or self.default_p_mode
                if mode == "neuron_constant":
                    p = float(self._neuron_p_release[pre_idx])
                elif mode == "site_constant":
                    p = float(site.p_release if site.p_release is not None else self._neuron_p_release[pre_idx])
                elif mode == "site_beta_static":
                    p = float(self._site_beta_static_p[site.uid])
                elif mode == "neuron_beta_static":
                    p = float(self._neuron_beta_static_p[pre])
                else:
                    p = 0.5  # ineligible neuron — value unused
                site_p_release[flat_si]         = p
                site_q_mean[flat_si]            = site.q_mean
                site_delay_bin[flat_si]         = self._cache_delay_bin[site_idx]
                site_jitter_sd[flat_si]         = site.jitter_sd_s
                site_partner_start[flat_si + 1] = len(site.partners)
                flat_si += 1

        np.cumsum(site_partner_start, out=site_partner_start)
        n_total_partners = int(site_partner_start[-1])

        # ---- site → partner CSR ----
        partner_post_idx   = np.empty(n_total_partners, dtype=np.int64)
        partner_eff_weight = np.empty(n_total_partners, dtype=np.float64)

        flat_si = 0
        for pre_idx in range(self.n_neurons):
            for site_idx in self.sites_by_pre_idx.get(pre_idx, []):
                p_start = int(site_partner_start[flat_si])
                n_p = len(self._cache_post_idx[site_idx])
                partner_post_idx[p_start:p_start + n_p]   = self._cache_post_idx[site_idx]
                partner_eff_weight[p_start:p_start + n_p] = self._cache_eff_weight[site_idx]
                flat_si += 1

        self._csr_neuron_site_start  = neuron_site_start
        self._csr_is_eligible        = is_eligible
        self._csr_site_p_release     = site_p_release
        self._csr_site_q_mean        = site_q_mean
        self._csr_site_delay_bin     = site_delay_bin
        self._csr_site_jitter_sd     = site_jitter_sd
        self._csr_site_partner_start = site_partner_start
        self._csr_partner_post_idx   = partner_post_idx
        self._csr_partner_eff_weight = partner_eff_weight

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
        # s1 = time.time()
        n_bins = int(np.ceil(self.t_stop_s / self.dt_s))
        time_s = np.arange(n_bins) * self.dt_s

        max_delay_s = self._max_delay_s()
        padding_bins = int(np.ceil(max_delay_s / self.dt_s)) + 10

        # F-order makes pending_input[:, b] a contiguous column read/write
        # instead of strided (40 KB stride in C-order = cache miss every element).
        pending_input = np.zeros(
            (self.n_neurons, n_bins + padding_bins),
            dtype=float, order='F',
        )

        # Store traces as (n_bins, n_neurons) so each bin write is a contiguous
        # row rather than a strided column — avoids cache misses on every store.
        lambda_hz_T    = np.zeros((n_bins, self.n_neurons), dtype=float)
        syn_drive_T    = np.zeros((n_bins, self.n_neurons), dtype=float)

        syn_drive = np.zeros(self.n_neurons, dtype=float)
        self_history = np.zeros(self.n_neurons, dtype=float)

        stp_x = np.ones(self.n_sites, dtype=float)

        # s2 = time.time()
        # print(f"Initial setup time: {s2 - s1:.2f} seconds")

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
        # s3 = time.time()
        # print(f"Pre-compute parameter arrays time: {s3 - s2:.2f} seconds")

        forced = self._make_forced_spike_matrix(clamped_spikes, n_bins)
        # s4 = time.time()
        # print(f"Process clamped spikes time: {s4 - s3:.2f} seconds")
        spike_bins: List[np.ndarray] = []
        spike_nidx: List[np.ndarray] = []
        spike_lams: List[np.ndarray] = []
        release_records: List[Dict[str, Any]] = []

        syn_decay = np.exp(-self.dt_s / max(self.syn_tau_s, 1e-12))
        self_decay = np.exp(-self.dt_s / self_taus)
        log_max_rate = np.log(max(self.max_rate_hz, 1e-12))

        # Pre-allocate reusable buffers so the hot loop creates zero temp arrays.
        _eta  = np.empty(self.n_neurons, dtype=float)
        _lam  = np.empty(self.n_neurons, dtype=float)
        _buf  = np.empty(self.n_neurons, dtype=float)

        # Pre-compute external drive as (n_bins, n_neurons) — row-per-bin so
        # ext_T[b] is a contiguous read instead of a per-bin zeros allocation.
        if external_log_drive is None:
            ext_T = np.zeros((n_bins, self.n_neurons), dtype=float)
        elif callable(external_log_drive):
            ext_T = np.array(
                [external_log_drive(t, self.neurons) for t in time_s], dtype=float
            )
        else:
            ext_T = np.asarray(external_log_drive, dtype=float).T  # (n_bins, n_neurons)

        has_stp = bool(self._stp_site_indices)

        # s5 = time.time()
        # print(f"Pre-compute decay factors time: {s5 - s4:.2f} seconds")
        for b, t in enumerate(time_s):
            # T0 = time.time() if b == 0 else None

            # Update synaptic drive.
            syn_drive *= syn_decay
            syn_drive += pending_input[:, b]   # contiguous column read (F-order)

            # Update self-history.
            self_history *= self_decay

            # Recover STP variables (guarded — zero cost when STP is disabled).
            if has_stp:
                self._recover_stp(stp_x)

            ext = ext_T[b]   # free row slice, no allocation

            # T1 = time.time() if b == 0 else None
            # if b == 0: print(f"  [b=0] syn_drive + self_history + STP + ext:  {1e3*(T1-T0):.3f} ms")

            # Point-process GLM intensity in Hz — zero temp arrays via out= buffers.
            np.add(base_log_rates, ext, out=_eta)
            _eta += syn_drive
            _eta += self_history
            np.minimum(_eta, log_max_rate, out=_eta)  # clip in log-space before exp
            np.exp(_eta, out=_lam)

            lambda_hz_T[b] = _lam       # contiguous row write (cache-friendly)
            syn_drive_T[b] = syn_drive

            # Spike sampling: Exp(1) < lam*dt  ↔  U < 1 - exp(-lam*dt)
            # Avoids computing spike_prob explicitly; reuses _buf for exp draws.
            np.multiply(_lam, self.dt_s, out=_buf)
            self.rng.standard_exponential(out=_eta)   # reuse _eta as draw buffer
            spiked = _eta < _buf

            # T2 = time.time() if b == 0 else None
            # if b == 0: print(f"  [b=0] lam + spike sampling:                  {1e3*(T2-T1):.3f} ms")

            # Override generated spikes for clamped neurons.
            if forced is not None:
                clamped_mask = forced["clamped_neuron_mask"]
                forced_spikes = forced["forced_spikes"]
                spiked[clamped_mask] = forced_spikes[clamped_mask, b]

            spiking_indices = np.flatnonzero(spiked)

            # T3 = time.time() if b == 0 else None
            # if b == 0: print(f"  [b=0] forced override + flatnonzero:          {1e3*(T3-T2):.3f} ms  ({len(spiking_indices)} spikes)")

            if len(spiking_indices) > 0:
                # Accumulate spike records without per-spike dict allocation.
                spike_bins.append(np.full(len(spiking_indices), b, dtype=np.int32))
                spike_nidx.append(spiking_indices.astype(np.int32))
                spike_lams.append(_lam[spiking_indices])

                # Self-history: vectorised across all spiking neurons.
                self_history[spiking_indices] += self_weights[spiking_indices]

                # Split into eligible (CSR batch) and ineligible (Python fallback).
                elig_mask  = self._csr_is_eligible[spiking_indices]
                elig_pre   = spiking_indices[elig_mask]
                inelig_pre = spiking_indices[~elig_mask]

                # T4 = time.time() if b == 0 else None
                # if b == 0: print(f"  [b=0] spike bookkeeping + eligibility split:  {1e3*(T4-T3):.3f} ms  ({len(elig_pre)} elig, {len(inelig_pre)} inelig)")

                # ---- Vectorised path: all eligible spiking neurons in one batch ----
                if len(elig_pre) > 0:
                    site_counts = (self._csr_neuron_site_start[elig_pre + 1]
                                   - self._csr_neuron_site_start[elig_pre])
                    total_sites = int(site_counts.sum())

                    if total_sites > 0:
                        # Gather flat site indices for all elig_pre without a Python loop.
                        cumcounts = np.empty(len(elig_pre), dtype=np.int64)
                        cumcounts[0] = 0
                        if len(elig_pre) > 1:
                            np.cumsum(site_counts[:-1], out=cumcounts[1:])
                        within   = np.arange(total_sites, dtype=np.int64) - np.repeat(cumcounts, site_counts)
                        flat_si  = np.repeat(self._csr_neuron_site_start[elig_pre], site_counts) + within

                        # T5 = time.time() if b == 0 else None
                        # if b == 0: print(f"  [b=0] CSR site gather ({total_sites} sites):              {1e3*(T5-T4):.3f} ms")

                        released = self.rng.random(total_sites) < self._csr_site_p_release[flat_si]

                        # T6 = time.time() if b == 0 else None
                        # if b == 0: print(f"  [b=0] release draws ({released.sum()} released):          {1e3*(T6-T5):.3f} ms")

                        if released.any():
                            rsi   = flat_si[released]
                            jbins = np.round(
                                self.rng.standard_normal(rsi.size)
                                * self._csr_site_jitter_sd[rsi] / self.dt_s
                            ).astype(np.int64)
                            arr_bins = np.maximum(b + self._csr_site_delay_bin[rsi] + jbins, b + 1)

                            # Expand released sites → partners (same ragged trick).
                            p_counts = (self._csr_site_partner_start[rsi + 1]
                                        - self._csr_site_partner_start[rsi])
                            total_partners = int(p_counts.sum())

                            # T7 = time.time() if b == 0 else None
                            # if b == 0: print(f"  [b=0] jitter + arrival bins + partner count:  {1e3*(T7-T6):.3f} ms  ({total_partners} partners)")

                            if total_partners > 0:
                                pcum = np.empty(len(rsi), dtype=np.int64)
                                pcum[0] = 0
                                if len(rsi) > 1:
                                    np.cumsum(p_counts[:-1], out=pcum[1:])
                                pw      = np.arange(total_partners, dtype=np.int64) - np.repeat(pcum, p_counts)
                                flat_pi = np.repeat(self._csr_site_partner_start[rsi], p_counts) + pw

                                p_arr_bins = np.repeat(arr_bins, p_counts)
                                amplitudes = (np.repeat(self._csr_site_q_mean[rsi], p_counts)
                                              * self._csr_partner_eff_weight[flat_pi])

                                mask = p_arr_bins < pending_input.shape[1]

                                # T8 = time.time() if b == 0 else None
                                # if b == 0: print(f"  [b=0] partner gather + amplitudes:            {1e3*(T8-T7):.3f} ms")

                                if mask.any():
                                    np.add.at(
                                        pending_input,
                                        (self._csr_partner_post_idx[flat_pi[mask]], p_arr_bins[mask]),
                                        amplitudes[mask],
                                    )

                                # T9 = time.time() if b == 0 else None
                                # if b == 0: print(f"  [b=0] np.add.at pending_input:                {1e3*(T9-T8):.3f} ms")

                # ---- Python fallback: STP / gamma-Cox / non-fixed q ----
                for pre_idx in inelig_pre:
                    pre = self.neurons[pre_idx]
                    spike_context: Dict[str, float] = {}
                    for site_idx in self.sites_by_pre_idx.get(pre_idx, []):
                        site = self.release_sites[site_idx]
                        p_eff = self._draw_release_probability(
                            pre=pre, pre_idx=pre_idx, site=site,
                            stp_x_value=stp_x[site_idx], spike_context=spike_context,
                        )
                        self._process_release_site(
                            pre=pre, pre_idx=pre_idx, site_idx=site_idx, site=site,
                            parent_spike_time_s=t, current_bin=b, p_eff=p_eff,
                            stp_x=stp_x, pending_input=pending_input,
                            release_records=release_records,
                        )

            # if b == 0:
                # print(f"  [b=0] TOTAL first bin: {1e3*(time.time()-T0):.3f} ms")

        # Build spike DataFrame from accumulated arrays (no per-spike dict overhead).
        if spike_bins:
            all_bins  = np.concatenate(spike_bins)
            all_nidx  = np.concatenate(spike_nidx)
            all_lams  = np.concatenate(spike_lams)
            spikes_df = pd.DataFrame({
                "time_s":     time_s[all_bins],
                "neuron":     [self.neurons[i] for i in all_nidx],
                "neuron_idx": all_nidx.astype(int),
                "lambda_hz":  all_lams,
            })
        else:
            spikes_df = pd.DataFrame(columns=["time_s", "neuron", "neuron_idx", "lambda_hz"])
        release_df = pd.DataFrame(release_records)

        return SimulationResult(
            spikes=spikes_df,
            release_events=release_df,
            lambda_hz=lambda_hz_T.T,       # zero-copy view back to (n_neurons, n_bins)
            syn_drive=syn_drive_T.T,
            time_s=time_s,
            neuron_names=self.neurons,
        )

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
        post_idx = self._cache_post_idx[site_idx]       # pre-computed
        eff_weight = self._cache_eff_weight[site_idx]   # pre-computed
        delay_bin = self._cache_delay_bin[site_idx]     # pre-computed

        # Release success/failure — one draw if shared, one per partner otherwise.
        if site.share_release_across_partners:
            released = np.full(n_partners, self.rng.random() < p_eff)
        else:
            released = self.rng.random(n_partners) < p_eff

        # Amplitude Q.
        if site.q_mode == "fixed":
            q = np.full(n_partners, site.q_mean)
        elif site.share_amplitude_across_partners:
            q = np.full(n_partners, self._draw_amplitude(site))
        else:
            q = np.array([self._draw_amplitude(site) for _ in range(n_partners)])
        if site.conserve_amplitude_across_partners and n_partners > 0:
            q = q / n_partners

        # Latency jitter — vectorised draw.
        if site.jitter_sd_s == 0.0:
            jitter = np.zeros(n_partners)
        elif site.share_jitter_across_partners:
            jitter = np.full(n_partners, self.rng.normal(0.0, site.jitter_sd_s))
        else:
            jitter = self.rng.normal(0.0, site.jitter_sd_s, size=n_partners)

        # STP depletion on any success.
        if site.stp_enabled and released.any():
            stp_x[site_idx] = max(
                0.0, stp_x[site_idx] - site.stp_depletion_per_success
            )

        # Arrival bins (vectorised). parent_spike_time_s == current_bin * dt_s exactly.
        jitter_bins = np.round(jitter / self.dt_s).astype(int)
        arrival_bins = np.maximum(current_bin + delay_bin + jitter_bins, current_bin + 1)

        # Amplitudes and pending_input update (vectorised, safe for repeated post_idx).
        amplitudes = released.astype(float) * q * eff_weight
        mask = released & (arrival_bins < pending_input.shape[1])
        if mask.any():
            np.add.at(pending_input, (post_idx[mask], arrival_bins[mask]), amplitudes[mask])

        # Optional recording.
        if self.record_failed_release_attempts or released.any():
            for i in range(n_partners):
                if released[i] or self.record_failed_release_attempts:
                    release_records.append(
                        {
                            "parent_spike_time_s": parent_spike_time_s,
                            "arrival_time_s": float(arrival_bins[i]) * self.dt_s,
                            "pre": pre,
                            "pre_idx": pre_idx,
                            "synapse_id": site.synapse_id,
                            "site_uid": site.uid,
                            "post": site.partners[i],
                            "post_idx": int(post_idx[i]),
                            "released": bool(released[i]),
                            "p_release_eff": p_eff,
                            "q": float(q[i]),
                            "weight": site.weight,
                            "effective_weight": float(eff_weight[i]),
                            "amplitude": float(amplitudes[i]),
                            "delay_s": site.delay_s,
                            "jitter_s": float(jitter[i]),
                            "n_partners_at_site": n_partners,
                            "stp_x_after": float(stp_x[site_idx]),
                        }
                    )

    def _draw_release_probability(
        self,
        *,
        pre: str,
        pre_idx: int,
        site: ReleaseSite,
        stp_x_value: float,
        spike_context: Dict[str, float],
    ) -> float:
        mode = site.p_mode or self.default_p_mode

        if mode == "neuron_constant":
            p = self._neuron_p_release[pre_idx]

        elif mode == "site_constant":
            p = self._neuron_p_release[pre_idx] if site.p_release is None else site.p_release

        elif mode == "site_beta_static":
            p = self._site_beta_static_p[site.uid]

        elif mode == "neuron_beta_static":
            p = self._neuron_beta_static_p[pre]

        elif mode == "site_gamma_cox_per_spike":
            npar = self.neuron_params[pre]
            rho = self.rng.gamma(site.gamma_shape, site.gamma_scale)
            p = 1.0 - np.exp(-rho * site.gamma_cox_scale)

        elif mode == "neuron_gamma_cox_per_spike":
            key = f"neuron_gamma_cox:{pre_idx}"
            if key not in spike_context:
                npar = self.neuron_params[pre]
                rho = self.rng.gamma(npar.gamma_shape, npar.gamma_scale)
                spike_context[key] = 1.0 - np.exp(-rho * npar.gamma_cox_scale)
            p = spike_context[key]

        else:
            raise ValueError(f"Unknown p_mode: {mode!r}")

        p = float(p)
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0

        if site.stp_enabled:
            p *= max(0.0, min(1.0, stp_x_value))

        return p

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
        for site_idx in self._stp_site_indices:
            site = self.release_sites[site_idx]
            tau = max(site.stp_tau_recovery_s, 1e-12)
            recovery_fraction = 1.0 - np.exp(-self.dt_s / tau)
            stp_x[site_idx] += (1.0 - stp_x[site_idx]) * recovery_fraction
            stp_x[site_idx] = np.clip(stp_x[site_idx], 0.0, 1.0)

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

def load_connectome_from_csv(con_path: str, inhibitory_neurons_path: str, axo_axonic_path: str, neurons_path: str,
                             postsyn_col: str = 'postsynaptic_id', presyn_col: str = 'presynaptic_id', 
                             connector_col: str = 'connector_id',
                             syn_weight: float = 8.0):
    neuron_details = pd.read_csv(neurons_path)
    all_neurons = set(neuron_details['skeleton_id'].astype(int).tolist())
    con = pd.read_csv(con_path)
    con = con.dropna(subset=[presyn_col]) # drop rows with missing presynaptic_id

    con[postsyn_col] = con[postsyn_col].apply(ast.literal_eval)
    con[presyn_col] = con[presyn_col].astype(int)
    con[connector_col] = con[connector_col].astype(str)

    # drop presynaptic neurons not in all_neurons
    con = con[con[presyn_col].isin(all_neurons)]
    # drop any postsynaptic partners not in all_neurons
    con[postsyn_col] = con[postsyn_col].apply(lambda lst: [p for p in lst if p in all_neurons])
    # drop rows with empty postsynaptic partners
    con = con[con[postsyn_col].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    # convert to connectivity dict format: {pre: {synapse_id: [postsynaptic partners]}}
    connectivity: Dict[int, Dict[str, List[int]]] = {}
    con.groupby(presyn_col).apply(lambda g: connectivity.setdefault(int(g.name), {}).update(
        {connector_id: postsyn_list for connector_id, postsyn_list in zip(g[connector_col], g[postsyn_col])}
    ), include_groups=False)

    # inhibitory neurons
    inh = pd.read_csv(inhibitory_neurons_path)
    inhibitory_neurons = set(inh['skid'].astype(int).tolist())

    # get neurons for sim
    neurons = list(pd.concat([con[presyn_col], con[postsyn_col].explode()]).unique())

    inhibitory_neurons = set([n for n in inhibitory_neurons if n in neurons])

    # synapse defaults by pre - make inhibitory neurons have negative weights
    synapse_defaults_by_pre = {
        neuron: {"weight": -syn_weight if neuron in inhibitory_neurons else syn_weight}
        for neuron in neurons
    }

    # kc aco axonic synapses with special synapses - should be inhibitory if axo-axonic KC to KC
    axo_axonic = pd.read_csv(axo_axonic_path) # should have same column names as con_path
    if presyn_col not in axo_axonic.columns:
        axo_axonic.rename(columns={'presynaptic_to': presyn_col}, inplace=True)
    if postsyn_col not in axo_axonic.columns:
        axo_axonic.rename(columns={'postsynaptic_to': postsyn_col}, inplace=True)
    axo_axonic[presyn_col] = axo_axonic[presyn_col].astype(int)
    axo_axonic[connector_col] = axo_axonic[connector_col].astype(str)
    axo_axonic[postsyn_col] = axo_axonic[postsyn_col].apply(ast.literal_eval)

    # use synapse_params with partner_weight_multipliers to invert effect
    synapse_params: Dict[int, Dict[str, Dict[int, Any]]] = {}
    for _, row in axo_axonic.iterrows():
        pre = int(row[presyn_col])
        synapse_id = row[connector_col]
        partners = row[postsyn_col]

        if pre not in synapse_params:
            synapse_params[pre] = {}

        synapse_params[pre][synapse_id] = {
            "partner_weight_multipliers": {int(partner): -1.0 for partner in partners}
        }
    
    return neurons, connectivity, synapse_defaults_by_pre, inhibitory_neurons, synapse_params






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
    synapse_params: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Expand polyadic synapses into monadic ones, keeping the same
    nested-dict format as `connectivity`.

    Each (pre, synapse_id) with N>1 partners becomes N synapses
    (synapse_id_0, synapse_id_1, ...), each with a single partner.

    synapse_params entries for split synapses are propagated to each
    sub-synapse. `partner_weight_multipliers` is filtered to only the
    single partner that sub-synapse targets.

    Returns (flat_connectivity, flat_synapse_params).
    """
    flat: Dict[str, Dict[str, List[str]]] = {}
    flat_synapse_params: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for pre, synapse_dict in connectivity.items():
        flat[pre] = {}
        pre_params = (synapse_params or {}).get(pre, {})

        for synapse_id, partners in synapse_dict.items():
            partners = list(partners)
            original_params = pre_params.get(synapse_id, {})

            if len(partners) <= 1:
                flat[pre][synapse_id] = list(partners)
                if original_params:
                    flat_synapse_params.setdefault(pre, {})[synapse_id] = dict(original_params)
            else:
                for i, post in enumerate(partners):
                    new_id = f"{synapse_id}_{i}"
                    flat[pre][new_id] = [post]

                    if original_params:
                        new_params = dict(original_params)
                        pwm = new_params.get("partner_weight_multipliers")
                        if pwm is not None:
                            filtered = {post: pwm[post]} if post in pwm else {}
                            if filtered:
                                new_params["partner_weight_multipliers"] = filtered
                            else:
                                del new_params["partner_weight_multipliers"]
                        if new_params:
                            flat_synapse_params.setdefault(pre, {})[new_id] = new_params

    return flat, flat_synapse_params


def create_spikes(start: float, end: float, Hz: float) -> List[float]:
    interval = 1.0 / Hz
    return list(np.arange(start, end, interval))

def create_spikes_with_endpoint_jitter(
    start: float,
    end: float,
    Hz: float,
    jitter: float,
    rng=None,
) -> List[float]:
    """
    Like create_spikes, but the start and end times are independently
    jittered by Uniform(-jitter, +jitter). Inter-spike intervals stay
    perfectly regular within the resulting [start', end'] window.
    """
    _u = rng.uniform if rng is not None else np.random.uniform
    start_j = start + _u(-jitter, jitter)
    end_j = end + _u(-jitter, jitter)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n_sims", type=int, default=10, help="Number of simulations to run.")
    # main synaptic args
    parser.add_argument("--global_syn_weight", type=float, default=3.5, help="Global synaptic weight for all synapses.")
    parser.add_argument("--pr", type=float, default=0.1, help="Global release probability for all synapses.")
    # input 
    parser.add_argument("--input_frequency_hz", type=float, default=80.0, help="Firing rate of input neurons in Hz.")   
    args = parser.parse_args()


    _rng = np.random.default_rng(args.seed)

    global_syn_weight = args.global_syn_weight
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

    # neuron params 
    global_base_rate_hz = 0.0
    global_p_release = args.pr
    self_history_weight = -10.0 # negative values create refractoriness, positive values create self-excitation
    # it acts as a rate of suppression, more negative means stronger suppression 
    # -10 probably suppresses neuron for too long
    self_history_tau_s = 0.0022 # decay time constant of the self-history effect - 2.2ms refractory period 

    # input params 
    input_frequency_hz = args.input_frequency_hz
    input_start = 0.0
    input_end = 5.0
    jitter_input = 0.002 # add some jitter to input spike times to avoid perfect synchrony across partners

    # input params
    n_input_neurons = 4

    # restrict to just kcs for instance 
    neuron_details = pd.read_csv("data/neuron_details.csv")
    potential_inputs = neuron_details[neuron_details["name"].str.startswith('42') & neuron_details["name"].str.contains('ORN')]["skeleton_id"].tolist() # None is an option for random connectome

    ### REAL NETWORK 
    neurons, connectivity, synapse_defaults_by_pre, inhibitory_neurons, synapse_params = load_connectome_from_csv(
        con_path="data/polyadic_connectors.csv",
        inhibitory_neurons_path="data/2026_02_inhibitory_neurons.csv",
        axo_axonic_path="data/kc_axo_axonic.csv",
        neurons_path="data/neuron_details.csv",
        syn_weight=global_syn_weight,
    )

    ### RANDOM NETWORK 
    # connectome creation parameters 
    # mean_polyadic_fraction = 0.4
    # mean_polyadic_size = 4.0
    # mean_outgoing = 20
    # sd_outgoing = 5

    # neurons, connectivity, synapse_defaults_by_pre, inhibitory_neurons = create_connectome(n_neurons = 10, n_inhibitory= 2, \
    #                         mean_outgoing = mean_outgoing, sd_outgoing = sd_outgoing, \
    #                         smean_polyadic_fraction = mean_polyadic_fraction, sd_polyadic_fraction = 0.3,
    #                         mean_polyadic_size = mean_polyadic_size,   # mean # partners when site is polyadic
    #                         sd_polyadic_size = 1.0, syn_weight = global_syn_weight)
    # synapse_params = None 


    ### FLATTEN NETWORK
    flat_connectivity, flat_synapse_params = flatten_connectome(connectivity, synapse_params=synapse_params)


    ### Create neuron params 
    neuron_params = { neu: {"base_rate_hz": global_base_rate_hz, "p_release": global_p_release, 
            "self_history_weight": self_history_weight, "self_history_tau_s": self_history_tau_s} for neu in neurons}

    ### Get inputs 
    excitatory_neurons = [n for n in neurons if n not in inhibitory_neurons]
    if potential_inputs is None:
        potential_inputs = excitatory_neurons

    # choose a random set of those
    input_neuron = _rng.choice(potential_inputs, size=min(n_input_neurons, len(potential_inputs)), replace=False).tolist()

    n_sims = args.n_sims
    spikes = []
    flat_spikes = []
    for sim_id in range(n_sims):
        # print(f"Running simulation {sim_id+1}/{n_sims}")
        model = LinkedReleasePPGLM.from_simple_connectivity(
            neurons=neurons,
            connectivity=connectivity,
            neuron_params=neuron_params,
            global_synapse_defaults=global_synapse_defaults,
            synapse_defaults_by_pre=synapse_defaults_by_pre,
            synapse_params=synapse_params,
            dt_s=0.001,
            t_stop_s=5.0,
            seed=int(_rng.integers(2**31)),
        )

        flat_model = LinkedReleasePPGLM.from_simple_connectivity(
            neurons=neurons,
            connectivity=flat_connectivity,
            neuron_params=neuron_params,
            global_synapse_defaults=global_synapse_defaults,
            synapse_defaults_by_pre=synapse_defaults_by_pre,
            synapse_params=flat_synapse_params,
            dt_s=0.001,
            t_stop_s=5.0,
            seed=int(_rng.integers(2**31)),
        )

        clamped_spikes = {inp: create_spikes_with_endpoint_jitter(input_start, input_end, input_frequency_hz, jitter_input, rng=_rng) for inp in input_neuron}

        stime = time.time()
        result = model.simulate(
            clamped_spikes=clamped_spikes
        )
        
        print(f"Polyadic simulation {sim_id+1} took {time.time() - stime:.4f} seconds")
        stime = time.time()
        flat_result = flat_model.simulate(
            clamped_spikes=clamped_spikes
        )
        print(f"Flat simulation {sim_id+1} took {time.time() - stime:.4f} seconds")

        spikes.append(result.spikes.assign(sim_id=sim_id))
        flat_spikes.append(flat_result.spikes.assign(sim_id=sim_id))


    # save results
    all_spikes = pd.concat(spikes, ignore_index=True)
    all_flat_spikes = pd.concat(flat_spikes, ignore_index=True)
    all_spikes['arch'] = 'full'
    all_flat_spikes['arch'] = 'flat'
    combined = pd.concat([all_spikes, all_flat_spikes], ignore_index=True)
    pr_str = f"{global_p_release:.2f}".replace('.', 'p')
    syn_scale_str = f"{global_syn_weight:.1f}".replace('.', 'p')
    output = {
        "seed": args.seed,
        "n_sims": args.n_sims,
        "global_syn_weight": args.global_syn_weight,
        "pr": args.pr,
        "input_frequency_hz": args.input_frequency_hz,
        "data": combined.to_dict(orient='records'),
    }
    with open(f'out/res_seed{args.seed}_pr0{pr_str}_synscale{syn_scale_str}_inphz{int(args.input_frequency_hz)}.json', 'w') as f:
        json.dump(output, f)
