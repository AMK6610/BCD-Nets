import numpy as onp
import jax.numpy as jnp
from typing import Tuple, Optional, cast
import itertools
import warnings
import sys
import pickle as pkl


warnings.simplefilter(action="ignore", category=FutureWarning)

from .doubly_stochastic import GumbelSinkhorn
import jax.random as rnd
from jax import vmap, grad, jit, lax, pmap, partial, value_and_grad
from .utils import (
    lower,
    eval_W_non_ev,
    eval_W_ev,
    ff2,
    num_params,
    save_params,
    get_variance,
    get_variances,
    from_W,
    rk,
    get_double_tree_variance,
    un_pmap,
    auroc,
)
from jax.tree_util import tree_map, tree_multimap

import jax
from tensorflow_probability.substrates.jax.distributions import (
    Normal,
    Horseshoe,
)

import matplotlib.pyplot as plt
import matplotlib as mpl
from jax import config
import haiku as hk
from .models import (
    get_model,
    get_model_arrays,
)
import time
from jax.flatten_util import ravel_pytree
import optax
from PIL import Image
from .flows import get_flow_CIF

from .golem_utils import solve_golem_cv, bootstrapped_golem_cv
from argparse import ArgumentParser
from .baselines import run_all_baselines, eval_W_samples
from .metrics import intervention_distance, ensemble_intervention_distance
from ._types import PParamType, LStateType

print("finished imports")

config.update("jax_enable_x64", True)
import jax


mpl.rcParams["figure.dpi"] = 300


PRNGKey = jnp.ndarray
QParams = Tuple[jnp.ndarray, hk.Params]

def init_params(args):
    global pars
    global tau
    pars = args
    if pars['fixed_tau'] is not None:
        tau = pars['fixed_tau']
    else:
        tau = tau_schedule(0)



def init_parallel_params(rng_key: PRNGKey):
    @pmap
    def init_params(rng_key: PRNGKey):
        if pars['use_flow']:
            L_params, L_states = get_flow_arrays()
        else:
            L_params = jnp.concatenate(
                (
                    jnp.zeros(pars['l_dim']),
                    jnp.zeros(pars['noise_dim']),
                    jnp.zeros(pars['l_dim'] + pars['noise_dim']) - 1,
                )
            )
            # Would be nice to put none here, but need to pmap well
            L_states = jnp.array([0.0])
        P_params = get_model_arrays(
            pars['num_nodes'],
            pars['batch_size'],
            pars['num_perm_layers'],
            rng_key,
            hidden_size=pars['hidden_size'],
            do_ev_noise=pars['do_ev_noise'],
        )
        if pars['factorized']:
            P_params = jnp.zeros((pars['num_nodes'], pars['num_nodes']))
        P_opt_params = pars['opt_P'].init(P_params)
        L_opt_params = pars['opt_L'].init(L_params)
        return (
            P_params,
            L_params,
            L_states,
            P_opt_params,
            L_opt_params,
        )

    rng_keys = jnp.tile(rng_key[None, :], (pars['num_devices'], 1))
    output = init_params(rng_keys)
    return output

def get_P_logits(
    P_params: PParamType, L_samples: jnp.ndarray, rng_key: PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    if pars['factorized']:
        # We ignore L when giving the P parameters
        assert type(P_params) is jnp.ndarray
        p_logits = jnp.tile(P_params.reshape((1, pars['num_nodes'], pars['num_nodes'])), (len(L_samples), 1, 1))
    else:
        P_params = cast(hk.Params, P_params)
        p_logits = pars['p_model'](P_params, rng_key, L_samples)  # type:ignore

    if pars['logit_constraint'] is not None:
        # Want to map -inf to -logit_constraint, inf to +logit_constraint
        p_logits = jnp.tanh(p_logits / pars['logit_constraint']) * pars['logit_constraint']

    return p_logits.reshape((-1, pars['num_nodes'], pars['num_nodes']))


def sample_L(
    L_params: PParamType, L_state: LStateType, rng_key: PRNGKey, n: int
) -> Tuple[jnp.ndarray, jnp.ndarray, LStateType]:
    if pars['use_flow']:
        L_state = cast(hk.State, L_state)
        L_params = cast(hk.State, L_params)
        full_l_batch, full_log_prob_l, out_L_states = sample_flow(
            L_params, L_state, rng_key, n
        )
        return full_l_batch, full_log_prob_l, out_L_states
    else:
        L_params = cast(jnp.ndarray, L_params)
        means, log_stds = L_params[: pars['l_dim'] + pars['noise_dim']], L_params[pars['l_dim'] + pars['noise_dim'] :]
        if pars['log_stds_max'] is not None:
            # Do a soft-clip here to stop instability
            log_stds = jnp.tanh(log_stds / pars['log_stds_max']) * pars['log_stds_max']
        L_dist = pars['L_dist']
        l_distribution = L_dist(loc=means, scale=jnp.exp(log_stds))
        if L_dist is Normal:
            full_l_batch = l_distribution.sample(
                seed=rng_key, sample_shape=(n,)
            )
            full_l_batch = cast(jnp.ndarray, full_l_batch)
        else:
            full_l_batch = (
                rnd.laplace(rng_key, shape=(n, pars['l_dim'] + pars['noise_dim']))
                * jnp.exp(log_stds)[None, :]
                + means[None, :]
            )
        full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)
        full_log_prob_l = cast(jnp.ndarray, full_log_prob_l)

        out_L_states = None
        return full_l_batch, full_log_prob_l, out_L_states


def log_prob_x(Xs, log_sigmas, P, L, interv_targets, rng_key):
    """Calculates log P(X|Z) for latent Zs

    X|Z is Gaussian so easy to calculate

    Args:
        Xs: an (n x dim)-dimensional array of observations
        log_sigmas: A (dim)-dimension vector of log standard deviations
        P: A (dim x dim)-dimensional permutation matrix
        L: A (dim x dim)-dimensional strictly lower triangular matrix
        interv_targets: A (n x dim)-dimensional boolean array corresponding to the nodes intervened
    Returns:
        log_prob: Log probability of observing Xs given P, L
    """
    if pars['subsample']:
        num_full_xs = len(Xs)
        X_batch_size = 16
        adjustment_factor = num_full_xs / X_batch_size
        Xs = rnd.shuffle(rng_key, Xs)[:X_batch_size]
    else:
        adjustment_factor = 1
    n, dim = Xs.shape
    W = (P @ L @ P.T).T
    precision = (
        (jnp.eye(dim) - W) @ (jnp.diag(jnp.exp(-2 * log_sigmas))) @ (jnp.eye(dim) - W).T
    )
    # eye_minus_W_logdet = 0
    # log_det_precision = -2 * jnp.sum(log_sigmas) + 2 * eye_minus_W_logdet

    interv_log_sigmas = jnp.tile(log_sigmas, (Xs.shape[0], 1))
    interv_log_sigmas = jnp.where(interv_targets, 0.0, interv_log_sigmas)
    log_det_precision = -jnp.sum(interv_log_sigmas)

    def datapoint_exponent(x):
        return -0.5 * x.T @ precision @ x

    log_exponent = vmap(datapoint_exponent)(jnp.where(interv_targets, 0.0, Xs))

    return adjustment_factor * (
        log_det_precision - 0.5 * (jnp.sum(~interv_targets)) * jnp.log(2 * jnp.pi)
        + jnp.sum(log_exponent)
    )
    # return adjustment_factor * (
    #     0.5 * n * (log_det_precision - dim * jnp.log(2 * jnp.pi))
    #     + jnp.sum(log_exponent)
    # )


def elbo(
    P_params: PParamType,
    L_params: hk.Params,
    L_states: LStateType,
    Xs: jnp.ndarray,
    interv_targets: jnp.ndarray,
    rng_key: PRNGKey,
    tau: float,
    num_outer: int = 1,
    hard: bool = False,
) -> Tuple[jnp.ndarray, LStateType]:
    """Computes ELBO estimate from parameters.

    Computes ELBO(P_params, L_params), given by
    E_{e1}[E_{e2}[log p(x|L, P)] - D_KL(q_L(P), p(L)) -  log q_P(P)],
    where L = g_L(L_params, e2) and P = g_P(P_params, e1).
    The derivative of this corresponds to the pathwise gradient estimator

    Args:
        P_params: inputs to sampling path functions
        L_params: inputs parameterising function giving L|P distribution
        Xs: (n x dim)-dimension array of inputs
        rng_key: jax prngkey object
        log_sigma_W: (dim)-dimensional array of log standard deviations
        log_sigma_l: scalar prior log standard deviation on (Laplace) prior on l.
    Returns:
        ELBO: Estimate of the ELBO
    """
    num_bethe_iters = 20
    l_prior = Horseshoe(scale=jnp.ones(pars['l_dim'] + pars['noise_dim']) * pars['horseshoe_tau'])
    # else:
    #     l_prior = Laplace(
    #         loc=jnp.zeros(pars['l_dim'] + pars['noise_dim']),
    #         scale=jnp.ones(pars['l_dim'] + pars['noise_dim']) * jnp.exp(log_sigma_l),
    #     )

    def outer_loop(rng_key: PRNGKey):
        """Computes a term of the outer expectation, averaging over batch size"""
        rng_key, rng_key_1 = rnd.split(rng_key, 2)
        full_l_batch, full_log_prob_l, out_L_states = sample_L(
            L_params, L_states, rng_key, pars['batch_size']
        )
        w_noise = full_l_batch[:, -pars['noise_dim']:]
        l_batch = full_l_batch[:, :-pars['noise_dim']]
        batched_noises = jnp.ones((pars['batch_size'], pars['num_nodes'])) * w_noise.reshape(
            (pars['batch_size'], pars['noise_dim'])
        )
        batched_lower_samples = vmap(lower, in_axes=(0, None))(l_batch, pars['num_nodes'])
        batched_P_logits = get_P_logits(P_params, full_l_batch, rng_key_1)
        if hard:
            batched_P_samples = pars['ds'].sample_hard_batched_logits(
                batched_P_logits, tau, rng_key,
            )
        else:
            batched_P_samples = pars['ds'].sample_soft_batched_logits(
                batched_P_logits, tau, rng_key,
            )
        # interv_targets = jnp.zeros(Xs.shape, dtype=bool)
        # interv_targets = interv_targets.at[0].set(True)
        likelihoods = vmap(log_prob_x, in_axes=(None, 0, 0, 0, None, None))(
            Xs, batched_noises, batched_P_samples, batched_lower_samples, interv_targets, rng_key,
        )
        l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch)[:, :pars['l_dim']], axis=1)
        s_prior_probs = jnp.sum(
            full_l_batch[:, pars['l_dim']:] ** 2 / (2 * pars['s_prior_std'] ** 2), axis=-1
        )
        KL_term_L = full_log_prob_l - l_prior_probs - s_prior_probs
        logprob_P = vmap(pars['ds'].logprob, in_axes=(0, 0, None))(
            batched_P_samples, batched_P_logits, num_bethe_iters
        )
        log_P_prior = -jnp.sum(jnp.log(onp.arange(pars['num_nodes']) + 1))
        final_term = likelihoods - KL_term_L - logprob_P + log_P_prior

        return jnp.mean(final_term), out_L_states

    rng_keys = rnd.split(rng_key, num_outer)
    _, (elbos, out_L_states) = lax.scan(
        lambda _, rng_key: (None, outer_loop(rng_key)), None, rng_keys
    )
    elbo_estimate = jnp.mean(elbos)
    return elbo_estimate, tree_map(lambda x: x[-1], out_L_states)


def eval_mean(
    P_params, L_params, L_states, Xs, rng_key=rk(0), do_shd_c=False, tau=1,
):
    """Computes mean error statistics for P, L parameters and data"""
    P_params, L_params, L_states = (
        un_pmap(P_params),
        un_pmap(L_params),
        un_pmap(L_states),
    )

    if pars['do_ev_noise']:
        eval_W_fn = eval_W_ev
    else:
        eval_W_fn = eval_W_non_ev
    _, dim = Xs.shape
    x_prec = onp.linalg.inv(jnp.cov(Xs.T))
    full_l_batch, _, _ = sample_L(L_params, L_states, rng_key, pars['n_metric_samples'])
    w_noise = full_l_batch[:, -pars['noise_dim']:]
    l_batch = full_l_batch[:, :-pars['noise_dim']]
    batched_lower_samples = jit(vmap(lower, in_axes=(0, None)), static_argnums=(1,))(
        l_batch, dim
    )
    batched_P_logits = jit(get_P_logits)(P_params, full_l_batch, rng_key)
    batched_P_samples = jit(pars['ds'].sample_hard_batched_logits)(
        batched_P_logits, tau, rng_key
    )

    def sample_W(L, P):
        return (P @ L @ P.T).T

    Ws = jit(vmap(sample_W))(batched_lower_samples, batched_P_samples)

    def sample_stats(W, noise):
        stats = eval_W_fn(
            W,
            pars['ground_truth_W'],
            jnp.exp(pars['log_sigma_W']),
            0.3,
            Xs,
            jnp.ones(dim) * jnp.exp(noise),
            provided_x_prec=x_prec,
            do_shd_c=do_shd_c,
            do_sid=do_shd_c,
        )
        return stats

    stats = sample_stats(Ws[0], w_noise[0])
    stats = {key: [stats[key]] for key in stats}
    for i, W in enumerate(Ws[1:]):
        new_stats = sample_stats(W, w_noise[i])
        for key in new_stats:
            stats[key] = stats[key] + [new_stats[key]]

    # stats = vmap(sample_stats)(rng_keys)
    out_stats = {key: onp.mean(stats[key]) for key in stats}
    out_stats["auroc"] = auroc(Ws, pars['ground_truth_W'], 0.3)
    return out_stats


def get_num_sinkhorn_steps(P_params, L_params, L_states, rng_key):
    P_params, L_params, L_states, rng_key = (
        un_pmap(P_params),
        un_pmap(L_params),
        un_pmap(L_states),
        un_pmap(rng_key),
    )

    full_l_batch, _, _ = jit(sample_L, static_argnums=(3,))(L_params, L_states, rng_key, pars['batch_size'])
    batched_P_logits = jit(get_P_logits)(P_params, full_l_batch, rng_key)
    _, errors = jit(pars['ds'].sample_hard_batched_logits_debug)(
        batched_P_logits, tau, rng_key,
    )
    first_converged = jnp.where(jnp.sum(errors, axis=0) == -pars['batch_size'])[0]
    if len(first_converged) == 0:
        converged_idx = -1
    else:
        converged_idx = first_converged[0]
    return converged_idx


def eval_ID(P_params, L_params, L_states, Xs, rng_key, tau):
    """Computes mean error statistics for P, L parameters and data"""
    P_params, L_params, L_states = (
        un_pmap(P_params),
        un_pmap(L_params),
        un_pmap(L_states),
    )

    _, dim = Xs.shape
    full_l_batch, _, _ = jit(sample_L, static_argnums=3)(L_params, L_states, rng_key, pars['batch_size'])
    w_noise = full_l_batch[:, -pars['noise_dim']:]
    l_batch = full_l_batch[:, :-pars['noise_dim']]
    batched_lower_samples = jit(vmap(lower, in_axes=(0, None)), static_argnums=(1,))(
        l_batch, dim
    )
    batched_P_logits = jit(get_P_logits)(P_params, full_l_batch, rng_key)
    batched_P_samples = jit(pars['ds'].sample_hard_batched_logits)(
        batched_P_logits, tau, rng_key,
    )

    def sample_W(L, P):
        return (P @ L @ P.T).T

    Ws = jit(vmap(sample_W))(batched_lower_samples, batched_P_samples)
    eid = ensemble_intervention_distance(
        pars['ground_truth_W'],
        Ws,
        onp.exp(log_sigma_W),
        onp.exp(w_noise) * onp.ones(dim),
        sem_type,
    )
    return eid


@partial(
    pmap,
    axis_name="i",
    in_axes=(0, 0, 0, None, None, 0, None, None, None, None),
    static_broadcasted_argnums=(7, 8),
)
def parallel_elbo_estimate(P_params, L_params, L_states, Xs, interv_targets, rng_keys, tau, n, hard):
    elbos, _ = elbo(
        P_params, L_params, L_states, Xs, interv_targets, rng_keys, tau, n // pars['num_devices'], hard
    )
    mean_elbos = lax.pmean(elbos, axis_name="i")
    return jnp.mean(mean_elbos)


@partial(
    pmap,
    axis_name="i",
    in_axes=(0, 0, 0, None, None, 0, 0, 0, None),
    static_broadcasted_argnums=(3, 4),
)
def parallel_gradient_step(
    P_params, L_params, L_states, Xs, interv_targets, P_opt_state, L_opt_state, rng_key, tau,
):
    rng_key, rng_key_2 = rnd.split(rng_key, 2)
    tau_scaling_factor = 1.0 / tau

    (_, L_states), grads = value_and_grad(elbo, argnums=(0, 1), has_aux=True)(
        P_params, L_params, L_states, Xs, interv_targets, rng_key, tau, pars['num_outer'], hard=True
    )
    elbo_grad_P, elbo_grad_L = tree_map(lambda x: -tau_scaling_factor * x, grads)

    elbo_grad_P = lax.pmean(elbo_grad_P, axis_name="i")
    elbo_grad_L = lax.pmean(elbo_grad_L, axis_name="i")

    l2_elbo_grad_P = grad(
        lambda p: 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(p))
    )(P_params)
    elbo_grad_P = tree_multimap(lambda x, y: x + y, elbo_grad_P, l2_elbo_grad_P)

    P_updates, P_opt_state = pars['opt_P'].update(elbo_grad_P, P_opt_state, P_params)
    P_params = optax.apply_updates(P_params, P_updates)
    L_updates, L_opt_state = pars['opt_L'].update(elbo_grad_L, L_opt_state, L_params)
    if pars['fix_L_params']:
        pass
    else:
        L_params = optax.apply_updates(L_params, L_updates)

    return (
        P_params,
        L_params,
        L_states,
        P_opt_state,
        L_opt_state,
        rng_key_2,
    )


@jit
def compute_grad_variance(
    P_params, L_params, L_states, Xs, interv_targets, rng_key, tau,
):
    P_params, L_params, L_states, rng_key = (
        un_pmap(P_params),
        un_pmap(L_params),
        un_pmap(L_states),
        un_pmap(rng_key),
    )
    (_, L_states), grads = value_and_grad(elbo, argnums=(0, 1), has_aux=True)(
        P_params, L_params, L_states, Xs, interv_targets, rng_key, tau, pars['num_outer'], hard=True
    )

    return get_double_tree_variance(*grads)


def tau_schedule(i):
    boundaries = jnp.array([5_000, 10_000, 20_000, 60_000, 100_000])
    values = jnp.array([30.0, 10.0, 1.0, 1.0, 0.5, 0.25])
    index = jnp.sum(boundaries < i)
    return jnp.take(values, index)


def get_histogram(L_params, L_states, P_params, rng_key):
    permutations = jax.nn.one_hot(
        jnp.vstack(list(itertools.permutations([0, 1, 2]))), num_classes=3
    )
    num_samples = 100
    if pars['use_flow']:
        full_l_batch, _, _ = jit(pars['sample_flow'], static_argnums=(3,))(
            L_params, L_states, rng_key, 100
        )
        P_logits = get_P_logits(P_params, full_l_batch, rng_key)
    else:
        means, log_stds = (
            L_params[: pars['l_dim'] + pars['noise_dim']],
            L_params[pars['l_dim'] + pars['noise_dim'] :],
        )
        L_dist = pars['L_dist']
        l_distribution = L_dist(loc=means, scale=jnp.exp(log_stds))
        full_l_batch = l_distribution.sample(seed=rng_key, sample_shape=(num_samples,))
        assert type(full_l_batch) is jnp.ndarray
        P_logits = get_P_logits(P_params, full_l_batch, rng_key)
    batched_P_samples = jit(pars['ds'].sample_hard_batched_logits)(P_logits, tau, rng_key)
    histogram = onp.zeros(6)
    for P_sample in onp.array(batched_P_samples):
        for i, perm in enumerate(permutations):
            if jnp.all(P_sample == perm):
                histogram[i] += 1
    return histogram
