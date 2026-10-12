# Copyright 2026 dodowon
# SPDX-License-Identifier: MIT

"""
Callback for diagnosing why sparsity reverts to its previous value after GA is applied.

Hypotheses to verify (why does sparsity regress quickly right after GA):
  H1) The Lagrangian loss enforces only the *overall* target_sparsity and not the per-module distribution.
      → Even if GA only changes the per-module distribution, the lag loss should not directly pull it back.
         (If lag_grad acts strongly opposite to the GA change direction, H1 is wrong)
  H2) hidden_intermediate_ratio_term inside lag_loss ties hidden = intermediate.
      → If GA breaks the ratio of the two sparsities, it acts as strong regression pressure.
  H3) The CE Loss gradient of z_loga is the dominant force pushing toward the pre-GA distribution.
      → Supports the conclusion that "the distribution it reverts to is the true CE optimum".
  H4) Adam optimizer momentum (the 1st/2nd moments of z_loga) keeps the pre-GA training direction
      and thus makes it regress quickly. (GA only changes the z_loga values and leaves the optimizer state as is)
  H5) The Lagrangian multipliers (lambda_1/2/3) have accumulated to be very large, so even a small sparsity deviation
      produces a large gradient that quickly absorbs the change made by GA.
  H6) 7B→1.3B vs 7B→2.7B difference: the lower the target_sparsity (2.7B), the weaker the regression pressure
      or the weaker the dominance of the Adam moment may be. → Compare via lambda magnitude and the lag_grad/ce_grad ratio.

Recorded every N batches (log_interval) — **no access to model parameters / optimizer state**:
  - lambda_* (ParameterDict of l0_module; usually unsharded 1-element tensors under FSDP)
  - ce_loss / lag_loss / total of state.loss (only reads values that are already computed)
  ※ Accessing model params / Adam state from outside at every batch in an FSDP environment
     can cause subtle NaNs due to race / sync / cache-consistency issues.
     The z statistics and Adam moments are measured all at once inside summon at eval time.

At every eval (eval_start / eval_end) — using summon:
  - *full* z_loga statistics of each mask + expected sparsity
  - Adam moments (m1, m2) for z_loga  ← H4 analysis
  - eval_start: L2/cos up to the previous post-GA baseline
    → "how much z regressed during one eval period"
  - eval_end:   L2/cos up to the preGA of the same round
    → "how much GA changed z this time"

Every grad_decompose_interval batches (optional, default OFF):
  - Compute d(ce_loss)/dz and d(lag_loss)/dz for z_loga separately with *autograd.grad*
    → does not contaminate the .grad of model weights / lambdas
  - Two gradient norms/cosines, and the alignment sign of baseline-away/toward-preGA
    → Directly measures whether CE drives the regression (H3) or Lag does (H1/H2/H5)
"""

import math
import os
from typing import Any, Dict, Optional

import torch
from composer import Callback, Logger, State
from composer.loggers import Logger
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class SparsityRevertDiagnosticCallback(Callback):
    """
    Stores the baseline right after GA forcibly changes z_loga,
    and then measures at every batch how much z_loga regresses toward the baseline / pre-GA distribution.

    Callback registration order: if registered *after* GeneticSparsityCallback, the state *after* GA changes z_loga
    at the same eval_end can be taken as the baseline.
    """

    def __init__(
        self,
        log_interval: int = 5,
        grad_decompose_interval: int = 100,
        enable_grad_decomposition: bool = False,
        save_folder: Optional[str] = None,
    ) -> None:
        self.log_interval = max(1, int(log_interval))
        self.grad_decompose_interval = max(1, int(grad_decompose_interval))
        self.enable_grad_decomposition = bool(enable_grad_decomposition)
        self.save_folder = save_folder

        # Baseline right after GA (mask_name -> CPU tensor)
        self._baseline_z: Dict[str, torch.Tensor] = {}
        self._baseline_step: Optional[int] = None
        # Just before GA (= the previous baseline). A candidate for the "regression target".
        self._pre_ga_z: Dict[str, torch.Tensor] = {}
        self._pre_ga_step: Optional[int] = None

        print(
            f"[RevertDiag] init: log_interval={self.log_interval}, "
            f"grad_decompose_interval={self.grad_decompose_interval}, "
            f"enable_grad_decomposition={self.enable_grad_decomposition}",
            flush=True,
        )

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _l0_module(state: State):
        return getattr(state.model.model, "l0_module", None)

    @staticmethod
    def _summon(state: State, writeback: bool = False):
        return FSDP.summon_full_params(state.model, writeback=writeback, recurse=True)

    @staticmethod
    def _snapshot_z(l0_module) -> Dict[str, torch.Tensor]:
        out = {}
        for name in l0_module.masks:
            out[name] = l0_module.masks[name].z_loga.detach().clone().cpu()
        return out

    @staticmethod
    def _is_rank0() -> bool:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    @staticmethod
    def _z_stats_and_distances(
        l0_module,
        ref_z: Dict[str, torch.Tensor],
        ref_tag: str,
        optimizers=None,
    ) -> Dict[str, float]:
        """
        Called within the summon context. For each mask, computes *full* z_loga statistics + L2/cos against ref_z.
        If optimizers are given, also measures the Adam moments (H4) of z_loga.
        """
        out: Dict[str, float] = {}
        with torch.no_grad():
            for name in l0_module.masks:
                mask = l0_module.masks[name]
                z = mask.z_loga.detach().float()
                z_flat = z.reshape(-1)

                out[f"revert_diag/z/{name}/mean"] = float(z_flat.mean().cpu().item())
                out[f"revert_diag/z/{name}/std"] = float(z_flat.std().cpu().item())
                out[f"revert_diag/z/{name}/min"] = float(z_flat.min().cpu().item())
                out[f"revert_diag/z/{name}/max"] = float(z_flat.max().cpu().item())
                out[f"revert_diag/z/{name}/frac_near_keep"] = float(
                    (z_flat > 3.5).float().mean().cpu().item()
                )
                out[f"revert_diag/z/{name}/frac_near_prune"] = float(
                    (z_flat < -3.5).float().mean().cpu().item()
                )

                try:
                    score, sparsity = mask.calculate_expected_score_sparsity()
                    out[f"revert_diag/expected_sparsity/{name}"] = float(
                        sparsity.detach().mean().cpu().item()
                    )
                except Exception:
                    pass

                if name in ref_z and ref_z[name].shape == z.shape:
                    base = ref_z[name].to(z.device).float()
                    diff = (z - base).reshape(-1)
                    l2 = float(diff.norm().cpu().item())
                    base_n = max(float(base.reshape(-1).norm().cpu().item()), 1e-8)
                    out[f"revert_diag/{ref_tag}/{name}/l2"] = l2
                    out[f"revert_diag/{ref_tag}/{name}/rel"] = l2 / base_n
                    out[f"revert_diag/{ref_tag}/{name}/cos"] = float(
                        torch.nn.functional.cosine_similarity(
                            z_flat, base.reshape(-1), dim=0
                        ).cpu().item()
                    )

                # Adam moment (H4) — accessed safely only inside summon
                if optimizers is not None:
                    p = mask.z_loga
                    for opt in optimizers:
                        opt_state = opt.state.get(p, None)
                        if not opt_state:
                            continue
                        if "exp_avg" in opt_state:
                            ea = opt_state["exp_avg"].detach()
                            out[f"revert_diag/adam/{name}/m1_abs_mean"] = float(
                                ea.abs().mean().cpu().item()
                            )
                            out[f"revert_diag/adam/{name}/m1_norm"] = float(
                                ea.norm().cpu().item()
                            )
                        if "exp_avg_sq" in opt_state:
                            out[f"revert_diag/adam/{name}/m2_mean"] = float(
                                opt_state["exp_avg_sq"].detach().mean().cpu().item()
                            )
                        break  # measure only the first optimizer
        return out

    # ──────────────────────────────────────────────────────────
    # Composer hooks
    # ──────────────────────────────────────────────────────────
    # NOTE: fit_start summon removed. A summon right before training starts may conflict with FSDP init,
    #       so the initial baseline is set naturally at the first eval_start.

    def eval_start(self, state: State, logger: Logger) -> None:
        """
        GA changes z_loga at eval_end. At the eval_start point it is just before GA.
        How far z at this point has moved from the previous post-GA baseline = "amount of regression".
        And this z becomes the "just before GA (= preGA)" of the next round.
        """
        l0 = self._l0_module(state)
        if l0 is None:
            return
        step = int(state.timestamp.batch.value)
        log: Dict[str, Any] = {}
        try:
            with self._summon(state):
                # Distance/cos to the previous post-GA baseline = degree of regression during one eval period
                log.update(self._z_stats_and_distances(
                    l0, self._baseline_z, ref_tag="dist_to_baseline_at_eval",
                    optimizers=state.optimizers,
                ))
                # Update snapshot
                self._pre_ga_z = self._snapshot_z(l0)
                self._pre_ga_step = step
        except Exception as e:
            print(f"[RevertDiag] eval_start snapshot failed: {e}", flush=True)

        if self._is_rank0():
            log["revert_diag/pre_ga_step"] = step
            logger.log_metrics(log)
            print(
                f"[RevertDiag] preGA snapshot at eval_start, step={step}",
                flush=True,
            )

    def eval_end(self, state: State, logger: Logger) -> None:
        """
        Takes the state *after* GA forcibly changes z_loga as the new baseline.
        Compare with the preGA (_pre_ga_z) of the same round → how much GA actually changed.
        """
        l0 = self._l0_module(state)
        if l0 is None:
            return
        step = int(state.timestamp.batch.value)
        log: Dict[str, Any] = {}

        try:
            with self._summon(state):
                # Compare with the preGA of the same round → how much GA changed this time
                log.update(self._z_stats_and_distances(
                    l0, self._pre_ga_z, ref_tag="post_ga_vs_pre_ga",
                    optimizers=state.optimizers,
                ))
                new_baseline = self._snapshot_z(l0)
        except Exception as e:
            print(f"[RevertDiag] eval_end snapshot failed: {e}", flush=True)
            new_baseline = self._baseline_z

        # Two kinds of overall |Δz|
        delta_sq = 0.0
        for k, v in new_baseline.items():
            if k in self._pre_ga_z and self._pre_ga_z[k].shape == v.shape:
                delta_sq += (v - self._pre_ga_z[k]).pow(2).sum().item()
        delta_from_pre_ga = math.sqrt(delta_sq)

        delta_sq_prev = 0.0
        for k, v in new_baseline.items():
            if k in self._baseline_z and self._baseline_z[k].shape == v.shape:
                delta_sq_prev += (v - self._baseline_z[k]).pow(2).sum().item()
        delta_from_prev_baseline = math.sqrt(delta_sq_prev)

        self._baseline_z = new_baseline
        self._baseline_step = step

        if self._is_rank0():
            log.update({
                "revert_diag/baseline_step": step,
                "revert_diag/ga_step_delta": delta_from_pre_ga,
                "revert_diag/baseline_delta_from_prev": delta_from_prev_baseline,
            })
            logger.log_metrics(log)
            print(
                f"[RevertDiag] new baseline at step={step}, "
                f"|Δz| from preGA(this round)={delta_from_pre_ga:.4f}, "
                f"|Δz| from prev baseline={delta_from_prev_baseline:.4f}",
                flush=True,
            )

    def after_train_batch(self, state: State, logger: Logger) -> None:
        """
        Called every batch. **Never accesses model parameters / optimizer state.**
        In an FSDP environment, outside access can cause subtle NaNs due to race / cache-consistency / sync issues.

        What is recorded here is only safe scalars outside the model:
          - lambdas (ParameterDict of l0_module; 1-element tensors that FSDP usually leaves unsharded)
          - ce/lag/total of state.loss  (values that are detachable since backward has finished)

        ※ z_loga statistics / Adam moments / expected_sparsity / baseline distance are
          all handled at once in eval_start / eval_end (summon context).
        """
        l0 = self._l0_module(state)
        if l0 is None:
            return
        step = int(state.timestamp.batch.value)
        if step % self.log_interval != 0:
            return

        log: Dict[str, Any] = {}

        with torch.no_grad():
            # 1) lambdas (H5) - small 1-element parameters (usually unsharded under FSDP)
            for lk, lam in l0.lambdas.items():
                try:
                    log[f"revert_diag/lambda/{lk}"] = float(
                        lam.detach().mean().cpu().item()
                    )
                except Exception:
                    pass

            # 2) loss values (read the dict that composer has already computed)
            loss = state.loss
            if isinstance(loss, dict):
                for lk in ("ce_loss", "lag_loss", "total"):
                    v = loss.get(lk, None)
                    if v is None:
                        continue
                    try:
                        val = v.detach().cpu().item() if torch.is_tensor(v) else float(v)
                        log[f"revert_diag/loss/{lk}"] = float(val)
                    except Exception:
                        pass

        # 3) gradient decomposition (optional, default OFF). It uses autograd.grad so .grad is not contaminated, but it is expensive.
        if self.enable_grad_decomposition and step % self.grad_decompose_interval == 0:
            try:
                grad_log = self._compute_grad_decomposition(state)
                log.update(grad_log)
            except Exception as e:
                print(
                    f"[RevertDiag] grad decomposition failed at step {step}: {e}",
                    flush=True,
                )

        if log and self._is_rank0():
            logger.log_metrics(log)

    # ──────────────────────────────────────────────────────────
    # Gradient decomposition
    # ──────────────────────────────────────────────────────────
    def _compute_grad_decomposition(self, state: State) -> Dict[str, float]:
        """
        Computes gradients only for z_loga to separate the contributions of CE / Lag.

        Implementation points:
          - Use torch.autograd.grad(loss, [z_loga ...]) instead of .backward().
            → does not contaminate the .grad of model weights / lambdas / any other parameters.
          - Build the forward graph only once with retain_graph=True and use it twice.
          - Comparison with baseline / pre_ga in the analysis stage: needs the full param, so summon is used.

        Sign interpretation:
          - update = -lr * grad. proj(grad, away_from_baseline) > 0  ⇒  drives regression toward the baseline.
          - proj(grad, toward_preGA) < 0  ⇒  the update pulls toward preGA, driving regression.
        """
        l0 = self._l0_module(state)
        model = state.model
        batch = state.batch
        if batch is None or l0 is None:
            return {}

        if "pruned_steps" not in batch:
            try:
                input_ids = batch["input_ids"]
                batch["pruned_steps"] = torch.LongTensor(
                    [int(state.timestamp.batch.value)] * len(input_ids)
                ).to(input_ids.device)
            except Exception:
                pass

        z_params = [(name, l0.masks[name].z_loga) for name in l0.masks]
        z_list = [p for _, p in z_params]
        names = [n for n, _ in z_params]

        was_training = model.training
        # Must be in train mode so dropout/sample_z etc. are the same as during training. (for an accurate post-hoc comparison.)
        model.train()

        out: Dict[str, float] = {}

        try:
            outputs = model(batch)
            l0_output = outputs.get("l0_output", None)

            lag_grads: Dict[str, Optional[torch.Tensor]] = {n: None for n in names}
            ce_grads: Dict[str, Optional[torch.Tensor]] = {n: None for n in names}

            # ── (1) Lag-only gradient w.r.t. z_loga ──
            if (
                l0_output is not None
                and torch.is_tensor(l0_output[0])
                and l0_output[0].requires_grad
            ):
                grads = torch.autograd.grad(
                    l0_output[0], z_list,
                    retain_graph=True, allow_unused=True, create_graph=False,
                )
                for n, g in zip(names, grads):
                    lag_grads[n] = g.detach() if g is not None else None

            # ── (2) CE-only gradient w.r.t. z_loga ──
            ret = state.model.loss(outputs, batch)
            if (
                isinstance(ret, dict)
                and "ce_loss" in ret
                and torch.is_tensor(ret["ce_loss"])
                and ret["ce_loss"].requires_grad
            ):
                grads = torch.autograd.grad(
                    ret["ce_loss"], z_list,
                    retain_graph=False, allow_unused=True, create_graph=False,
                )
                for n, g in zip(names, grads):
                    ce_grads[n] = g.detach() if g is not None else None

            # ── (3) Analysis (needs full param) ──
            with self._summon(state):
                with torch.no_grad():
                    for name, p in z_params:
                        cg = ce_grads.get(name)
                        lg = lag_grads.get(name)
                        cg_n = float(cg.norm().cpu().item()) if cg is not None else 0.0
                        lg_n = float(lg.norm().cpu().item()) if lg is not None else 0.0
                        out[f"revert_diag/grad/{name}/ce_norm"] = cg_n
                        out[f"revert_diag/grad/{name}/lag_norm"] = lg_n
                        out[f"revert_diag/grad/{name}/lag_over_ce"] = (
                            (lg_n / cg_n) if cg_n > 1e-12 else float("inf")
                        )
                        if cg is not None and lg is not None and cg_n > 1e-12 and lg_n > 1e-12:
                            cos = float(
                                (cg.reshape(-1) * lg.reshape(-1)).sum().cpu().item()
                                / (cg_n * lg_n)
                            )
                            out[f"revert_diag/grad/{name}/cos_ce_lag"] = cos

                        if (
                            name in self._baseline_z
                            and self._baseline_z[name].shape == p.shape
                        ):
                            base = self._baseline_z[name].to(p.device).float()
                            away = (p.detach().float() - base).reshape(-1)
                            away_n = float(away.norm().cpu().item())
                            if away_n > 1e-8:
                                away_unit = away / away_n
                                if cg is not None:
                                    proj_ce = float(
                                        (cg.reshape(-1).float() * away_unit).sum().cpu().item()
                                    )
                                    out[f"revert_diag/grad/{name}/ce_drives_revert"] = proj_ce
                                if lg is not None:
                                    proj_lag = float(
                                        (lg.reshape(-1).float() * away_unit).sum().cpu().item()
                                    )
                                    out[f"revert_diag/grad/{name}/lag_drives_revert"] = proj_lag

                        if (
                            name in self._pre_ga_z
                            and self._pre_ga_z[name].shape == p.shape
                        ):
                            pre = self._pre_ga_z[name].to(p.device).float().reshape(-1)
                            toward = (pre - p.detach().float().reshape(-1))
                            toward_n = float(toward.norm().cpu().item())
                            if toward_n > 1e-8:
                                toward_unit = toward / toward_n
                                if cg is not None:
                                    proj_ce = float(
                                        (cg.reshape(-1).float() * toward_unit).sum().cpu().item()
                                    )
                                    out[f"revert_diag/grad/{name}/ce_grad_dot_toward_preGA"] = proj_ce
                                if lg is not None:
                                    proj_lag = float(
                                        (lg.reshape(-1).float() * toward_unit).sum().cpu().item()
                                    )
                                    out[f"revert_diag/grad/{name}/lag_grad_dot_toward_preGA"] = proj_lag
        finally:
            if not was_training:
                model.eval()

        return out

    # ──────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────
    def state_dict(self) -> Dict[str, Any]:
        return {
            "baseline_z": self._baseline_z,
            "baseline_step": self._baseline_step,
            "pre_ga_z": self._pre_ga_z,
            "pre_ga_step": self._pre_ga_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._baseline_z = state_dict.get("baseline_z", {})
        self._baseline_step = state_dict.get("baseline_step", None)
        self._pre_ga_z = state_dict.get("pre_ga_z", {})
        self._pre_ga_step = state_dict.get("pre_ga_step", None)
