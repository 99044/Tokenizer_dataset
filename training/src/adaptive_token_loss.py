"""Soft token-level excess-loss weighting with an online reference."""

import math

import torch
import torch.nn.functional as F


class AdaptiveTokenLoss:
    """Use recent loss for the same target token ID as a cheap reference."""

    def __init__(
        self, vocab_size, device, *, ema_decay, prior_count, warmup_steps,
        ramp_steps, max_weight_delta,
    ):
        if not 0 <= ema_decay < 1 or not math.isfinite(prior_count) or prior_count <= 0:
            raise ValueError("adaptive_token_loss requires 0 <= ema_decay < 1 and prior_count > 0")
        if type(warmup_steps) is not int or warmup_steps < 1:
            raise ValueError("adaptive_token_loss.warmup_steps must be a positive integer")
        if type(ramp_steps) is not int or ramp_steps < 1:
            raise ValueError("adaptive_token_loss.ramp_steps must be a positive integer")
        if not 0 < max_weight_delta < 1:
            raise ValueError("adaptive_token_loss.max_weight_delta must be in (0, 1)")
        self.vocab_size = vocab_size
        self.ema_decay = ema_decay
        self.prior_count = prior_count
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps
        self.max_weight_delta = max_weight_delta
        # Per-ID mean counts and total NLL per optimizer update.
        self.history = torch.zeros(2, vocab_size, device=device, dtype=torch.float32)
        self.moments = torch.zeros(2, device=device, dtype=torch.float32)
        self.initialized = False

    def strength(self, step):
        progress = min(max(step - self.warmup_steps, 0) / self.ramp_steps, 1.0)
        return self.max_weight_delta * progress

    def new_step_stats(self):
        # Pack per-ID counts and losses with scalar summaries for one all-reduce.
        return self.history.new_zeros(2 * self.vocab_size + 5)

    @torch.no_grad()
    def weights(self, losses, labels, step):
        strength = self.strength(step)
        if strength == 0:
            return torch.ones_like(losses)
        count, loss_sum = self.history[:, labels].unbind()
        mean, second_moment = self.moments.unbind()
        denominator = count + self.prior_count
        reference = (loss_sum + self.prior_count * mean) / denominator
        scale = (second_moment - mean.square()).clamp_min(1e-4).sqrt()
        excess = (losses - reference) / scale
        confidence = count / denominator
        return 1 + strength * confidence * excess.tanh()

    def objective(self, logits, labels, step, stats):
        labels = labels.reshape(-1)
        losses = F.cross_entropy(
            logits.float().reshape(-1, self.vocab_size), labels, reduction="none"
        )
        with torch.no_grad():
            nll = losses.detach()
            valid = labels.ne(-100)
            token_ids = labels.masked_fill(~valid, 0)
            weights = self.weights(nll, token_ids, step).masked_fill(~valid, 0)
            counts, loss_sums = stats[:-5].view_as(self.history).unbind()
            counts.scatter_add_(0, token_ids, valid.to(nll.dtype))
            loss_sums.scatter_add_(0, token_ids, nll)
            weight_sum = weights.sum()
            stats[-5:].add_(torch.stack((
                nll.sum(), nll.square().sum(), valid.sum(dtype=nll.dtype),
                weight_sum, weights.square().sum(),
            )))
        # Normalize CE before adding the router auxiliary loss. DDP and gradient
        # accumulation still average the rank-local microbatch objectives.
        return (weights * losses).sum() / weight_sum.clamp_min(torch.finfo(weights.dtype).tiny)

    @torch.no_grad()
    def update(self, stats, step):
        nll_sum, _, count, weight_sum, weight_square_sum = stats[-5:].tolist()
        if count:
            decay = self.ema_decay if self.initialized else 0.0
            self.history.lerp_(stats[:-5].view_as(self.history), 1 - decay)
            self.moments.lerp_(stats[-5:-3] / count, 1 - decay)
            self.initialized = True
        count = max(count, 1.0)
        weight_mean = weight_sum / count
        return {
            "train/token_nll": nll_sum / count,
            "train/token_weight_mean": weight_mean,
            "train/token_weight_std": math.sqrt(max(weight_square_sum / count - weight_mean**2, 0)),
            "train/token_weight_strength": self.strength(step),
        }

    def state_dict(self):
        return {"history": self.history, "moments": self.moments, "initialized": self.initialized}

    def load_state_dict(self, state):
        self.history.copy_(state["history"])
        self.moments.copy_(state["moments"])
        self.initialized = state["initialized"]
