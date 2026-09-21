"""Training and held-out diagnostics share exactly the same loss definitions."""
import random

import numpy as np
import torch
import torch.nn.functional as F

from .augmentation import transform_sample

LOSS_KEYS = ('total_loss', 'policy_loss', 'value_loss', 'target_entropy', 'policy_kl')


def loss_terms(logits, predictions, policies, values):
    log_probs = F.log_softmax(logits, dim=1)
    policy = -(policies * log_probs).sum(dim=1).mean()
    entropy = -(policies * policies.clamp_min(1e-30).log()).sum(dim=1).mean()
    value = F.mse_loss(predictions.flatten(), values.flatten())
    return dict(total_loss=policy + value, policy_loss=policy, value_loss=value,
                target_entropy=entropy, policy_kl=policy - entropy)


def tensors(samples, device, augment=False, rng=None):
    rng = rng or random
    states, policies, values = zip(*samples)
    if augment:
        states, policies = zip(*(transform_sample(s, p, rng.randrange(8)) for s, p in zip(states, policies)))
    return (torch.stack(list(states)).float().to(device),
            torch.as_tensor(np.stack(policies), dtype=torch.float32, device=device),
            torch.tensor(values, dtype=torch.float32, device=device))


def update_model(model, optimizer, samples, device, augment=True, rng=None):
    states, policies, values = tensors(samples, device, augment, rng)
    model.train()
    logits, predictions = model(states)
    terms = loss_terms(logits, predictions, policies, values)
    optimizer.zero_grad()
    terms['total_loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {key: float(value.detach().item()) for key, value in terms.items()}


def validate_model(model, samples, device, batch_size=256):
    """All held-out positions, no augmentation or BatchNorm running-stat updates."""
    if not samples:
        return dict(validation_samples=0, **{f'val_{key}': None for key in LOSS_KEYS})
    was_training = model.training
    model.eval()
    totals = dict.fromkeys(LOSS_KEYS, 0.0)
    try:
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                states, policies, values = tensors(batch, device)
                terms = loss_terms(*model(states), policies, values)
                for key, value in terms.items():
                    totals[key] += float(value.item()) * len(batch)
    finally:
        model.train(was_training)
    return dict(validation_samples=len(samples), **{f'val_{key}': value / len(samples) for key, value in totals.items()})
