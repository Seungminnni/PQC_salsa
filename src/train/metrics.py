import torch


def compute_prediction_accuracies(pred_mask, word_scores, y):
    """
    Compute the legacy ACC1 / ACC2 metrics used by the original logs.

    ACC1: first predicted token is correct.
    ACC2: at least two predicted tokens are correct.

    The second metric is intentionally less strict than exact-match, which
    makes it much more informative for short base-encoded targets such as the
    q=17 toy runs where exact sequence accuracy often stays at 0 for a long
    time even though the model is improving.
    """
    with torch.no_grad():
        if y.numel() == 0:
            return 0.0, 0.0

        pred_mask = pred_mask.to(y.device)
        correct = word_scores.max(1)[1].eq(y)
        token_grid = torch.zeros_like(pred_mask, dtype=torch.bool, device=y.device)
        token_grid[pred_mask] = correct

        acc1 = token_grid[0].float().mean().item() * 100
        target_lengths = pred_mask.sum(0)
        threshold = torch.minimum(
            target_lengths,
            torch.full_like(target_lengths, 2),
        )
        acc2 = token_grid.sum(0).ge(threshold).float().mean().item() * 100

    return acc1, acc2
