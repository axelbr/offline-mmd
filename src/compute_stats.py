import argparse
import json
import json

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as stats


def compute_metric_stats(metrics: dict) -> dict:

    def compute_seed_stats(x):
        mean = jnp.mean(x, 0)
        low, high = stats.t.interval(0.95, x.shape[0] - 1, loc=mean, scale=stats.sem(x, 0))
        return jax.tree.map(lambda x: np.array(x).tolist(), {
            "mean": mean,
            "std": jnp.std(x, 0),
            "ci_95_low": low,
            "ci_95_high": high
        })

    # Compute intervals over seeds
    data = jax.tree.map(compute_seed_stats, metrics)
    return data

def main(args) -> None:
    with open(f"{args.logdir}/results.json", "r") as f:
        metrics = json.load(f)
    scalar_metrics = {k: v for k, v in metrics.items() if k.startswith("metric")}
    scalar_metrics = {k: jnp.array(v) for k, v in scalar_metrics.items()}
    stats = compute_metric_stats(scalar_metrics)
    with open(f"{args.logdir}/stats.json", "w") as f:
        json.dump(stats, f)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--logdir", type=str, required=True)
    args = argparser.parse_args()
    main(args)
