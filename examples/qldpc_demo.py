from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import qldpc

def make_figure(
    codes: Sequence[qldpc.codes.ClassicalCode | qldpc.codes.CSSCode],
    num_samples: int = 10**3,
    physical_rates: Sequence[float] = list(np.logspace(-4, -0.1, 100)),
    figsize: tuple[int, int] = (8, 6),
    **decoding_args: object,
) -> tuple[plt.Figure, plt.Axes]:
    """Make a figure comparing physical vs. logical error rates in a code family."""
    figure, axis = plt.subplots(figsize=figsize)

    for code in codes:
        # distance = code.get_distance()
        get_logical_error_rate = code.get_logical_error_rate_func(
            num_samples, max(physical_rates), **decoding_args
        )
        logical_rates, stderrs = get_logical_error_rate(physical_rates)
        bound = code.get_distance_bound(num_trials=10**3)
        line, *_ = axis.plot(physical_rates, logical_rates, label=f"[{code.num_qubits}, {code.dimension}, {bound}]")
        # line, *_ = axis.plot(physical_rates, logical_rates, label=f"hi")
        axis.fill_between(
            physical_rates,
            logical_rates - stderrs,
            logical_rates + stderrs,
            color=line.get_color(),
            alpha=0.2,
        )

    axis.axline(
        [0, 0],
        slope=1,
        color="k",
        linestyle=":",
        label=r"$p_{\mathrm{log}}=p_{\mathrm{phys}}$",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(right=1)
    axis.set_ylim(bottom=max(min(physical_rates) ** 2, axis.get_ylim()[0]), top=1)
    axis.set_xlabel(r"physical error rate $p_{\mathrm{phys}}$")
    axis.set_ylabel(r"logical error rate $p_{\mathrm{log}}$")
    axis.set_title(f"Logical vs Physical Error Rate")
    axis.legend(loc="best")
    figure.tight_layout();

    return figure, axis