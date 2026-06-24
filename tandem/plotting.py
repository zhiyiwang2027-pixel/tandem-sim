from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_POLICY_COLORS = ["#d6231f", "#1f4eb4", "#0f9b9b", "#c64bd1", "#7b3fa0", "#3fa83f", "#888888"]
SWEEP_STYLE = {
    "Joint FGMW": ("^-", "#d6231f"),
    "iso1 + iso2": ("o--", "#1f4eb4"),
    "iso1 + theorem-SRP2": ("x--", "#c64bd1"),
    "SRP1 + iso2": ("s--", "#0f9b9b"),
    "Greedy": (">--", "#3fa83f"),
}


def plot_anchor_objectives(anchor_table, anchor_params, order_bar, colors=None):
    colors = DEFAULT_POLICY_COLORS if colors is None else colors
    Avals = anchor_table["Aq"].values
    Hvals = anchor_table["h"].values
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(order_bar))
    ax[0].bar(x, Avals, color=colors[:len(order_bar)])
    ax[0].axhline(anchor_params["lb_bsside"], ls="--", color="k", lw=1.3, label="BS-side LB")
    ax[0].set_title("weighted VOQ age  A^Q")
    ax[0].set_ylabel("A^Q")
    ax[0].legend(fontsize=8)
    ax[1].bar(x, Hvals, color=colors[:len(order_bar)])
    ax[1].axhline(anchor_params["lb_dest_joint"], ls="--", color="k", lw=1.3, label="joint dest LB")
    ax[1].set_title("weighted destination AoI  h")
    ax[1].set_ylabel("h")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(order_bar, rotation=35, ha="right", fontsize=8)
        a.grid(alpha=.3, axis="y")
    plt.tight_layout()
    return fig, ax


def plot_pipeline_rates(anchor_table):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    anchor_table[["VOQ arrivals", "deliveries"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("post-warmup rate per slot")
    ax.set_title("Pipeline rates at the anchor")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_stress_dest_aoi(stress_table, stress_params):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    stress_table["h"].plot(kind="bar", ax=ax)
    ax.axhline(stress_params["lb_dest_joint"], linestyle="--", linewidth=1.5, label="joint destination LB")
    ax.set_ylabel("weighted average destination AoI")
    ax.set_title("Heterogeneous dual-active stress case")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_sweep_grid(sweep_v3, style=None):
    style = SWEEP_STYLE if style is None else style
    titles = {
        "p": "channel reliability p",
        "L": "link length L",
        "w": "high-class weight",
        "mu": "edge rate mu",
        "N": "number of sources N",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axf = axes.ravel()
    for ax, nm in zip(axf, ["p", "L", "w", "mu", "N"]):
        s = sweep_v3[nm]
        x = s["x"]
        for k, (st, cl) in style.items():
            ax.plot(x, s["policies"][k], st, color=cl, lw=1.8, ms=8, label=k, mfc=("none" if "--" in st else cl))
        ax.plot(x, s["lb"], "-", color="k", lw=2.2, label="joint LB")
        ax.set_xlabel(s["xlabel"])
        ax.set_ylabel("weighted sum AoI")
        ax.set_title("vs " + titles[nm])
        ax.grid(alpha=.3)
    axf[5].axis("off")
    h_, l_ = axf[0].get_legend_handles_labels()
    axf[5].legend(h_, l_, loc="center", fontsize=12, title="policy")
    plt.tight_layout()
    return fig, axes


def plot_heterogeneous_binding_regimes(HV, ci_fun):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, key in enumerate(["A", "B"]):
        d = HV[key]
        mus = np.array(d["mu"])
        Jm = []
        Je = []
        Im = []
        Ie = []
        Dm = []
        De = []
        for mu in d["mu"]:
            jm, je = ci_fun(d["joint"][f"{mu}"])
            im, ie = ci_fun(d["iso"][f"{mu}"])
            dd = np.array(d["joint"][f"{mu}"]) - np.array(d["iso"][f"{mu}"])
            dm, de = ci_fun(dd)
            Jm += [jm]
            Je += [je]
            Im += [im]
            Ie += [ie]
            Dm += [dm]
            De += [de]
        a0, a1 = axes[row]
        a0.fill_between(mus, np.array(Jm) - Je, np.array(Jm) + Je, color="#d6231f", alpha=.18)
        a0.fill_between(mus, np.array(Im) - Ie, np.array(Im) + Ie, color="#1f4eb4", alpha=.18)
        a0.plot(mus, Jm, "^-", color="#d6231f", lw=1.8, ms=6, label="Joint FGMW")
        a0.plot(mus, Im, "o--", color="#1f4eb4", lw=1.8, ms=6, mfc="none", label="iso1+iso2")
        a1.errorbar(mus, Dm, yerr=De, fmt="s-", color="#7b3fa0", capsize=3, lw=1.6)
        a1.axhline(0, ls="--", color="k", lw=1)
        for k, mu in enumerate(d["mu"]):
            if d["regime"][k] == "both":
                a0.axvspan(mu - 0.004, mu + 0.004, color="gold", alpha=.45)
                a1.axvspan(mu - 0.004, mu + 0.004, color="gold", alpha=.45)
        a0.set_title(f"Config {key}: AoI (band = 95% CI; gold = both bind)")
        a0.set_ylabel("weighted sum AoI")
        a0.set_xlabel("edge rate mu")
        a0.legend(fontsize=8)
        a0.grid(alpha=.3)
        a1.set_title(f"Config {key}: paired difference (joint - iso, negative = joint better)")
        a1.set_ylabel("difference")
        a1.set_xlabel("edge rate mu")
        a1.grid(alpha=.3)
    plt.tight_layout()
    return fig, axes
