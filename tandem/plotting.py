from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_POLICY_COLORS = ["#d6231f", "#1f4eb4", "#0f9b9b", "#c64bd1", "#7b3fa0", "#3fa83f", "#888888"]
SWEEP_STYLE = {
    "Joint FGMW": ("^-", "#d6231f"),
    "iso1 + iso2": ("o--", "#1f4eb4"),
    "iso1 + iso2-lambda": ("D--", "#6f4dbf"),
    "iso1 + theorem-SRP2": ("x--", "#c64bd1"),
    "SRP1 + iso2": ("s--", "#0f9b9b"),
    "SRP-iso": ("s--", "#0f9b9b"),
    "SRP-tandem-LB": ("x--", "#c64bd1"),
    "Downstream-Aware MW": ("P-", "#e07a1f"),
    "Greedy": (">--", "#3fa83f"),
    "Uniform": ("v--", "#888888"),
}
QUICK_CONFIG_ORDER = ["det_aligned", "det_conflict", "aligned", "neutral", "conflict"]
QUICK_POLICY_ORDER = [
    "Joint FGMW",
    "iso1 + iso2-lambda",
    "Downstream-Aware MW",
    "Greedy",
    "SRP-iso",
    "SRP-tandem-LB",
]
QUICK_POLICY_COLORS = {
    "Joint FGMW": "#d6231f",
    "iso1 + iso2": "#1f4eb4",
    "iso1 + iso2-lambda": "#6f4dbf",
    "Downstream-Aware MW": "#e07a1f",
    "SRP-iso": "#0f9b9b",
    "SRP-tandem-LB": "#c64bd1",
    "Greedy": "#3fa83f",
    "Uniform": "#888888",
}
SRP_POLICY_ORDER = ["SRP-iso", "SRP-tandem-LB"]
SRP_POLICY_COLORS = {
    "SRP-iso": "#0f9b9b",
    "SRP-tandem-LB": "#c64bd1",
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
        "w": "raw weight spread",
        "mu": "edge rate mu",
        "N": "number of sources N",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axf = axes.ravel()
    for ax, nm in zip(axf, ["p", "L", "w", "mu", "N"]):
        s = sweep_v3[nm]
        x = s["x"]
        for k, (st, cl) in style.items():
            if k not in s["policies"]:
                continue
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


def _quick_orders(df):
    configs = [c for c in QUICK_CONFIG_ORDER if c in set(df["config"])]
    policies = [p for p in QUICK_POLICY_ORDER if p in set(df["policy"])]
    return configs, policies


def _quick_metric(df, metric, configs, policies):
    return (
        df.pivot(index="config", columns="policy", values=metric)
        .reindex(index=configs, columns=policies)
    )


def plot_quick_heterogeneous_aoi(df):
    configs, policies = _quick_orders(df)
    values = _quick_metric(df, "weighted_dest_aoi", configs, policies)
    errors = _quick_metric(df, "weighted_dest_aoi_se", configs, policies)
    x = np.arange(len(configs))
    width = 0.8 / max(1, len(policies))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for j, policy in enumerate(policies):
        offset = (j - (len(policies) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values[policy].to_numpy(),
            width,
            yerr=errors[policy].to_numpy(),
            capsize=3,
            color=QUICK_POLICY_COLORS.get(policy),
            label=policy,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("mean weighted destination AoI")
    ax.set_title("Quick deterministic raw-weight heterogeneous comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, ax


def _plot_quick_heterogeneous_gap(df, metric, ylabel, title):
    configs, policies = _quick_orders(df)
    values = _quick_metric(df, metric, configs, policies)
    x = np.arange(len(configs))
    width = 0.8 / max(1, len(policies))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for j, policy in enumerate(policies):
        offset = (j - (len(policies) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values[policy].to_numpy(),
            width,
            color=QUICK_POLICY_COLORS.get(policy),
            label=policy,
        )
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, ax


def plot_quick_heterogeneous_gap_vs_iso(df):
    return _plot_quick_heterogeneous_gap(
        df,
        "gap_vs_iso_pct",
        "gap vs iso1+iso2 (%)",
        "Deterministic raw-weight gap relative to relaxed isolated MW",
    )


def plot_quick_heterogeneous_gap_vs_iso_lambda(df):
    return _plot_quick_heterogeneous_gap(
        df,
        "gap_vs_iso_lambda_pct",
        "gap vs iso1+iso2-lambda (%)",
        "Deterministic raw-weight gap relative to lambda-aware isolated MW",
    )


def plot_quick_heterogeneous_gap(df):
    return plot_quick_heterogeneous_gap_vs_iso(df)


def plot_quick_heterogeneous_pipeline(df):
    configs, policies = _quick_orders(df)
    metrics = [
        ("total_VOQ_arrival_rate", "VOQ arrivals"),
        ("total_delivery_rate", "deliveries"),
        ("stage2_idle_empty_frac", "S2 idle-empty"),
        ("total_overwrite_rate", "overwrites"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = _quick_metric(df, metric, configs, policies)
        values.plot(
            kind="bar",
            ax=ax,
            color=[QUICK_POLICY_COLORS.get(policy) for policy in policies],
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("post-warmup rate/fraction")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Deterministic raw-weight pipeline diagnostics", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


def _screening_alignment_order(df):
    return [a for a in QUICK_CONFIG_ORDER if a in set(df["alignment"])]


def _screening_regime_order(df):
    preferred = ["edge-only", "both", "link-only", "slack"]
    return [r for r in preferred if r in set(df["kkt_regime"])]


def _ordered_present(values, preferred):
    present = [value for value in preferred if value in set(values)]
    extras = sorted(value for value in set(values) if value not in set(preferred))
    return present + extras


def _srp_data(df):
    if "policy" not in df:
        raise ValueError("Expected a policy column.")
    return df[df["policy"].isin(SRP_POLICY_ORDER)].copy()


def _with_column(data, column, default):
    if column not in data:
        data[column] = default
    return data


def _plot_empty(ax, message):
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])


def _srp_grouped_bar(df, group_col, metric, ylabel, title, preferred_groups):
    data = _srp_data(df)
    data = _with_column(data, group_col, "all")
    data = _with_column(data, metric, np.nan)
    data[group_col] = data[group_col].fillna("all")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    if data.empty:
        _plot_empty(ax, "No SRP rows available")
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax

    groups = _ordered_present(data[group_col], preferred_groups)
    policies = [policy for policy in SRP_POLICY_ORDER if policy in set(data["policy"])]
    values = (
        data.pivot_table(index=group_col, columns="policy", values=metric, aggfunc="mean")
        .reindex(index=groups, columns=policies)
    )
    x = np.arange(len(groups))
    width = 0.75 / max(1, len(policies))
    for j, policy in enumerate(policies):
        offset = (j - (len(policies) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values[policy].to_numpy(),
            width,
            color=SRP_POLICY_COLORS.get(policy),
            label=policy,
        )
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, ax


def plot_srp_iso_vs_tandem_gap_by_alignment(df):
    return _srp_grouped_bar(
        df,
        "alignment",
        "gap_vs_iso_lambda_pct",
        "gap vs iso1+iso2-lambda (%)",
        "SRP gap vs iso1+iso2-lambda by alignment",
        QUICK_CONFIG_ORDER,
    )


def plot_srp_iso_vs_tandem_gap_by_regime(df):
    preferred = ["edge-only", "both", "link-only", "slack", "all"]
    return _srp_grouped_bar(
        df,
        "kkt_regime",
        "gap_vs_iso_lambda_pct",
        "gap vs iso1+iso2-lambda (%)",
        "SRP gap vs iso1+iso2-lambda by KKT regime",
        preferred,
    )


def plot_srp_tandem_minus_iso_gap(df):
    data = _srp_data(df)
    data = _with_column(data, "alignment", "all")
    data = _with_column(data, "gap_vs_iso_lambda_pct", np.nan)
    data["alignment"] = data["alignment"].fillna("all")

    fig, ax = plt.subplots(figsize=(8, 4.4))
    if data.empty:
        _plot_empty(ax, "No SRP rows available")
        ax.set_title("SRP-tandem-LB minus SRP-iso gap")
        plt.tight_layout()
        return fig, ax

    alignments = _ordered_present(data["alignment"], QUICK_CONFIG_ORDER + ["all"])
    values = (
        data.pivot_table(
            index="alignment",
            columns="policy",
            values="gap_vs_iso_lambda_pct",
            aggfunc="mean",
        )
        .reindex(index=alignments, columns=SRP_POLICY_ORDER)
    )
    diff = values["SRP-tandem-LB"] - values["SRP-iso"]
    ax.bar(alignments, diff.to_numpy(), color="#7b3fa0")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_ylabel("mean gap difference (percentage points)")
    ax.set_title("SRP-tandem-LB minus SRP-iso gap by alignment")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_srp_pipeline_diagnostics(df):
    data = _srp_data(df)
    metrics = [
        ("total_VOQ_arrival_rate", "VOQ arrivals"),
        ("total_delivery_rate", "deliveries"),
        ("total_overwrite_rate", "overwrites"),
        ("stage2_idle_empty_frac", "S2 idle-empty"),
    ]
    for metric, _ in metrics:
        data = _with_column(data, metric, np.nan)

    policies = [policy for policy in SRP_POLICY_ORDER if policy in set(data["policy"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    if data.empty or not policies:
        for ax in axes.ravel():
            _plot_empty(ax, "No SRP rows available")
        plt.tight_layout()
        return fig, axes

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = data.groupby("policy")[metric].mean().reindex(policies)
        ax.bar(
            policies,
            values.to_numpy(),
            color=[SRP_POLICY_COLORS.get(policy) for policy in policies],
        )
        ax.set_title(title)
        ax.set_ylabel("post-warmup rate/fraction")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=20)
    plt.tight_layout()
    return fig, axes


def plot_screening_joint_gap_by_regime(df):
    data = df[df["policy"] == "Joint FGMW"].copy()
    regimes = _screening_regime_order(data)
    grouped = data.groupby("kkt_regime")["gap_vs_iso_lambda_pct"]
    means = grouped.mean().reindex(regimes)
    counts = grouped.count().reindex(regimes)
    errors = (grouped.std().reindex(regimes) / np.sqrt(counts)).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(regimes, means.to_numpy(), yerr=errors.to_numpy(), capsize=3, color="#d6231f")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_ylabel("Joint gap vs iso1+iso2-lambda (%)")
    ax.set_title("Joint FGMW gap by KKT regime")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_screening_greedy_gap_by_alignment(df):
    data = df[df["policy"] == "Greedy"].copy()
    alignments = _screening_alignment_order(data)
    ratios = sorted(data["weight_ratio"].unique())
    pivot = (
        data.pivot_table(
            index="alignment",
            columns="weight_ratio",
            values="gap_vs_iso_lambda_pct",
            aggfunc="mean",
        )
        .reindex(index=alignments, columns=ratios)
    )

    fig, ax = plt.subplots(figsize=(9, 4.6))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Greedy gap vs iso1+iso2-lambda (%)")
    ax.set_title("Greedy gap by alignment and weight spread")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="weight ratio", fontsize=8)
    plt.tight_layout()
    return fig, ax


def _plot_screening_heatmap(df, policy, title):
    data = df[df["policy"] == policy].copy()
    alignments = _screening_alignment_order(data)
    mus = sorted(data["mu"].unique())
    pivot = (
        data.pivot_table(
            index="alignment",
            columns="mu",
            values="gap_vs_iso_lambda_pct",
            aggfunc="mean",
        )
        .reindex(index=alignments, columns=mus)
    )
    values = pivot.to_numpy(float)
    finite = values[np.isfinite(values)]
    vmax = max(1.0, float(np.max(np.abs(finite)))) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(mus)))
    ax.set_xticklabels([f"{mu:g}" for mu in mus])
    ax.set_yticks(np.arange(len(alignments)))
    ax.set_yticklabels(alignments)
    ax.set_xlabel("mu")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("gap vs iso1+iso2-lambda (%)")
    for i in range(len(alignments)):
        for j in range(len(mus)):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    return fig, ax


def plot_screening_joint_heatmap(df):
    return _plot_screening_heatmap(
        df,
        "Joint FGMW",
        "Joint FGMW gap by alignment and edge rate",
    )


def plot_screening_greedy_heatmap(df):
    return _plot_screening_heatmap(
        df,
        "Greedy",
        "Greedy gap by alignment and edge rate",
    )


def plot_deterministic_heterogeneous_gap(df):
    networks = [name for name in ["det_aligned", "det_conflict"] if name in set(df["network"])]
    L_values = sorted(df["L"].unique())
    policy_order = [
        "Joint FGMW",
        "iso1 + iso2-lambda",
        "Downstream-Aware MW",
        "Greedy",
        "SRP-iso",
        "SRP-tandem-LB",
    ]
    policies = [policy for policy in policy_order if policy in set(df["policy"])]
    fig, axes = plt.subplots(
        len(networks),
        len(L_values),
        figsize=(5.2 * max(1, len(L_values)), 3.8 * max(1, len(networks))),
        squeeze=False,
        sharex=True,
    )
    for r, network in enumerate(networks):
        for c, L in enumerate(L_values):
            ax = axes[r, c]
            data = df[(df["network"] == network) & (df["L"] == L)]
            for policy in policies:
                pdata = data[data["policy"] == policy].sort_values("mu")
                if pdata.empty:
                    continue
                style, color = SWEEP_STYLE.get(policy, ("o-", None))
                ax.plot(
                    pdata["mu"],
                    pdata["gap_vs_iso_lambda_pct"],
                    style,
                    color=color,
                    lw=1.8,
                    ms=6,
                    label=policy,
                    mfc=("none" if "--" in style else color),
                )
            ax.axhline(0.0, color="k", lw=1.0)
            ax.set_title(f"{network}, L={L}")
            ax.set_xlabel("mu")
            ax.set_ylabel("gap vs iso1+iso2-lambda (%)")
            ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Aggressive deterministic raw-weight heterogeneous comparison",
        y=0.995,
    )
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), fontsize=8)
        fig.subplots_adjust(top=0.82)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    return fig, axes
