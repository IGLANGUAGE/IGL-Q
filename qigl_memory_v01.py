"""
qigl_memory_v01.py

Q-IGL Persistent Memory v0.1
============================

Purpose:
Test the architecture

    M -> L -> D -> Gate -> K -> outcome -> M

NOT a quantum-physics test.

Core questions:
1. Can contextual memory learn which policy works where?
2. Can Gate refuse stale/irrelevant experience?
3. Does provenance remain intact?
4. Can the system recover from deliberately wrong D?
5. Does memory beat "always B1" and a memoryless selector?

Policies:
    B1
    GLOBAL
    LOCAL
    HYBRID

D stores contextual performance, uncertainty,
validity, and provenance.

No LLM.
No embeddings.
No hidden access to true rewards.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class Config:
    seed: int = 20260925

    train_episodes: int = 600
    test_episodes: int = 400

    context_dim: int = 3

    # Similarity / Gate
    validity_radius: float = 0.42
    min_samples: int = 4

    # Confidence-aware selection
    uncertainty_penalty: float = 0.80

    # Exploration during learning
    exploration_rate: float = 0.12

    # Reward noise
    reward_sigma: float = 0.035

    # Number of context prototypes
    max_prototypes: int = 64

    # If Gate does not trust memory
    safe_policy: str = "B1"

    output_dir: str = "results_memory_v01"


CFG = Config()

POLICIES = (
    "B1",
    "GLOBAL",
    "LOCAL",
    "HYBRID",
)


# ============================================================
# Provenance helpers
# ============================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def config_hash(cfg: Config) -> str:
    payload = json.dumps(
        asdict(cfg),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return sha256_bytes(payload)


# ============================================================
# M — episodic memory
# ============================================================

@dataclass
class Episode:
    episode_id: int

    phase: str

    context: list[float]

    policy: str

    reward: float

    # Ground truth oracle is stored for offline evaluation,
    # but NEVER supplied to Gate/L during decision.
    oracle_policy: str

    gate_mode: str

    prototype_id: int | None


# ============================================================
# D — compiled experience
# ============================================================

@dataclass
class PolicyStats:
    count: int = 0

    reward_sum: float = 0.0

    reward_sq_sum: float = 0.0

    provenance: list[int] = field(
        default_factory=list
    )

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0

        return self.reward_sum / self.count

    @property
    def variance(self) -> float:
        if self.count < 2:
            return float("inf")

        mean = self.mean

        value = (
            self.reward_sq_sum
            / self.count
            - mean * mean
        )

        return max(value, 1e-9)

    @property
    def stderr(self) -> float:
        if self.count < 2:
            return float("inf")

        return math.sqrt(
            self.variance / self.count
        )


@dataclass
class Prototype:
    prototype_id: int

    center: list[float]

    visits: int = 0

    stats: dict[str, PolicyStats] = field(
        default_factory=lambda: {
            p: PolicyStats()
            for p in POLICIES
        }
    )


@dataclass
class CompiledMemory:
    prototypes: list[Prototype] = field(
        default_factory=list
    )

    next_prototype_id: int = 0


# ============================================================
# Synthetic environment
#
# Context:
#   x[0] = global structure strength
#   x[1] = local structure strength
#   x[2] = task sensitivity / mixedness
#
# Memory sees x, but does NOT see expected rewards.
#
# This is deliberately synthetic:
# v0.1 tests memory architecture, not quantum physics.
# ============================================================

def expected_rewards(
    context: np.ndarray,
    shifted: bool = False,
):
    g, l, t = context

    # Generalist baseline: robust, rarely optimal.
    b1 = (
        0.76
        + 0.08 * (1.0 - abs(g - l))
        + 0.04 * t
    )

    global_r = (
        0.66
        + 0.26 * g
        - 0.12 * l
        + 0.03 * t
    )

    local_r = (
        0.66
        + 0.26 * l
        - 0.12 * g
        + 0.03 * (1.0 - t)
    )

    hybrid = (
        0.68
        + 0.14 * g
        + 0.14 * l
        + 0.12 * t
        - 0.06 * abs(g - l)
    )

    if shifted:
        # Regime shift:
        # past experience is not perfectly valid anymore.
        #
        # B1 becomes slightly more robust.
        # Hybrid suffers in strongly global regimes.
        # Local improves in local-heavy contexts.
        b1 += 0.025

        hybrid -= 0.10 * g

        local_r += 0.07 * l

        global_r -= 0.04 * l

    values = {
        "B1": b1,
        "GLOBAL": global_r,
        "LOCAL": local_r,
        "HYBRID": hybrid,
    }

    return {
        k: float(
            np.clip(v, 0.0, 1.0)
        )
        for k, v in values.items()
    }


def sample_context(
    rng: np.random.Generator,
):
    return rng.uniform(
        0.0,
        1.0,
        size=3,
    )


def observe_reward(
    context,
    policy,
    rng,
    cfg,
    shifted=False,
):
    means = expected_rewards(
        context,
        shifted=shifted,
    )

    reward = rng.normal(
        means[policy],
        cfg.reward_sigma,
    )

    return float(
        np.clip(reward, 0.0, 1.0)
    )


def oracle_policy(
    context,
    shifted=False,
):
    means = expected_rewards(
        context,
        shifted=shifted,
    )

    return max(
        means,
        key=means.get,
    )


# ============================================================
# Distance / nearest prototype
# ============================================================

def distance(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a)
            - np.asarray(b)
        )
    )


def nearest_prototype(
    D: CompiledMemory,
    context,
):
    if not D.prototypes:
        return None, float("inf")

    distances = [
        distance(
            p.center,
            context,
        )
        for p in D.prototypes
    ]

    idx = int(
        np.argmin(distances)
    )

    return (
        D.prototypes[idx],
        distances[idx],
    )


# ============================================================
# Gate
# ============================================================

def confidence_score(
    stats: PolicyStats,
    cfg: Config,
):
    """
    Conservative value estimate.

    Higher is better.

    Unknown / poorly supported memories
    are penalized.
    """

    if stats.count < cfg.min_samples:
        return -float("inf")

    return (
        stats.mean
        - cfg.uncertainty_penalty
        * stats.stderr
    )


def gate_select(
    D: CompiledMemory,
    context,
    cfg: Config,
):
    """
    Gate decides whether D is applicable.

    Returns:
        policy
        gate_mode
        prototype_id
    """

    prototype, d = nearest_prototype(
        D,
        context,
    )

    if prototype is None:
        return (
            cfg.safe_policy,
            "NO_MEMORY",
            None,
        )

    if d > cfg.validity_radius:
        return (
            cfg.safe_policy,
            "OUT_OF_REGION",
            prototype.prototype_id,
        )

    scores = {
        policy: confidence_score(
            prototype.stats[policy],
            cfg,
        )
        for policy in POLICIES
    }

    if all(
        not np.isfinite(v)
        for v in scores.values()
    ):
        return (
            cfg.safe_policy,
            "LOW_CONFIDENCE",
            prototype.prototype_id,
        )

    policy = max(
        scores,
        key=scores.get,
    )

    if not np.isfinite(
        scores[policy]
    ):
        return (
            cfg.safe_policy,
            "LOW_CONFIDENCE",
            prototype.prototype_id,
        )

    return (
        policy,
        "MEMORY",
        prototype.prototype_id,
    )


# ============================================================
# L — compiler
# ============================================================

def create_prototype(
    D,
    context,
):
    prototype = Prototype(
        prototype_id=D.next_prototype_id,
        center=list(
            map(float, context)
        ),
    )

    D.next_prototype_id += 1

    D.prototypes.append(
        prototype
    )

    return prototype


def compiler_update(
    D: CompiledMemory,
    episode: Episode,
    cfg: Config,
):
    """
    L(D, M_new) -> D'

    No raw episode replay is needed here.
    The episode updates compact policy statistics.

    Provenance is retained as episode IDs.
    """

    context = np.asarray(
        episode.context
    )

    prototype, d = nearest_prototype(
        D,
        context,
    )

    if (
        prototype is None
        or (
            d > cfg.validity_radius
            and len(D.prototypes)
            < cfg.max_prototypes
        )
    ):
        prototype = create_prototype(
            D,
            context,
        )

    # Incrementally move prototype center.
    prototype.visits += 1

    center = np.asarray(
        prototype.center
    )

    rate = 1.0 / prototype.visits

    center = (
        (1.0 - rate) * center
        + rate * context
    )

    prototype.center = list(
        map(float, center)
    )

    stats = prototype.stats[
        episode.policy
    ]

    stats.count += 1

    stats.reward_sum += (
        episode.reward
    )

    stats.reward_sq_sum += (
        episode.reward ** 2
    )

    stats.provenance.append(
        episode.episode_id
    )


# ============================================================
# Deliberately wrong D
# ============================================================

def inject_wrong_memory(
    D: CompiledMemory,
):
    """
    Corrupt learned values without altering M.

    This tests whether stale/wrong compiled experience
    can hurt the system.

    Reversibility later means D can be rebuilt from M.
    """

    for prototype in D.prototypes:
        # Make GLOBAL falsely attractive.
        s = prototype.stats["GLOBAL"]

        s.count = max(s.count, 20)
        s.reward_sum = (
            s.count * 0.99
        )
        s.reward_sq_sum = (
            s.count * 0.99 ** 2
        )


# ============================================================
# Training
# ============================================================

def train_memory(
    cfg,
    rng,
):
    D = CompiledMemory()
    M: list[Episode] = []

    for episode_id in range(
        cfg.train_episodes
    ):
        context = sample_context(
            rng
        )

        # Explore enough to populate policy evidence.
        if (
            rng.random()
            < cfg.exploration_rate
        ):
            policy = rng.choice(
                POLICIES
            )

            gate_mode = "EXPLORE"
            prototype_id = None

        else:
            policy, gate_mode, prototype_id = (
                gate_select(
                    D,
                    context,
                    cfg,
                )
            )

            # During training, if Gate falls back,
            # occasionally force structured exploration.
            if gate_mode != "MEMORY":
                if rng.random() < 0.45:
                    policy = rng.choice(
                        POLICIES
                    )
                    gate_mode = (
                        "FALLBACK_EXPLORE"
                    )

        reward = observe_reward(
            context,
            policy,
            rng,
            cfg,
            shifted=False,
        )

        episode = Episode(
            episode_id=episode_id,
            phase="TRAIN",
            context=list(
                map(float, context)
            ),
            policy=str(policy),
            reward=reward,
            oracle_policy=oracle_policy(
                context,
                shifted=False,
            ),
            gate_mode=gate_mode,
            prototype_id=prototype_id,
        )

        M.append(episode)

        compiler_update(
            D,
            episode,
            cfg,
        )

    return M, D


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    D,
    cfg,
    rng,
    episodes,
    shifted,
    start_id,
    update_memory=False,
):
    rows = []

    for k in range(episodes):
        episode_id = start_id + k

        context = sample_context(
            rng
        )

        selected, mode, proto_id = (
            gate_select(
                D,
                context,
                cfg,
            )
        )

        reward = observe_reward(
            context,
            selected,
            rng,
            cfg,
            shifted=shifted,
        )

        means = expected_rewards(
            context,
            shifted=shifted,
        )

        oracle = max(
            means,
            key=means.get,
        )

        oracle_reward = means[
            oracle
        ]

        selected_expected = means[
            selected
        ]

        b1_expected = means["B1"]

        regret = (
            oracle_reward
            - selected_expected
        )

        b1_regret = (
            oracle_reward
            - b1_expected
        )

        row = {
            "episode_id":
                episode_id,

            "context_g":
                context[0],

            "context_l":
                context[1],

            "context_t":
                context[2],

            "selected":
                selected,

            "gate_mode":
                mode,

            "prototype_id":
                proto_id,

            "observed_reward":
                reward,

            "expected_selected":
                selected_expected,

            "b1_expected":
                b1_expected,

            "oracle":
                oracle,

            "oracle_expected":
                oracle_reward,

            "regret":
                regret,

            "b1_regret":
                b1_regret,

            "memory_advantage":
                b1_regret - regret,

            "shifted":
                int(shifted),
        }

        rows.append(row)

        if update_memory:
            episode = Episode(
                episode_id=episode_id,
                phase=(
                    "SHIFT"
                    if shifted
                    else "TEST"
                ),
                context=list(
                    map(float, context)
                ),
                policy=selected,
                reward=reward,
                oracle_policy=oracle,
                gate_mode=mode,
                prototype_id=proto_id,
            )

            compiler_update(
                D,
                episode,
                cfg,
            )

    return rows


# ============================================================
# Paired bootstrap
# ============================================================

def paired_bootstrap_ci(
    values,
    seed,
    samples=10000,
    alpha=0.05,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(samples)

    for b in range(samples):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        means[b] = np.mean(
            values[idx]
        )

    lo = np.quantile(
        means,
        alpha / 2,
    )

    hi = np.quantile(
        means,
        1 - alpha / 2,
    )

    return (
        float(lo),
        float(hi),
    )

# ============================================================
# Serialization
# ============================================================

def serialize_D(D):
    return {
        "next_prototype_id":
            D.next_prototype_id,

        "prototypes": [
            {
                "prototype_id":
                    p.prototype_id,

                "center":
                    p.center,

                "visits":
                    p.visits,

                "stats": {
                    policy: {
                        "count":
                            s.count,

                        "mean":
                            (
                                s.mean
                                if s.count
                                else None
                            ),

                        "stderr":
                            (
                                s.stderr
                                if s.count >= 2
                                else None
                            ),

                        "provenance":
                            s.provenance,
                    }
                    for policy, s
                    in p.stats.items()
                },
            }
            for p in D.prototypes
        ],
    }


def save_rows(
    path,
    rows,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Summary
# ============================================================

def summarize(
    name,
    rows,
    seed,
):
    regret = np.asarray(
        [
            r["regret"]
            for r in rows
        ]
    )

    b1_regret = np.asarray(
        [
            r["b1_regret"]
            for r in rows
        ]
    )

    memory_advantage = (
        b1_regret - regret
    )

    reward = np.asarray(
        [
            r["observed_reward"]
            for r in rows
        ]
    )

    memory_fraction = np.mean(
        [
            r["gate_mode"]
            == "MEMORY"
            for r in rows
        ]
    )

    oracle_match = np.mean(
        [
            r["selected"]
            == r["oracle"]
            for r in rows
        ]
    )

    ci = paired_bootstrap_ci(
        memory_advantage,
        seed=seed,
        samples=10000,
        alpha=0.05,
    )

    result = {
        "name": name,

        "mean_regret":
            float(
                np.mean(regret)
            ),

        "median_regret":
            float(
                np.median(regret)
            ),

        "mean_observed_reward":
            float(
                np.mean(reward)
            ),

        "oracle_match":
            float(oracle_match),

        "memory_gate_fraction":
            float(memory_fraction),

                "mean_b1_regret":
            float(
                np.mean(b1_regret)
            ),

        "memory_vs_b1":
            float(
                np.mean(memory_advantage)
            ),

        "memory_vs_b1_95ci":
            list(ci),

        "fraction_positive_advantage":
            float(
                np.mean(
                    memory_advantage > 0
                )
            ),
    }

    print(
        f"\n{name}"
    )
    print(
        "-" * len(name)
    )

    for k, v in result.items():
        if k == "name":
            continue

        if isinstance(v, list):
            formatted = ", ".join(
                f"{x:+.6f}" for x in v
            )
            print(
                f"{k:22s}: [{formatted}]"
            )
        else:
            print(
                f"{k:22s}: {v:.6f}"
            )

    return result


# ============================================================
# Main
# ============================================================

def main():
    cfg = CFG

    out = Path(
        cfg.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = Path(
        __file__
    ).resolve()

    print(
        "Q-IGL Persistent Memory v0.1"
    )
    print(
        "============================"
    )

    print(
        "SOURCE_SHA256 =",
        sha256_file(source)
    )

    print(
        "CONFIG_SHA256 =",
        config_hash(cfg)
    )

    master = np.random.SeedSequence(
        cfg.seed
    )

    (
        train_seq,
        test_seq,
        shift_seq,
        wrong_seq,
    ) = master.spawn(4)

    train_rng = (
        np.random.default_rng(
            train_seq
        )
    )

    test_rng = (
        np.random.default_rng(
            test_seq
        )
    )

    shift_rng = (
        np.random.default_rng(
            shift_seq
        )
    )

    wrong_rng = (
        np.random.default_rng(
            wrong_seq
        )
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nTraining memory..."
    )

    M, D = train_memory(
        cfg,
        train_rng,
    )

    d_before_path = (
        out / "D_trained.json"
    )

    d_before_path.write_text(
        json.dumps(
            serialize_D(D),
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "prototypes =",
        len(D.prototypes)
    )

    print(
        "D_SHA256 =",
        sha256_file(
            d_before_path
        )
    )

    # ========================================================
    # TEST 1: familiar distribution
    # ========================================================

    familiar_rows = evaluate(
        D,
        cfg,
        test_rng,
        cfg.test_episodes,
        shifted=False,
        start_id=100000,
        update_memory=False,
    )

    # ========================================================
    # TEST 2: regime shift, stale D
    # ========================================================

    shift_rows = evaluate(
        D,
        cfg,
        shift_rng,
        cfg.test_episodes,
        shifted=True,
        start_id=200000,
        update_memory=False,
    )

    # ========================================================
    # TEST 3: deliberately corrupted D
    #
    # Deep-copy through serialization would be cumbersome;
    # reconstruct training once with identical RNG seed.
    # ========================================================

    replay_train_rng = (
        np.random.default_rng(
            train_seq
        )
    )

    _, D_wrong = train_memory(
        cfg,
        replay_train_rng,
    )

    inject_wrong_memory(
        D_wrong
    )

    wrong_rows = evaluate(
        D_wrong,
        cfg,
        wrong_rng,
        cfg.test_episodes,
        shifted=False,
        start_id=300000,
        update_memory=False,
    )

    # ========================================================
    # Save
    # ========================================================

    familiar_path = (
        out / "familiar.csv"
    )

    shift_path = (
        out / "shift.csv"
    )

    wrong_path = (
        out / "wrong_D.csv"
    )

    save_rows(
        familiar_path,
        familiar_rows,
    )

    save_rows(
        shift_path,
        shift_rows,
    )

    save_rows(
        wrong_path,
        wrong_rows,
    )

    summaries = [
        summarize(
            "FAMILIAR",
            familiar_rows,
            seed=cfg.seed + 1,
        ),

        summarize(
            "REGIME SHIFT / STALE D",
            shift_rows,
            seed=cfg.seed + 2,
        ),

        summarize(
            "WRONG D",
            wrong_rows,
            seed=cfg.seed + 3,
        ),
    ]

    criteria = {
        "FAMILIAR":
            "lower 95% CI(memory_vs_b1) > 0 "
            "→ memory adds contextual value",

        "STALE":
            "compare memory_vs_b1 vs familiar; "
            "expect degradation. Not required "
            "to be negative.",

        "WRONG_D":
            "upper 95% CI(memory_vs_b1) < 0 "
            "→ wrong memory is provably worse "
            "than safe baseline",
    }

    summary = {
        "config":
            asdict(cfg),

        "summaries":
            summaries,

        "artifacts": {
            "D_trained_sha256":
                sha256_file(
                    d_before_path
                ),

            "familiar_sha256":
                sha256_file(
                    familiar_path
                ),

            "shift_sha256":
                sha256_file(
                    shift_path
                ),

            "wrong_sha256":
                sha256_file(
                    wrong_path
                ),
        },

        "criteria":
            criteria,
    }

    summary_path = (
        out / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "\nSUMMARY_SHA256 =",
        sha256_file(
            summary_path
        )
    )

    print(
        "\nMemory v0.1 complete."
    )
    print(
        "No automatic success verdict."
    )


if __name__ == "__main__":
    main()