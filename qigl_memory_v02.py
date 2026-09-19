"""
qigl_memory_v02.py

Q-IGL Persistent Memory v0.2
============================

Tests:

    Exist(D) != Trust(D) != Apply(D)

and:

    InsufficientEvidence != NegativeEvidence

Changes from v0.1:
1. Coverage debt.
2. LEARN vs DEPLOY.
3. UCB in LEARN / LCB in DEPLOY.
4. Surprise EWMA -> trust invalidation.
5. D generations after confirmed mismatch.
6. Paired immutable streams for fair comparison.

Environment:
expected_rewards() is unchanged from Memory v0.1.

Arms:
    ALWAYS_B1
    V01_LOGIC
    V02_ADAPTIVE
    ORACLE (offline only)

No LLM.
No embeddings.
No quantum claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


# ============================================================
# Frozen config
# ============================================================

@dataclass(frozen=True)
class Config:
    seed: int = 20260926

    context_dim: int = 3

    # Pre-deployment learning
    train_episodes: int = 800

    # Familiar primary test
    familiar_episodes: int = 1000

    # Recovery tests
    recovery_episodes: int = 800
    event_step: int = 300

    # Prototypes
    validity_radius: float = 0.42
    max_prototypes: int = 64

    # Evidence
    min_samples: int = 4

    # Learning / deployment confidence
    ucb_beta: float = 1.00
    lcb_lambda: float = 0.80

    # Small residual exploration even after coverage
    deploy_exploration: float = 0.05

    # Known observation noise
    reward_sigma: float = 0.035

    # Trust / mismatch
    surprise_alpha: float = 0.18
    surprise_threshold: float = 2.20
    surprise_consecutive: int = 4
    sigma_floor: float = 0.035

    # Recovery: minimum observations before trusted again
    recovery_min_samples: int = 4

    # Bootstrap
    bootstrap_samples: int = 10000
    ci_alpha: float = 0.05

    safe_policy: str = "B1"

    output_dir: str = "results_memory_v02"


CFG = Config()

POLICIES = (
    "B1",
    "GLOBAL",
    "LOCAL",
    "HYBRID",
)


# ============================================================
# Provenance
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
# Synthetic environment
#
# EXACT reward model from Memory v0.1.
# ============================================================

def expected_rewards(
    context: np.ndarray,
    shifted: bool = False,
):
    g, l, t = context

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
        k: float(np.clip(v, 0.0, 1.0))
        for k, v in values.items()
    }


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
# Immutable paired stream
# ============================================================

@dataclass
class StreamItem:
    t: int
    context: np.ndarray

    # One frozen noise draw per possible policy.
    reward_noise: dict[str, float]


def generate_stream(
    cfg: Config,
    episodes: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    stream = []

    for t in range(episodes):
        context = rng.uniform(
            0.0,
            1.0,
            size=cfg.context_dim,
        )

        reward_noise = {
            p: float(
                rng.normal(
                    0.0,
                    cfg.reward_sigma,
                )
            )
            for p in POLICIES
        }

        stream.append(
            StreamItem(
                t=t,
                context=context,
                reward_noise=reward_noise,
            )
        )

    return stream


def realized_reward(
    item: StreamItem,
    policy: str,
    shifted=False,
):
    mu = expected_rewards(
        item.context,
        shifted=shifted,
    )[policy]

    return float(
        np.clip(
            mu
            + item.reward_noise[policy],
            0.0,
            1.0,
        )
    )


# ============================================================
# Memory structures
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
    def mean(self):
        if self.count == 0:
            return 0.0

        return self.reward_sum / self.count

    @property
    def variance(self):
        if self.count < 2:
            return float("inf")

        m = self.mean

        value = (
            self.reward_sq_sum
            / self.count
            - m * m
        )

        return max(value, 1e-9)

    @property
    def stderr(self):
        if self.count < 2:
            return float("inf")

        return math.sqrt(
            self.variance
            / self.count
        )

    def update(
        self,
        reward,
        episode_id,
    ):
        self.count += 1
        self.reward_sum += reward
        self.reward_sq_sum += reward * reward
        self.provenance.append(episode_id)


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
class MemoryGeneration:
    generation_id: int

    prototypes: list[Prototype] = field(
        default_factory=list
    )

    next_prototype_id: int = 0

    trusted: bool = True

    surprise_ewma: float = 0.0
    high_surprise_streak: int = 0

    invalidated_at: int | None = None


@dataclass
class MemoryState:
    generations: list[MemoryGeneration] = field(
        default_factory=list
    )

    active_generation: int = 0

    def active(self):
        return self.generations[
            self.active_generation
        ]


# ============================================================
# Geometry
# ============================================================

def distance(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a)
            - np.asarray(b)
        )
    )


def nearest_prototype(
    generation,
    context,
):
    if not generation.prototypes:
        return None, float("inf")

    distances = [
        distance(
            p.center,
            context,
        )
        for p in generation.prototypes
    ]

    idx = int(np.argmin(distances))

    return (
        generation.prototypes[idx],
        distances[idx],
    )


def create_prototype(
    generation,
    context,
):
    p = Prototype(
        prototype_id=(
            generation.next_prototype_id
        ),
        center=list(
            map(float, context)
        ),
    )

    generation.next_prototype_id += 1
    generation.prototypes.append(p)

    return p


# ============================================================
# Compiler L
# ============================================================

def compiler_update(
    generation: MemoryGeneration,
    context,
    policy,
    reward,
    episode_id,
    cfg,
):
    p, d = nearest_prototype(
        generation,
        context,
    )

    if (
        p is None
        or (
            d > cfg.validity_radius
            and len(generation.prototypes)
            < cfg.max_prototypes
        )
    ):
        p = create_prototype(
            generation,
            context,
        )

    p.visits += 1

    center = np.asarray(
        p.center,
        dtype=float,
    )

    rate = 1.0 / p.visits

    center = (
        (1.0 - rate) * center
        + rate * np.asarray(context)
    )

    p.center = list(
        map(float, center)
    )

    p.stats[policy].update(
        reward,
        episode_id,
    )

    return p


# ============================================================
# Coverage / confidence
# ============================================================

def evidence_debt(
    prototype,
    cfg,
):
    return {
        policy: max(
            0,
            cfg.min_samples
            - prototype.stats[policy].count,
        )
        for policy in POLICIES
    }


def coverage_complete(
    prototype,
    cfg,
):
    debt = evidence_debt(
        prototype,
        cfg,
    )

    return all(
        value == 0
        for value in debt.values()
    )


def ucb_score(
    stats,
    cfg,
):
    if stats.count == 0:
        return float("inf")

    if stats.count == 1:
        # Still deliberately attractive.
        return stats.mean + 1.0

    return (
        stats.mean
        + cfg.ucb_beta
        * stats.stderr
    )


def lcb_score(
    stats,
    cfg,
):
    if stats.count < cfg.min_samples:
        return -float("inf")

    return (
        stats.mean
        - cfg.lcb_lambda
        * stats.stderr
    )


# ============================================================
# V02 Gate
# ============================================================

def gate_v02(
    state: MemoryState,
    context,
    cfg,
    rng,
):
    generation = state.active()

    # Exist(D) but Trust(D) == false.
    if not generation.trusted:
        return (
            cfg.safe_policy,
            "UNTRUSTED_FALLBACK",
            None,
        )

    p, d = nearest_prototype(
        generation,
        context,
    )

    if p is None:
        return (
            cfg.safe_policy,
            "NO_MEMORY",
            None,
        )

    if d > cfg.validity_radius:
        # Unknown region: do not pretend memory applies.
        return (
            cfg.safe_policy,
            "OUT_OF_REGION",
            p.prototype_id,
        )

    debt = evidence_debt(
        p,
        cfg,
    )

    # ----------------------------------------
    # LEARN MODE:
    # Unknown != bad.
    # ----------------------------------------
    if any(
        value > 0
        for value in debt.values()
    ):
        max_debt = max(
            debt.values()
        )

        candidates = [
            policy
            for policy, value
            in debt.items()
            if value == max_debt
        ]

        # Among equally underexplored policies,
        # UCB breaks ties.
        policy = max(
            candidates,
            key=lambda x:
                ucb_score(
                    p.stats[x],
                    cfg,
                )
        )

        return (
            policy,
            "LEARN_COVERAGE",
            p.prototype_id,
        )

    # Small continuing exploration.
    if (
        rng.random()
        < cfg.deploy_exploration
    ):
        scores = {
            policy: ucb_score(
                p.stats[policy],
                cfg,
            )
            for policy in POLICIES
        }

        policy = max(
            scores,
            key=scores.get,
        )

        return (
            policy,
            "LEARN_UCB",
            p.prototype_id,
        )

    # ----------------------------------------
    # DEPLOY MODE:
    # uncertainty now penalizes policy.
    # ----------------------------------------
    scores = {
        policy: lcb_score(
            p.stats[policy],
            cfg,
        )
        for policy in POLICIES
    }

    policy = max(
        scores,
        key=scores.get,
    )

    return (
        policy,
        "DEPLOY",
        p.prototype_id,
    )


# ============================================================
# v01 frozen approximation
# ============================================================

def gate_v01(
    generation,
    context,
    cfg,
):
    p, d = nearest_prototype(
        generation,
        context,
    )

    if p is None:
        return (
            cfg.safe_policy,
            "NO_MEMORY",
            None,
        )

    if d > cfg.validity_radius:
        return (
            cfg.safe_policy,
            "OUT_OF_REGION",
            p.prototype_id,
        )

    scores = {}

    for policy in POLICIES:
        stats = p.stats[policy]

        if stats.count < cfg.min_samples:
            scores[policy] = -float("inf")
        else:
            scores[policy] = (
                stats.mean
                - cfg.lcb_lambda
                * stats.stderr
            )

    if all(
        not np.isfinite(v)
        for v in scores.values()
    ):
        return (
            cfg.safe_policy,
            "LOW_CONFIDENCE",
            p.prototype_id,
        )

    policy = max(
        scores,
        key=scores.get,
    )

    return (
        policy,
        "MEMORY",
        p.prototype_id,
    )


# ============================================================
# Trust monitor
# ============================================================

def expected_from_stats(
    prototype,
    policy,
):
    if prototype is None:
        return None, None

    s = prototype.stats[policy]

    if s.count < 2:
        return None, None

    return s.mean, s.stderr


def update_trust(
    state: MemoryState,
    prototype,
    policy,
    observed_reward,
    t,
    cfg,
):
    generation = state.active()

    mu, se = expected_from_stats(
        prototype,
        policy,
    )

    if mu is None:
        return False

    denom = math.sqrt(
        se * se
        + cfg.sigma_floor ** 2
    )

    z = abs(
        observed_reward - mu
    ) / max(
        denom,
        1e-9,
    )

    a = cfg.surprise_alpha

    generation.surprise_ewma = (
        (1.0 - a)
        * generation.surprise_ewma
        + a * z
    )

    if (
        generation.surprise_ewma
        > cfg.surprise_threshold
    ):
        generation.high_surprise_streak += 1
    else:
        generation.high_surprise_streak = 0

    if (
        generation.high_surprise_streak
        >= cfg.surprise_consecutive
    ):
        generation.trusted = False
        generation.invalidated_at = t
        return True

    return False


def open_new_generation(
    state: MemoryState,
):
    generation = MemoryGeneration(
        generation_id=len(
            state.generations
        )
    )

    state.generations.append(
        generation
    )

    state.active_generation = (
        len(state.generations) - 1
    )

    return generation


# ============================================================
# Training v02
# ============================================================

def train_v02(
    cfg,
    stream,
    rng,
):
    state = MemoryState(
        generations=[
            MemoryGeneration(
                generation_id=0
            )
        ],
        active_generation=0,
    )

    rows = []

    for item in stream:
        policy, mode, pid = gate_v02(
            state,
            item.context,
            cfg,
            rng,
        )

        reward = realized_reward(
            item,
            policy,
            shifted=False,
        )

        p = compiler_update(
            state.active(),
            item.context,
            policy,
            reward,
            item.t,
            cfg,
        )

        rows.append({
            "t": item.t,
            "policy": policy,
            "mode": mode,
            "reward": reward,
            "prototype_id":
                p.prototype_id,
        })

    return state, rows


# ============================================================
# Rebuild a v01 state on SAME stream
# ============================================================

def train_v01_like(
    cfg,
    stream,
    rng,
):
    generation = MemoryGeneration(
        generation_id=0
    )

    for item in stream:
        policy, mode, pid = gate_v01(
            generation,
            item.context,
            cfg,
        )

        # Approximate v0.1 residual exploration.
        if rng.random() < 0.12:
            policy = str(
                rng.choice(POLICIES)
            )

        reward = realized_reward(
            item,
            policy,
            shifted=False,
        )

        compiler_update(
            generation,
            item.context,
            policy,
            reward,
            item.t,
            cfg,
        )

    return generation


# ============================================================
# Corruption
# ============================================================

def corrupt_active_D(
    state: MemoryState,
):
    """
    Same basic failure mode as v0.1:
    make GLOBAL falsely attractive and overconfident.
    """

    generation = state.active()

    for p in generation.prototypes:
        s = p.stats["GLOBAL"]

        s.count = max(
            s.count,
            30,
        )

        s.reward_sum = (
            0.99 * s.count
        )

        s.reward_sq_sum = (
            (0.99 ** 2)
            * s.count
        )


# ============================================================
# Evaluation helpers
# ============================================================

def oracle_info(
    item,
    shifted,
):
    means = expected_rewards(
        item.context,
        shifted=shifted,
    )

    oracle = max(
        means,
        key=means.get,
    )

    return (
        means,
        oracle,
        means[oracle],
    )


def regret_for_policy(
    item,
    policy,
    shifted,
):
    means, oracle, oracle_value = (
        oracle_info(
            item,
            shifted,
        )
    )

    return (
        oracle_value
        - means[policy]
    )


# ============================================================
# Familiar paired test
# ============================================================

def familiar_test(
    cfg,
    stream,
    v01_generation,
    v02_state,
    rng_v02,
):
    rows = []

    for item in stream:
        # ----- B1 -----

        b1 = "B1"

        # ----- v01 -----

        v01_policy, _, _ = gate_v01(
            v01_generation,
            item.context,
            cfg,
        )

        # ----- v02 -----

        v02_policy, mode, pid = gate_v02(
            v02_state,
            item.context,
            cfg,
            rng_v02,
        )

        r_b1 = regret_for_policy(
            item,
            b1,
            shifted=False,
        )

        r_v01 = regret_for_policy(
            item,
            v01_policy,
            shifted=False,
        )

        r_v02 = regret_for_policy(
            item,
            v02_policy,
            shifted=False,
        )

        rows.append({
            "t": item.t,

            "b1_policy":
                b1,

            "v01_policy":
                v01_policy,

            "v02_policy":
                v02_policy,

            "v02_mode":
                mode,

            "regret_b1":
                r_b1,

            "regret_v01":
                r_v01,

            "regret_v02":
                r_v02,

            "adv_v02_vs_b1":
                r_b1 - r_v02,

            "adv_v02_vs_v01":
                r_v01 - r_v02,
        })

    return rows


# ============================================================
# Adaptive recovery test
# ============================================================

def recovery_test(
    cfg,
    initial_state,
    stream,
    rng,
    event_type,
):
    """
    event_type:
        "CORRUPT_D"
        "REGIME_SHIFT"

    Before event:
        normal environment

    After event:
        CORRUPT_D:
            environment unchanged,
            D is corrupted.

        REGIME_SHIFT:
            environment switches to shifted=True,
            D itself initially unchanged.

    v02 learns online during recovery.
    """

    state = initial_state

    rows = []

    event_happened = False

    detection_t = None
    new_generation_t = None

    for item in stream:
        t = item.t

        if (
            t == cfg.event_step
            and not event_happened
        ):
            if event_type == "CORRUPT_D":
                corrupt_active_D(
                    state
                )

            event_happened = True

        shifted = (
            event_type
            == "REGIME_SHIFT"
            and t >= cfg.event_step
        )

        policy, mode, pid = gate_v02(
            state,
            item.context,
            cfg,
            rng,
        )

        reward = realized_reward(
            item,
            policy,
            shifted=shifted,
        )

        generation = state.active()

        p, d = nearest_prototype(
            generation,
            item.context,
        )

        invalidated = update_trust(
            state,
            p,
            policy,
            reward,
            t,
            cfg,
        )

        if invalidated:
            detection_t = t

            open_new_generation(
                state
            )

            new_generation_t = t

            # Current surprising observation
            # becomes first evidence in new D.
            compiler_update(
                state.active(),
                item.context,
                policy,
                reward,
                1_000_000 + t,
                cfg,
            )

            mode = (
                "INVALIDATE_AND_RECOMPILE"
            )

        else:
            compiler_update(
                state.active(),
                item.context,
                policy,
                reward,
                1_000_000 + t,
                cfg,
            )

        r_v02 = regret_for_policy(
            item,
            policy,
            shifted=shifted,
        )

        r_b1 = regret_for_policy(
            item,
            "B1",
            shifted=shifted,
        )

        rows.append({
            "t": t,
            "shifted": int(shifted),

            "generation":
                state.active_generation,

            "policy":
                policy,

            "mode":
                mode,

            "trusted":
                int(
                    state.active().trusted
                ),

            "reward":
                reward,

            "regret_v02":
                r_v02,

            "regret_b1":
                r_b1,

            "adv_v02_vs_b1":
                r_b1 - r_v02,
        })

    return (
        rows,
        detection_t,
        new_generation_t,
    )


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_ci(
    values,
    cfg,
    seed,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    boot = np.empty(
        cfg.bootstrap_samples
    )

    for b in range(
        cfg.bootstrap_samples
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[b] = np.mean(
            values[idx]
        )

    lo = np.quantile(
        boot,
        cfg.ci_alpha / 2,
    )

    hi = np.quantile(
        boot,
        1 - cfg.ci_alpha / 2,
    )

    return float(lo), float(hi)


# ============================================================
# Serialization helpers
# ============================================================

def save_csv(
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


def generation_summary(
    generation,
    cfg,
):
    policy_counts = {
        p: 0
        for p in POLICIES
    }

    full_coverage = 0

    for proto in generation.prototypes:
        for policy in POLICIES:
            policy_counts[
                policy
            ] += (
                proto.stats[
                    policy
                ].count
            )

        if coverage_complete(
            proto,
            cfg,
        ):
            full_coverage += 1

    return {
        "generation_id":
            generation.generation_id,

        "trusted":
            generation.trusted,

        "prototype_count":
            len(
                generation.prototypes
            ),

        "policy_counts":
            policy_counts,

        "full_coverage_prototypes":
            full_coverage,
    }


def clone_state(
    state: MemoryState,
):
    """
    Exact in-memory clone via JSON-like reconstruction
    would be verbose.

    deepcopy is appropriate here: D is an experimental
    derived artifact, not external state.
    """
    import copy
    return copy.deepcopy(state)


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
        "Q-IGL Persistent Memory v0.2"
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
        train_seed,
        familiar_seed,
        corruption_seed,
        shift_seed,
        gate_train_seed,
        gate_familiar_seed,
        gate_corruption_seed,
        gate_shift_seed,
        v01_seed,
    ) = master.spawn(9)

    # ========================================================
    # Immutable streams
    # ========================================================

    train_stream = generate_stream(
        cfg,
        cfg.train_episodes,
        int(
            train_seed.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        ),
    )

    familiar_stream = generate_stream(
        cfg,
        cfg.familiar_episodes,
        int(
            familiar_seed.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        ),
    )

    corruption_stream = generate_stream(
        cfg,
        cfg.recovery_episodes,
        int(
            corruption_seed.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        ),
    )

    shift_stream = generate_stream(
        cfg,
        cfg.recovery_episodes,
        int(
            shift_seed.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        ),
    )

    # ========================================================
    # TRAIN v02
    # ========================================================

    rng_train_gate = np.random.default_rng(
        gate_train_seed
    )

    v02_state, train_rows = train_v02(
        cfg,
        train_stream,
        rng_train_gate,
    )

    g0 = v02_state.active()

    print(
        "\nTRAIN COVERAGE"
    )
    print(
        "--------------"
    )

    train_summary = generation_summary(
        g0,
        cfg,
    )

    print(
        json.dumps(
            train_summary,
            indent=2,
        )
    )

    # ========================================================
    # TRAIN v01-like on identical training stream
    # ========================================================

    rng_v01 = np.random.default_rng(
        v01_seed
    )

    v01_generation = train_v01_like(
        cfg,
        train_stream,
        rng_v01,
    )

    # ========================================================
    # FAMILIAR PRIMARY
    # ========================================================

    rng_familiar_gate = (
        np.random.default_rng(
            gate_familiar_seed
        )
    )

    familiar_rows = familiar_test(
        cfg,
        familiar_stream,
        v01_generation,
        clone_state(v02_state),
        rng_familiar_gate,
    )

    adv_b1 = np.asarray(
        [
            r["adv_v02_vs_b1"]
            for r in familiar_rows
        ]
    )

    adv_v01 = np.asarray(
        [
            r["adv_v02_vs_v01"]
            for r in familiar_rows
        ]
    )

    ci_b1 = bootstrap_ci(
        adv_b1,
        cfg,
        cfg.seed + 1,
    )

    ci_v01 = bootstrap_ci(
        adv_v01,
        cfg,
        cfg.seed + 2,
    )

    print(
        "\nFAMILIAR PRIMARY"
    )
    print(
        "----------------"
    )

    print(
        "mean regret B1 :",
        f"{np.mean([r['regret_b1'] for r in familiar_rows]):.6f}"
    )

    print(
        "mean regret v01:",
        f"{np.mean([r['regret_v01'] for r in familiar_rows]):.6f}"
    )

    print(
        "mean regret v02:",
        f"{np.mean([r['regret_v02'] for r in familiar_rows]):.6f}"
    )

    print(
        "v02 advantage vs B1:",
        f"{np.mean(adv_b1):+.6f}",
        f"CI [{ci_b1[0]:+.6f}, {ci_b1[1]:+.6f}]"
    )

    print(
        "v02 advantage vs v01:",
        f"{np.mean(adv_v01):+.6f}",
        f"CI [{ci_v01[0]:+.6f}, {ci_v01[1]:+.6f}]"
    )

    # ========================================================
    # CORRUPTION RECOVERY
    # ========================================================

    rng_corrupt_gate = (
        np.random.default_rng(
            gate_corruption_seed
        )
    )

    (
        corrupt_rows,
        corrupt_detect,
        corrupt_generation,
    ) = recovery_test(
        cfg,
        clone_state(v02_state),
        corruption_stream,
        rng_corrupt_gate,
        event_type="CORRUPT_D",
    )

    # ========================================================
    # REGIME SHIFT RECOVERY
    # ========================================================

    rng_shift_gate = (
        np.random.default_rng(
            gate_shift_seed
        )
    )

    (
        shift_rows,
        shift_detect,
        shift_generation,
    ) = recovery_test(
        cfg,
        clone_state(v02_state),
        shift_stream,
        rng_shift_gate,
        event_type="REGIME_SHIFT",
    )

    print(
        "\nRECOVERY"
    )
    print(
        "--------"
    )

    print(
        "corruption event:",
        cfg.event_step,
        "detected:",
        corrupt_detect,
        "delay:",
        (
            None
            if corrupt_detect is None
            else corrupt_detect
            - cfg.event_step
        )
    )

    print(
        "regime shift event:",
        cfg.event_step,
        "detected:",
        shift_detect,
        "delay:",
        (
            None
            if shift_detect is None
            else shift_detect
            - cfg.event_step
        )
    )

    # ========================================================
    # Save artifacts
    # ========================================================

    train_path = out / "train.csv"
    familiar_path = out / "familiar.csv"
    corrupt_path = out / "corruption.csv"
    shift_path = out / "regime_shift.csv"

    save_csv(
        train_path,
        train_rows,
    )

    save_csv(
        familiar_path,
        familiar_rows,
    )

    save_csv(
        corrupt_path,
        corrupt_rows,
    )

    save_csv(
        shift_path,
        shift_rows,
    )

    summary = {
        "config":
            asdict(cfg),

        "train":
            train_summary,

        "familiar": {
            "mean_regret_b1":
                float(np.mean(
                    [
                        r["regret_b1"]
                        for r in familiar_rows
                    ]
                )),

            "mean_regret_v01":
                float(np.mean(
                    [
                        r["regret_v01"]
                        for r in familiar_rows
                    ]
                )),

            "mean_regret_v02":
                float(np.mean(
                    [
                        r["regret_v02"]
                        for r in familiar_rows
                    ]
                )),

            "advantage_vs_b1":
                float(np.mean(adv_b1)),

            "advantage_vs_b1_ci95":
                list(ci_b1),

            "advantage_vs_v01":
                float(np.mean(adv_v01)),

            "advantage_vs_v01_ci95":
                list(ci_v01),
        },

        "corruption": {
            "event_step":
                cfg.event_step,

            "detected_at":
                corrupt_detect,

            "detection_delay":
                (
                    None
                    if corrupt_detect is None
                    else corrupt_detect
                    - cfg.event_step
                ),
        },

        "regime_shift": {
            "event_step":
                cfg.event_step,

            "detected_at":
                shift_detect,

            "detection_delay":
                (
                    None
                    if shift_detect is None
                    else shift_detect
                    - cfg.event_step
                ),
        },

        "artifacts": {
            "train_sha256":
                sha256_file(train_path),

            "familiar_sha256":
                sha256_file(familiar_path),

            "corruption_sha256":
                sha256_file(corrupt_path),

            "shift_sha256":
                sha256_file(shift_path),
        }
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
        "\nARTIFACTS"
    )
    print(
        "---------"
    )

    print(
        "TRAIN_SHA256      =",
        sha256_file(train_path)
    )

    print(
        "FAMILIAR_SHA256   =",
        sha256_file(familiar_path)
    )

    print(
        "CORRUPTION_SHA256 =",
        sha256_file(corrupt_path)
    )

    print(
        "SHIFT_SHA256      =",
        sha256_file(shift_path)
    )

    print(
        "SUMMARY_SHA256    =",
        sha256_file(summary_path)
    )

    print(
        "\nNo automatic success verdict."
    )


if __name__ == "__main__":
    main()