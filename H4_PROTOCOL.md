H4 — Task-Weighted Context Hypothesis

Frozen:
1 noise-generating process.
3 task sensitivities: global/local/mixed.
4 controller arms: B1/G/L/H.
Training telemetry separate from test trajectories.

Selector sees:
Sigma_hat + predefined task vector h.
Selector never sees test fidelities.

Selection:
q < .50       -> B1
rG >= .75     -> Global
rL >= .75     -> Local
otherwise     -> Hybrid

Oracle:
best mean test fidelity arm per task.

Primary:
mean regret across three tasks.

Comparators:
always-B1
always-Global
always-Local
always-Hybrid
random

Support:
selector regret < always-Hybrid regret
and selector regret < random regret.

Strong support:
selector <= always-B1,
and changing h changes selected topology in the
pre-registered direction, subsequently consistent
with test ranking.

Regardless of result:
NO v25.
Next project = Persistent Memory.