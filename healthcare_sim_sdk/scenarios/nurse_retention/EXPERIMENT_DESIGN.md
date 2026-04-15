# Nurse Retention Scenario — Experiment Design

## What We're Simulating

A vendor (Laudio) sells nurse manager tools. One tool is an ML predictor on nurse turnover risk. It flags nurses as low / medium / high risk so nurse managers can decide who gets targeted check-ins to improve retention.

We met with the vendor. Their claims are backed by a large observational study (Laudio + AONL, April 2025, 5,000+ managers, ~75,000 nurses, 100+ hospitals) — not an RCT. They report +6pp retention improvement early and +13pp later in the first year from targeted manager engagement.

This scenario stress-tests those claims under Geisinger-realistic conditions using the healthcare-sim-sdk's branched counterfactual engine. The causal question: **"How many fewer nurses leave when managers use AI-directed check-ins vs. standard management practice?"**

## Domain Grounding

### Geisinger Nursing Workforce
- ~6,800 RN positions (~4,000 inpatient, ~700 outpatient, ~1,950 specialty)
- ~9,415 total nursing positions
- Nurse managers typically manage ~100 nurses each (vendor confirmed)

### Industry Benchmarks
- Annual RN turnover: ~22% national average
- At high span of control (90+ reports), turnover spikes to ~40% vs ~27% (Laudio data)
- New hires (first year) have roughly 2x turnover risk
- ~15% of workforce are early-tenure at any given time

### Vendor Claims (Laudio/AONL Findings)
- Operational signals predict turnover risk: skipped breaks, no PTO, working late, pattern changes
- Targeted check-ins at milestones (30, 45, 60 days, etc.) improve retention
- Retention improvement: +6pp early, +13pp later in first year
- Prediction without manager capacity to act doesn't work — capacity is the binding constraint

## Scenario Contract (5-Method Mapping)

### Unit of Analysis
`nurse` — each entity is one nurse tracked over the simulation period.

### State Representation
Dataclass with per-nurse numpy arrays:

| Field | Type | Description |
|---|---|---|
| `base_risk` | float array | Annual turnover probability (beta-distributed, right-skewed) |
| `ar1_modifier` | float array | Temporal risk drift via AR(1) process (job satisfaction fluctuates) |
| `tenure_weeks` | float array | Weeks employed (starts at hire date, advances each step) |
| `is_new_hire` | bool array | True if tenure < 52 weeks |
| `manager_id` | int array | Which manager this nurse reports to |
| `intervention_effect` | float array | Current risk reduction from last check-in (decays over time) |
| `weeks_since_intervention` | float array | Weeks since last manager check-in |
| `current_risk` | float array | Effective annual turnover probability = base × ar1 × (1 - intervention_effect) |
| `departed` | bool array | Absorbing state — nurse has left |
| `soc_intervention_effect` | float array | Standard-of-care check-in effect (applied in `step()` on both branches) |
| `soc_weeks_since_intervention` | float array | Weeks since last standard-of-care check-in |
| `total_departures` | int | Running count |
| `total_interventions` | int | Running count (AI-directed, factual branch only) |
| `total_soc_interventions` | int | Running count (standard-of-care, both branches) |

### Method 1: `create_population(n_entities)`
- Generate `base_risk` from `beta_distributed_risks()` with `annual_turnover_rate` and `risk_concentration`
- Assign tenure: new hires (fraction = `new_hire_fraction`) get 0–51 weeks, established get 52–260 weeks
- Apply `new_hire_risk_multiplier` (2.0x) to new hire base risk
- Apply `high_span_turnover_penalty` (+0.10) when `nurses_per_manager > 90`
- Round-robin assign `manager_id` (n_managers = n_nurses / nurses_per_manager)
- RNG: `self.rng.population`

### Method 2: `step(state, t)` — MUST BE PURE
1. **AR(1) drift**: `new_mod = rho * old_mod + (1-rho) * 1.0 + noise`, clipped to [0.5, 2.0]
2. **Advance tenure**: +1 week for active nurses; update `is_new_hire` flag at 52-week boundary
3. **Decay both intervention effects**: multiply `intervention_effect` and `soc_intervention_effect` by `0.5^(1/halflife)` each week
4. **Standard-of-care allocation** (runs on BOTH branches — this is the counterfactual intervention):
   - For each manager, allocate K check-ins using the standard heuristic:
     a. Active new hires not in SOC cooldown, random order → fill slots
     b. If slots remain, random selection from active non-cooldown nurses
   - Set `soc_intervention_effect = effectiveness` for selected nurses
   - Reset `soc_weeks_since_intervention = 0`
5. **Recompute current risk**: `base_risk * ar1_modifier * (1 - max(intervention_effect, soc_intervention_effect))`, clipped to [0, 0.99]. Uses the stronger of the two effects (AI-directed or SOC) — they don't stack.
6. **Realize departures**: convert annual risk to weekly probability via hazard function; draw Bernoulli for each active nurse
- RNG: `self.rng.temporal` only
- Purity: reads `self.config` and precomputed decay constant (immutable), nothing else
- Note: Because the SOC allocation runs in `step()`, it executes on both branches identically. On the factual branch, `intervene()` then potentially *upgrades* some nurses' protection via AI-directed targeting (overwriting `intervention_effect` for those nurses). On the counterfactual branch, `intervene()` never runs, so nurses only get the SOC check-ins.

### Method 3: `predict(state, t)`
- Use `ControlledMLModel` in `"discrimination"` mode with `target_auc` from config
- Risk signal = `state.current_risk` for active nurses
- Generate true labels from hazard-based probability over the prediction interval window
- Fit model on first call (3 iterations)
- Return scores for all nurses (departed get score=0)
- Threshold scores into binary labels at `risk_threshold`
- RNG: `self.rng.prediction`

### Method 4: `intervene(state, predictions, t)`

This method implements **two allocation strategies** — one for the factual branch (AI-directed) and one that is also applied on the counterfactual branch (standard of care). Both branches spend the **same total manager time**. The only difference is *who gets checked in on*.

**Factual branch (AI-directed):**
- For each manager:
  1. Get their flagged nurses (score >= threshold) who are active and not in cooldown
  2. Sort by risk score descending
  3. Take top K where K = `max_interventions_per_manager_per_week`
  4. Set `intervention_effect = effectiveness` and reset `weeks_since_intervention = 0`
- Recompute `current_risk` post-intervention

**Counterfactual branch (standard of care):**
- For each manager, same K check-ins per week, but allocated differently:
  1. **New hires first**: prioritize active nurses with `is_new_hire == True` who are not in cooldown, no particular risk ordering (random among new hires)
  2. **Random fill**: if fewer than K new hires are available, fill remaining slots with random selection from active non-cooldown nurses
- Same `intervention_effect` and `intervention_decay_halflife_weeks` apply — the check-in itself is identical, only the targeting differs

This is the critical design choice: the counterfactual is NOT "managers do nothing." It's "managers do the same amount of work, but use their existing heuristic (new hires first, then whoever) instead of the AI's prioritization." This isolates the value of the *model's targeting* from the value of *manager engagement itself*.

**Implementation note for the SDK**: The standard-of-care intervention must happen on the counterfactual branch. The SDK's default causal question is "what if we hadn't deployed the AI?" — which normally means no `predict()` or `intervene()` on the counterfactual branch. This scenario needs to override that behavior. Two implementation options:

1. **Encode standard-of-care in `step()`**: Since `step()` runs on both branches, add the standard-of-care allocation logic there (using `self.rng.temporal`). On the factual branch, `intervene()` then *overrides* those allocations with AI-directed ones. This keeps the engine contract intact but makes `step()` do double duty.

2. **Use a two-scenario comparison**: Run the engine twice — once with AI-directed `intervene()`, once with a `StandardCareScenario` that implements the new-hires-first logic in its own `intervene()`. Compare the two factual branches. Cleaner separation but requires two full runs per sweep cell.

Option 1 is recommended — it keeps the counterfactual comparison within a single engine run and ensures identical population seeds and temporal evolution. The standard-of-care logic in `step()` is still pure (uses only state + `self.rng.temporal`).

- RNG: `self.rng.intervention` (factual), `self.rng.temporal` (counterfactual standard-of-care in step)

### Method 5: `measure(state, t)`
- Events: `departed` array (cumulative — for delta, compare consecutive timesteps)
- Secondary: `active_mask`, `current_risk` arrays
- Metadata: `total_departures`, `total_interventions`, `retention_rate`, `n_active`, `n_new_hire_active`, `mean_risk`, `median_risk`
- RNG: `self.rng.outcomes`

## Configuration Parameters

```python
@dataclass
class RetentionConfig:
    # Population
    n_nurses: int = 1000              # Scale to 6800 for full Geisinger
    nurses_per_manager: int = 100     # Laudio: ~100 is typical
    annual_turnover_rate: float = 0.22
    risk_concentration: float = 0.5   # Beta dist shape (lower = more heterogeneous)
    new_hire_fraction: float = 0.15
    new_hire_risk_multiplier: float = 2.0
    high_span_turnover_penalty: float = 0.10  # Added when nurses_per_manager > 90

    # Temporal
    n_weeks: int = 52                 # 1-year simulation
    ar1_rho: float = 0.95            # Risk autocorrelation
    ar1_sigma: float = 0.04          # Risk noise
    prediction_interval: int = 2      # Score nurses every 2 weeks

    # Model
    model_auc: float = 0.80          # Sweep target

    # Intervention
    max_interventions_per_manager_per_week: int = 4   # Sweep target
    intervention_effectiveness: float = 0.50           # Risk reduction per check-in
    intervention_decay_halflife_weeks: float = 6.0     # Effect wears off
    cooldown_weeks: int = 4                            # Min gap between re-checks
```

## Experiment Sweep Design

### Primary Sweep: Model Quality × Manager Capacity

This is the core experiment. It answers: **given a model of quality X, what capacity level maximizes prevented departures?**

| Dimension | Values | Rationale |
|---|---|---|
| `model.auc` | 0.60, 0.70, 0.80, 0.85 | Range from barely-better-than-chance to strong. Laudio doesn't publish their AUC — this brackets the plausible range. |
| `policy.max_interventions_per_week` | 2, 4, 6, 8 | Per manager per week. At 100 nurses each, 4/week means the manager can reach ~16% of their team per month. |

**Grid size**: 4 × 4 = **16 cells** + 1 no-AI control = 17 runs

**No separate threshold dimension.** The manager's weekly capacity IS the threshold. Each manager is simply given their top-K riskiest nurses to check in with this week — where K is the capacity. Adding an absolute-score threshold on top would be dominated by capacity: if capacity allows 4 check-ins and any reasonable threshold yields at least 4 flagged candidates, the top-4 by score are the same nurses regardless of where the threshold is set. The threshold filter would only matter in the degenerate case of "fewer flagged candidates than capacity slots," which doesn't occur in realistic configurations.

### Control Conditions

Every run produces a built-in comparison: factual (AI-directed) vs. counterfactual (standard of care, same capacity). Additionally:

- **No-intervention baseline**: `capacity = 0` on both branches → pure natural turnover, no check-ins at all. Anchors how much *any* manager engagement helps.
- **AI-directed**: The sweep runs. The delta between factual and counterfactual in each cell isolates the value of AI targeting *over and above* standard management practice.

### Secondary Sweep: Span of Control (if primary results are interesting)

| Dimension | Values |
|---|---|
| `nurses_per_manager` | 50, 75, 100 |
| `model.auc` | 0.70, 0.80 |
| `capacity` | 4, 6 |

**Grid size**: 3 × 2 × 2 = 12 cells — tests the Laudio finding that span of control matters more than prediction quality.

### Hydra CLI Examples

```bash
# Single run (default params)
python scenarios/nurse_retention/run_evaluation.py

# Override for a quick test
python scenarios/nurse_retention/run_evaluation.py \
    --model-auc 0.75 --capacity 6

# Full primary sweep
python scenarios/nurse_retention/run_evaluation.py --sweep

# Hydra multirun (alternative)
python scenarios/nurse_retention/run_evaluation.py --multirun \
    model.auc=0.60,0.70,0.80,0.85 \
    policy.max_interventions_per_week=2,4,6,8
```

## Key Metrics (per run)

| Metric | Definition |
|---|---|
| `factual_departures` | Total nurses who left (AI-directed check-ins) |
| `counterfactual_departures` | Total nurses who left (standard of care — new hires first, then random) |
| `departures_prevented` | counterfactual − factual (value of AI targeting over standard practice) |
| `prevention_rate` | departures_prevented / counterfactual_departures |
| `total_interventions` | Total manager check-ins conducted |
| `interventions_per_prevented` | Cost: how many check-ins per prevented departure |
| `avg_flagged_per_round` | Average nurses flagged per prediction cycle |
| `factual_retention_rate` | Fraction still employed at week 52 (with AI) |
| `counterfactual_retention_rate` | Fraction still employed at week 52 (no AI) |
| `realized_auc` | Actual model discrimination achieved in simulation |

## Key Assumptions to Surface

1. **Intervention = state change, not information.** We model the check-in as directly reducing turnover risk. In reality, the causal chain is: flag → manager checks in → nurse feels supported → decides to stay. We collapse this to a multiplicative risk reduction.

2. **Equal manager effort on both branches.** The factual and counterfactual branches consume the same manager capacity (K check-ins/week). The AI doesn't create more time — it redirects existing time. This is the fair comparison.

3. **Standard of care is new-hires-first, then random.** This reflects real-world management heuristics: new hires get onboarding check-ins, and beyond that managers check in with whoever they happen to see. The AI's value is in identifying *established* nurses who are silently at risk.

4. **Effect decays.** A single check-in doesn't permanently retain someone. Half-life of 6 weeks means after 12 weeks the effect is ~25% of initial.

5. **Effects don't stack, best-of applies.** If a nurse gets both SOC and AI-directed check-ins, the stronger effect wins. This prevents double-counting.

6. **No replacement hires.** Population shrinks as nurses depart. This is conservative — in reality, new hires backfill and are themselves high-risk.

7. **Manager capacity is hard-capped.** In reality, managers might stretch to do one extra check-in. We model a strict ceiling.

8. **Risk signals are continuous, tiers are a UX layer.** The model produces continuous scores; "low/medium/high" tiers are just threshold cuts. The sweep over threshold values implicitly tests different tier configurations.

9. **No demographic subgroups (v1).** This version doesn't model differential turnover by race, shift, unit, or tenure cohort beyond the new-hire flag. An equity sweep is a natural v2 extension.

10. **Independence.** Nurse departures are independent conditional on risk. In reality, departures cluster (one person leaving triggers others). This could be added as a contagion mechanism.

## What We're Looking For

1. **AI targeting vs. standard practice**: How much does AI-directed allocation improve over new-hires-first + random? If the delta is small, the model isn't adding value — just doing check-ins is enough.

2. **Where the model earns its keep**: The AI's value should be highest when it identifies *established* nurses who are silently at risk — people the standard heuristic would miss. Look for the factual branch saving nurses with tenure > 52 weeks that the counterfactual branch loses.

3. **Threshold–capacity interaction**: At what point does increasing model quality stop mattering because manager capacity is the bottleneck? (Laudio's core claim)

4. **Diminishing returns on AUC**: Is there a meaningful difference between AUC=0.70 and AUC=0.85 when managers can only reach 4 nurses/week?

5. **Efficiency frontier**: Since both branches do the same number of total check-ins, efficiency here means: which configuration *redirects* the most check-ins toward nurses who would otherwise leave?

6. **Span-of-control effect**: Does going from 100→50 nurses per manager matter more than going from AUC 0.70→0.85?

7. **Boundary behavior**: AUC=0.50 → AI targeting is random → factual ≈ counterfactual (no delta). This is the key sanity check — when the model is useless, it should match standard of care.
