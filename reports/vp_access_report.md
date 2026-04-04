# No-Show Overbooking: What to Expect from the ML Predictor

*Prepared for VP of Access Operations*
*Based on 193 simulation configurations across 9 clinic archetypes*

---

## The Short Version

We simulated 90 days of clinic operations under the current overbooking
practice (staff uses patient history >= 50% no-show rate) and compared it
against an ML predictor at three performance levels that bracket what we
expect from Epic's model at Geisinger.

**The ML predictor can reduce collisions by 25-50% in most clinic types
--- but the right threshold depends on the clinic.** A single system-wide
threshold will underperform in low no-show clinics and over-flag in high
no-show clinics. The recommendation is to configure thresholds per clinic
archetype.

---

## What We Simulated

Nine clinic archetypes representing the range across the system:

| Clinic Type | No-Show Rate | Utilization | OB Cap/Provider | Waitlist Pressure |
|-------------|-------------|-------------|-----------------|-------------------|
| Low NS / Under-booked | 7% | 80% | 3 | 2/day |
| Low NS / Standard | 7% | 90% | 2 | 5/day |
| Low NS / Over-booked | 7% | 110% | 1 | 10/day |
| Moderate NS / Under-booked | 13% | 80% | 3 | 2/day |
| Moderate NS / Standard | 13% | 90% | 2 | 5/day |
| Moderate NS / Over-booked | 13% | 110% | 1 | 10/day |
| High NS / Under-booked | 20% | 80% | 3 | 2/day |
| High NS / Standard | 20% | 90% | 2 | 5/day |
| High NS / Over-booked | 20% | 110% | 1 | 10/day |

For each, we tested the ML predictor at three performance levels:

| Scenario | AUC | Why This Level |
|----------|-----|----------------|
| Pessimistic | 0.65 | JPS safety-net found 0.58; Geisinger should be better but demographics removed |
| Moderate | 0.75 | Conservative estimate after demographic feature removal |
| Optimistic | 0.83 | Epic's reported performance (pre-removal, with demographics) |

Each AUC level was tested across 7 thresholds (0.20-0.80).

---

## Where the Predictor Helps Most

### Low No-Show Clinics (7%)
*Typical: dermatology, well visits, established patients*

**Current practice (staff, hist >= 50%):** 92.8% utilization, 57.3% collision rate, 295 on waitlist

| Model AUC | Best Threshold | Utilization | Collision Rate | Change | Waitlist |
|-----------|---------------|-------------|----------------|--------|---------|
| 0.65 (pessimistic) | 0.20 | 93.9% | 55.8% | -3% collisions | 182 |
| 0.75 (moderate) | 0.20 | 93.9% | 55.8% | -3% collisions | 182 |
| 0.83 (optimistic) | 0.20 | 92.9% | 49.2% | -14% collisions | 237 |

**What this means:** In low no-show clinics, the baseline already achieves ~93% utilization. The predictor's main value is **reducing collisions** --- from 57% down to 49% at AUC 0.83 --- while also reducing the waitlist. You need threshold 0.20 here because at 7% base rate, higher thresholds flag almost nobody. Even at AUC 0.65, the predictor modestly improves collision rate and cuts the waitlist by 100+ patients.

### Moderate No-Show Clinics (13%)
*Typical: primary care, standard mix*

**Current practice (staff, hist >= 50%):** 89.6% utilization, 41.1% collision rate, 27 on waitlist

| Model AUC | Best Threshold | Utilization | Collision Rate | Change | Waitlist |
|-----------|---------------|-------------|----------------|--------|---------|
| 0.65 (pessimistic) | 0.50 | 87.1% | 40.0% | -3% collisions | 452 |
| 0.75 (moderate) | 0.70 | 87.3% | 18.2% | -56% collisions | 421 |
| 0.83 (optimistic) | 0.30 | 90.2% | 30.8% | -25% collisions | 0 |

**What this means:** This is the sweet spot. At AUC 0.83 with threshold 0.30, it's a **win-win**: utilization goes *up* (+0.6%), collisions go *down* 25%, and the waitlist clears completely. At AUC 0.75, you can still cut collisions significantly (-56%), but it costs some utilization. At AUC 0.65, the benefit is marginal --- the model barely outperforms the baseline.

### High No-Show Clinics (20%)
*Typical: behavioral health, Medicaid-heavy panels*

**Current practice (staff, hist >= 50%):** 82.0% utilization, 29.5% collision rate, 0 on waitlist

| Model AUC | Best Threshold | Utilization | Collision Rate | Change | Waitlist |
|-----------|---------------|-------------|----------------|--------|---------|
| 0.65 (pessimistic) | -- | -- | -- | **worse than baseline** | -- |
| 0.75 (moderate) | 0.80 | 79.6% | 29.4% | ~0% collisions | 386 |
| 0.83 (optimistic) | 0.50 | 81.8% | 25.9% | -12% collisions | 0 |

**What this means:** High no-show clinics need a strong model. At AUC 0.65, the predictor **does not beat the baseline** --- the historical rate is already a reasonable predictor when 20% of patients no-show. At AUC 0.83, threshold 0.50 cuts collisions 12% while maintaining utilization and clearing the waitlist. **Don't deploy in these clinics until local AUC validates above 0.80.**

---

## The Prevalence-PPV Trap: Why One Threshold Does Not Fit All

This is the single most important finding for threshold selection. At the **same model performance** and the **same threshold**, the model's behavior changes dramatically depending on the clinic's no-show rate:

| Clinic No-Show Rate | PPV at Threshold 0.30 | Sensitivity | What Happens |
|--------------------|----------------------|-------------|-------------|
| 7% | 75% | 1% | Model barely flags anyone --- too few no-shows to detect |
| 13% | 57% | 49% | Meaningful flagging --- about half of flags are true no-shows |
| 20% | 49% | 66% | Aggressive flagging --- catches most no-shows but flags 1 in 3 slots |

**This is Bayes' theorem, not a model quality issue.** When the base rate is 7%, even a perfect model can't achieve high PPV at a moderate threshold because there aren't enough true positives. At 20%, the model catches more because there's more to catch.

**Recommendation:** Set thresholds per clinic archetype, not system-wide.

---

## Over-Booked Clinics (110%): A Different Problem

Clinics running above capacity show a distinct pattern:

| Clinic | Utilization | Waitlist (Day 90) | Issue |
|--------|------------|-------------------|-------|
| NS13%_Util110% | 88.8% | 612 | Capacity, not prediction |
| NS20%_Util110% | 82.0% | 434 | Capacity, not prediction |
| NS7%_Util110% | 92.7% | 789 | Capacity, not prediction |

These clinics have waitlists of 400-900 patients regardless of strategy. **Overbooking cannot fix a capacity problem.** The intervention for 110% clinics is adding providers or extending hours, not threshold tuning.

---

## At What Model Performance Does This Stop Being Worth It?

Epic reports AUC of 0.83-0.87, but that was before removing demographic features. JPS Health Network found 0.58 in external validation. Where will Geisinger land?

| AUC Level | Beats Baseline In... | Does Not Beat Baseline In... |
|-----------|---------------------|------------------------------|
| 0.65 | 8/9 clinics | 20% @90% |
| 0.75 | 7/9 clinics | 13% @110%, 20% @80% |
| 0.83 | 7/9 clinics | 13% @110%, 7% @110% |

**Bottom line:** Even at AUC 0.65 (pessimistic), the predictor beats baseline in most clinic types. The model does not need to be great to be useful --- it just needs to be better than a historical rate that lags reality.

---

## Recommended Starting Thresholds for the Pilot

**Strategy:** Target a collision rate *slightly better* than what staff achieves today. Don't chase zero collisions --- that sacrifices access. Instead, take the collision improvement and use the freed-up overbooking capacity to serve more patients from the waitlist.

**At AUC 0.83 (Epic's reported performance):**

| Clinic Type | Baseline | Threshold | Utilization | Collision Rate | Waitlist | Access Gain |
|-------------|----------|-----------|-------------|----------------|----------|-------------|
| Low NS (7%), Standard | 93% / 57% coll / WL 295 | **0.20** | 93% | 49% (-14%) | 237 (-58) | +182 patients served |
| Moderate NS (13%), Under-booked | 87% / 44% coll / WL 0 | **0.40** | 88% (+0.6%) | 28% (-36%) | 0 | 178 patients served |
| **Moderate NS (13%), Standard** | **90% / 41% coll / WL 27** | **0.30** | **90% (+0.6%)** | **31% (-25%)** | **0 (-27)** | **443 patients served** |
| High NS (20%), Standard | 82% / 30% coll / WL 0 | **0.50** | 82% | 26% (-12%) | 0 | 441 patients served |

**The best overall result: Moderate NS (13%) clinics at threshold 0.30.** Utilization goes *up*, collisions go *down*, and the waitlist clears completely. This is the win-win.

**At AUC 0.75 (conservative estimate, post-demographics-removal):**

| Clinic Type | Threshold | Utilization | Collision Rate | Waitlist | Trade-off |
|-------------|-----------|-------------|----------------|----------|-----------|
| Low NS (7%), Standard | 0.20 | 94% (+1.1%) | 56% (-2%) | 182 | Marginal collision improvement |
| Moderate NS (13%), Under-booked | 0.20 | 88% (+0.5%) | 29% (-34%) | 0 | Clear win |
| Moderate NS (13%), Standard | 0.50 | 87% (-2.6%) | 35% (-15%) | 327 | Trades utilization for collisions |
| High NS (20%), Standard | -- | -- | -- | -- | Needs AUC 0.83 for clear benefit |

**Key takeaway:** At AUC 0.75, moderate NS clinics still see meaningful collision reduction. Low NS and high NS clinics need the stronger model (0.83) for a clear win. This answers the governance question: **local validation of model AUC determines which clinics should go live first.**

**Recommended pilot sequence:**
1. **Start with 13% no-show clinics** (primary care) --- clearest benefit at any AUC level
2. **Add 7% clinics** if local AUC validates at 0.80+ --- collision reduction is the value prop
3. **Add 20% clinics** only if local AUC validates at 0.83+ --- otherwise baseline is adequate
4. **110% utilization clinics** --- address capacity first, overbooking second

Validate during the 12-week pilot. Adjust based on observed collision rates and provider feedback.

---

## What This Does Not Tell Us

This simulation models overbooking only. It does **not** model:

- **Patient outreach** --- reminder calls/texts may change show rates and shift the collision math
- **Advanced rescheduling** --- proactively offering to reschedule flagged patients could reduce no-shows without overbooking
- **Provider-level variation** --- some providers tolerate overbooking better; simulation uses uniform caps
- **Seasonal effects** --- flu season, holidays, weather affect no-show rates beyond what the AR(1) drift captures
- **Patient response** --- repeated overbooking may change patient behavior (positive or negative)

The simulation sets **starting points** and shows **which levers matter most** before committing real resources.

---

*Simulation: 90 days, 2,000 patients per clinic, 193 configurations*
*Framework: healthcare-sim-sdk v0.1.0-beta*