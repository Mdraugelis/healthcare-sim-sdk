"""Paper 09: Lång et al. — MASAI Trial.

Lancet Oncology 2023. Swedish mammography screening. 80,033 women.
Transpara AI (ScreenPoint Medical) triage of screening mammograms.

Primary outcomes:
  - Cancer detection: 6.7 vs 5.0 per 1,000 screens (AI+1 reader vs. 2 readers)
  - Radiologist workload: 44% reduction in reads
  - Non-inferiority RCT for false positive rate

Unit of analysis: screening mammogram (woman)
State: 2D array [n_women x 4]
  col 0: true_cancer_risk (0-1 probability of true malignancy)
  col 1: transpara_score (0-10 AI score, discretized to 0-1)
  col 2: reading_pathway (0=single AI-assisted, 1=standard double)
  col 3: detected (0/1 cumulative)

Key parameters (paper-derived):
  - base_cancer_rate = 0.005 (5.0/1000, standard screening rate in control)
  - ai_detection_rate = 0.0067 (6.7/1000, AI arm)
  - ai_sensitivity_boost = 0.34 (ratio: 6.7/5.0 = 1.34, 34% more cancers)
  - auc_transpara = ~0.83 (typical reported; not in this abstract)
  - radiologist_reads_reduction = 0.44 (44% fewer reads)

Modeling note:
  The MASAI trial uses AI to TRIAGE cases to single vs. double reading,
  not to replace readers. High-AI-score cases get additional attention.
  Low-AI-score cases get single reading (workload reduction).
  Net effect: more cancers detected at same or lower false positive rate.

RNG DISCIPLINE:
- create_population() -> self.rng.population
- step()              -> self.rng.temporal
- predict()           -> self.rng.prediction
- intervene()         -> self.rng.intervention
- measure()           -> self.rng.outcomes
"""

from typing import Optional
import numpy as np

from healthcare_sim_sdk.core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)
from healthcare_sim_sdk.ml.model import ControlledMLModel

# Column indices
COL_CANCER_RISK = 0
COL_TRANSPARA = 1
COL_PATHWAY = 2
COL_DETECTED = 3

# Age distribution for Swedish mammography screening (40-74 years)
AGE_DIST = {
    "40-49": 0.25,
    "50-59": 0.30,
    "60-69": 0.30,
    "70-74": 0.15,
}
# Cancer prevalence multiplier by age (screening population)
AGE_CANCER_MULT = {
    "40-49": 0.70,  # lower incidence in younger women
    "50-59": 0.95,
    "60-69": 1.20,
    "70-74": 1.30,
}


class MASAIScenario(BaseScenario[np.ndarray]):
    """MASAI: AI-supported mammography screening triage simulation.

    Intervention: Transpara AI assigns a 1-10 score. Low-risk women
    (score ≤ 7) get single reader review. High-risk women (score > 7)
    get double reading + radiologist attention. In the control arm,
    all women get standard double reading.

    AI detection enhancement modeled as:
      - For true positives: AI triages high-risk to double-read pathway
        where second reader has higher detection sensitivity
      - For all: AI flags high-suspicion cases for recall consideration

    Parameters:
      - base_cancer_rate = 0.005 (5/1000 control arm, paper-derived)
      - ai_detection_rate = 0.0067 (6.7/1000 AI arm, paper-derived)
      - transpara_auc = 0.83 (ASSUMED; paper doesn't report in abstract)
      - single_reader_sensitivity = 0.70 (ASSUMED)
      - double_reader_sensitivity = 0.83 (ASSUMED; standard consensus)
      - ai_single_reader_sensitivity = 0.90 (ASSUMED; AI-augmented)
      - high_risk_threshold = 0.70 (score >7 on 1-10 scale = 0.70)
    """

    unit_of_analysis = "screening_mammogram"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        base_cancer_rate: float = 0.005,  # paper-derived: 5.0/1000
        transpara_auc: float = 0.83,  # ASSUMED
        single_reader_sensitivity: float = 0.70,  # ASSUMED
        double_reader_sensitivity: float = 0.83,  # ASSUMED
        ai_single_reader_sensitivity: float = 0.90,  # ASSUMED
        high_risk_threshold: float = 0.70,  # AI score >7/10
        workload_reduction: float = 0.44,  # paper-derived
    ):
        super().__init__(time_config=time_config, seed=seed)
        self.base_cancer_rate = base_cancer_rate
        self.transpara_auc = transpara_auc
        self.single_reader_sensitivity = single_reader_sensitivity
        self.double_reader_sensitivity = double_reader_sensitivity
        self.ai_single_reader_sensitivity = ai_single_reader_sensitivity
        self.high_risk_threshold = high_risk_threshold
        self.workload_reduction = workload_reduction

        self._model = ControlledMLModel(
            mode="discrimination",
            target_auc=transpara_auc,
        )
        self._model_fitted = False
        self._demographics = None

    def create_population(self, n_entities: int) -> np.ndarray:
        """Create screening mammogram population.

        Returns array (n_entities, 4).
        Base cancer rate: 5/1000 in control arm.
        """
        rng = self.rng.population

        # Most women have very low cancer risk; bimodal distribution
        # ~0.5% are true positives (base_cancer_rate = 0.005 = 5/1000)
        # Generate from mixture: 99.5% low-risk, 0.5% high-risk
        high_risk_n = max(1, int(n_entities * self.base_cancer_rate))
        low_risk_n = n_entities - high_risk_n

        # Low-risk: beta(0.3, 100) -> very low mean ~0.003, concentrated near 0
        low_risk = rng.beta(0.3, 100, low_risk_n)

        # High-risk (true cancers): uniform [0.5, 0.95] to represent confirmed cancer
        # Using uniform rather than beta to ensure all high-risk are above threshold
        high_risk = rng.uniform(0.5, 0.95, high_risk_n)

        all_risks = np.concatenate([low_risk, high_risk])
        rng.shuffle(all_risks)  # randomize order

        # Age assignment
        age_names = list(AGE_DIST.keys())
        age_probs = list(AGE_DIST.values())
        ages = rng.choice(age_names, n_entities, p=age_probs)
        self._demographics = ages

        # Apply age multipliers
        for i in range(n_entities):
            mult = AGE_CANCER_MULT[ages[i]]
            all_risks[i] = np.clip(all_risks[i] * mult, 0.0001, 0.99)

        state = np.zeros((n_entities, 4))
        state[:, COL_CANCER_RISK] = all_risks
        return state

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        """Screening batch: minimal temporal evolution.

        Each timestep represents a batch of screenings. Population
        is refreshed slightly (new women entering screening age).
        """
        rng = self.rng.temporal
        # Minimal drift — screening population is essentially static
        # within study period (~1-2 years)
        noise = rng.normal(0, 0.0001, state.shape[0])
        new_state = state.copy()
        new_state[:, COL_CANCER_RISK] = np.clip(
            state[:, COL_CANCER_RISK] + noise, 0.0001, 0.99
        )
        return new_state

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        """Transpara AI assigns cancer suspicion score (mapped to 0-1)."""
        true_risks = state[:, COL_CANCER_RISK]
        n = len(true_risks)

        if not self._model_fitted:
            true_labels = (
                self.rng.prediction.random(n) < true_risks
            ).astype(int)
            self._model.fit(
                true_labels, true_risks,
                self.rng.prediction, n_iterations=5,
            )
            self._model_fitted = True

        scores = self._model.predict(true_risks, self.rng.prediction)
        return Predictions(
            scores=scores,
            metadata={"true_risks": true_risks.copy()},
        )

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        """AI triage: route high-risk to double reading, low-risk to single.

        In factual (AI) arm:
          - Score > threshold: double reading (high sensitivity)
          - Score <= threshold: single AI-assisted reading (reduced workload)

        Detection enhancement for high-score cases:
          Reading pathway encoded in state col 2.
        """
        scores = predictions.scores
        high_risk_mask = scores >= self.high_risk_threshold
        treated_indices = np.where(high_risk_mask)[0]
        low_risk_indices = np.where(~high_risk_mask)[0]

        new_state = state.copy()
        # High-risk: double reading pathway (1)
        new_state[treated_indices, COL_PATHWAY] = 1.0
        # Low-risk: single AI-assisted reading (0)
        new_state[low_risk_indices, COL_PATHWAY] = 0.0

        n_high = int(high_risk_mask.sum())
        n_low = len(low_risk_indices)

        return new_state, Interventions(
            treated_indices=treated_indices,
            metadata={
                "n_high_risk": n_high,
                "n_low_risk": n_low,
                "pct_high_risk": float(high_risk_mask.mean()),
                "workload_reads": n_high * 2 + n_low * 1,  # factual reads
                "control_reads": len(scores) * 2,  # counterfactual reads
            },
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        """Detect cancers based on pathway and true risk.

        Detection probabilities:
          Factual (AI) arm:
            - High AI score -> double reading:
              sensitivity = double_reader_sensitivity
            - Low AI score -> single AI-augmented reading:
              sensitivity = ai_single_reader_sensitivity (slightly higher
              than unaided single read due to AI attention)

          Counterfactual (control) arm:
            - All cases get standard double reading:
              sensitivity = double_reader_sensitivity

          Note: step() is called identically for both branches.
          The factual vs. counterfactual difference is in COL_PATHWAY.
          Counterfactual branch always has COL_PATHWAY=0 (no pathway
          modification, so we set all as 1 = double reading default).
        """
        rng = self.rng.outcomes
        n = state.shape[0]
        true_risks = state[:, COL_CANCER_RISK]
        pathway = state[:, COL_PATHWAY]

        # Determine sensitivity per patient based on pathway
        # pathway=1: double reading (standard or AI-directed double)
        # pathway=0: single AI-assisted reading
        sensitivity = np.where(
            pathway == 1.0,
            self.double_reader_sensitivity,
            self.ai_single_reader_sensitivity,
        )

        # Counterfactual branch: pathway is always 0 (unmodified by intervene)
        # We can't distinguish factual/counterfactual in measure() directly.
        # The engine handles this: counterfactual branch never calls intervene(),
        # so pathway stays 0 for all. Default pathway=0 in counterfactual
        # gets ai_single_reader_sensitivity — but that's wrong for control.
        # FIX: Use the FACT that counterfactual state has pathway=0 uniformly.
        # In counterfactual: we want double_reader_sensitivity for all.
        # We detect this as: if all pathways are 0, use double_reader_sensitivity.
        if pathway.mean() == 0.0:
            # Counterfactual branch: standard double reading for all
            sensitivity = np.full(n, self.double_reader_sensitivity)

        # Sample detection events
        # P(detected) = P(true_cancer) * sensitivity
        true_cancer = (rng.random(n) < true_risks).astype(float)
        detected = (rng.random(n) < sensitivity).astype(float) * true_cancer

        # Secondary: reads performed
        reads_per_patient = np.where(pathway == 1.0, 2.0, 1.0)
        if pathway.mean() == 0.0:
            reads_per_patient = np.full(n, 2.0)  # counterfactual: all double

        age = (
            self._demographics if self._demographics is not None
            else np.array(["Unknown"] * n)
        )

        return Outcomes(
            events=detected,
            entity_ids=np.arange(n),
            secondary={
                "true_cancer": true_cancer,
                "reads_per_patient": reads_per_patient,
                "age_group": age,
                "pathway": pathway.copy(),
            },
            metadata={
                "detection_rate": float(detected.mean()),
                "detection_per_1000": float(detected.mean() * 1000),
                "total_reads": float(reads_per_patient.sum()),
                "mean_reads_per_patient": float(reads_per_patient.mean()),
            },
        )

    def clone_state(self, state: np.ndarray) -> np.ndarray:
        return state.copy()
