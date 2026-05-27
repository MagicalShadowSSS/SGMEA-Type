# PHASE16 LTEA Method Writing Framework

Date: 2026-05-21

This file records the current Method-section plan for the paper draft. It will serve as the editable working log for LTEA's technical narrative. Later decisions about formulas, module names, training details, or final paper wording should be updated here first.

## Current Method Name

Tentative method name:

- **LTEA**: LLM-guided Type-aware Entity Alignment

Recommended full paper title wording:

- **LTEA: LLM-guided Type-conditioned Evidence Routing for Multimodal Entity Alignment**

Core narrative:

- LTEA does not focus on generic type-adaptive fusion.
- LTEA formulates MMEA as **type-conditioned multimodal evidence calibration and routing**.
- The main technical flow is:

```text
MMKG entity information
        ↓
LLM-guided Type Construction
        ↓
Type-grounded Visual Evidence Calibration
        ↓
Type-conditioned Multimodal Evidence Routing
        ↓
Entity Alignment
```

Key anti-overlap wording with HSP:

> LTEA does not merely learn type-specific fusion weights over representations; instead, it first calibrates unreliable visual evidence and then performs type-conditioned routing over modality-specific alignment evidence.

## Method Section Structure

The Method section is currently planned as five subsections:

1. Problem Formulation
2. LLM-guided Type Construction
3. Type-grounded Visual Evidence Calibration
4. Type-conditioned Multimodal Evidence Routing
5. Training and Inference

If page budget becomes tight, Section 1 and Section 5 can be shortened. Sections 2-4 are the core technical content.

## 1. Problem Formulation

Purpose:

- Define the MMEA task and notation.
- Avoid emphasizing any specific baseline name.
- Present LTEA as a general framework built on top of a multimodal encoder.

Content to include:

- Given two MMKGs:

```latex
\mathcal{G}_1 = (\mathcal{E}_1, \mathcal{R}_1, \mathcal{A}_1, \mathcal{V}_1),
\quad
\mathcal{G}_2 = (\mathcal{E}_2, \mathcal{R}_2, \mathcal{A}_2, \mathcal{V}_2)
```

- Training seed alignment pairs:

```latex
\mathcal{S} = \{(e_i, e_j) \mid e_i \in \mathcal{E}_1, e_j \in \mathcal{E}_2\}
```

- Each entity has modality-specific representations:

```latex
\mathbf{z}_e^m, \quad m \in \mathcal{M} = \{\mathrm{img}, \mathrm{attr}, \mathrm{rel}, \mathrm{str}\}
```

- The goal is to learn an alignment scoring function:

```latex
s(e_i, e_j)
```

such that equivalent entities receive higher scores than non-equivalent candidates.

Writing note:

- Use neutral wording such as "we adopt a multimodal encoder to obtain modality-specific representations".
- Do not say the method is simply modifying a baseline fusion module.

## 2. LLM-guided Type Construction

Purpose:

- Explain where type information comes from.
- Make clear that the LLM is used to construct entity type semantics, not to directly rerank candidate pairs.

Inputs:

- Entity name.
- Attributes.
- Relation context.
- Neighbor/contextual descriptions when available.

Output:

- A coarse-grained entity type:

```latex
t_e = \mathrm{LLM}(n_e, A_e, C_e)
```

- A learnable or encoded type representation:

```latex
\mathbf{u}_e = \mathrm{Emb}(t_e)
```

Key writing points:

- The type labels are coarse-grained and stable, e.g., Person, Place, Organization, Creative Work.
- The LLM converts implicit, missing, and heterogeneous type cues in MMKGs into explicit type semantics.
- Type semantics are then used as conditions for visual evidence calibration and multimodal evidence routing.
- The LLM is not used for pairwise semantic reranking in the final method narrative.

Possible wording:

> We use the LLM only as a type constructor rather than a pairwise alignment oracle, ensuring that the alignment decision is still made by the proposed multimodal evidence model.

## 3. Type-grounded Visual Evidence Calibration

Purpose:

- Present TCMS as visual evidence calibration rather than image completion or simple visual down-weighting.
- Establish the key motivation: modality availability is different from modality validity.

Core claim:

> A visual modality can be present but invalid: it may be missing, weakly informative, noisy, or inconsistent with the entity type and non-visual context.

Inputs:

- Visual representation:

```latex
\mathbf{z}_e^{\mathrm{img}}
```

- Non-visual representations:

```latex
\mathbf{z}_e^{\mathrm{attr}},
\mathbf{z}_e^{\mathrm{rel}},
\mathbf{z}_e^{\mathrm{str}}
```

- Type representation:

```latex
\mathbf{u}_e
```

Planned mechanism:

1. Construct a non-visual semantic anchor:

```latex
\mathbf{a}_e = f_a([\mathbf{u}_e; \mathbf{z}_e^{\mathrm{attr}}; \mathbf{z}_e^{\mathrm{rel}}; \mathbf{z}_e^{\mathrm{str}}])
```

2. Estimate a visual calibration gate:

```latex
g_e = \sigma(f_g([\mathbf{u}_e; \mathbf{z}_e^{\mathrm{img}}; \mathbf{a}_e]))
```

3. Generate a conservative visual residual:

```latex
\mathbf{r}_e = f_r([\mathbf{z}_e^{\mathrm{img}}; \mathbf{a}_e; \mathbf{u}_e])
```

4. Calibrate visual evidence before fusion:

```latex
\tilde{\mathbf{z}}_e^{\mathrm{img}}
=
\mathbf{z}_e^{\mathrm{img}} + \beta g_e \mathbf{r}_e
```

Interpretation:

- `g_e` measures how strongly the visual evidence should be calibrated.
- `\beta` is a conservative residual strength.
- The module refines visual evidence instead of fully replacing or suppressing it.
- This preserves complementary visual information while reducing noisy or type-inconsistent visual signals.

Current experimental note:

- The fixed protocol currently uses `--tcms_missing_mode keep`.
- Therefore, the main paper story should emphasize **visual evidence calibration / sanitization**, not missing image completion as the core contribution.

Probe / explainability signals to mention if needed:

- Calibration gate mean / max.
- Visual residual norm.
- Missing image rate.
- Case studies of high-gate entities.

## 4. Type-conditioned Multimodal Evidence Routing

Purpose:

- Present DEHR as the main performance module.
- Avoid describing it as ordinary type-aware fusion.
- Emphasize score-level or evidence-level routing instead of representation-level weighted fusion.

Core idea:

> Different entity types rely on different modality-specific alignment evidence. LTEA routes image, attribute, relation, and structural evidence according to the LLM-derived entity type.

Modality-specific evidence score:

```latex
s_m(e_i, e_j) = \cos(\mathbf{z}_{e_i}^m, \mathbf{z}_{e_j}^m),
\quad m \in \mathcal{M}
```

For image evidence, use calibrated visual representations:

```latex
s_{\mathrm{img}}(e_i, e_j)
=
\cos(\tilde{\mathbf{z}}_{e_i}^{\mathrm{img}}, \tilde{\mathbf{z}}_{e_j}^{\mathrm{img}})
```

Type-conditioned routing vector:

```latex
\mathbf{r}_t = \mathrm{softmax}(f_\theta(\mathbf{u}_t))
```

Optional form with type bias:

```latex
\mathbf{r}_t = \mathrm{softmax}(\mathbf{b}_t + f_\theta(\mathbf{u}_t))
```

Final evidence score:

```latex
s(e_i, e_j)
=
\sum_{m \in \mathcal{M}} r_{t,m} \cdot s_m(e_i, e_j)
```

If both source and target types are used:

```latex
\mathbf{r}_{ij} = \frac{1}{2}(\mathbf{r}_{t_i} + \mathbf{r}_{t_j})
```

Then:

```latex
s(e_i, e_j)
=
\sum_{m \in \mathcal{M}} r_{ij,m} \cdot s_m(e_i, e_j)
```

Key writing points:

- DEHR is the major source of performance improvement in the fixed experiments.
- It should be described as **type-conditioned evidence routing**, not merely modality weighting.
- The routing operates on modality-specific alignment evidence.
- TCMS improves the visual evidence before it enters the routing process.

Anti-overlap wording:

> Unlike representation-level adaptive fusion, LTEA performs type-conditioned routing over modality-specific alignment evidence after calibrating unreliable visual evidence.

## 5. Training and Inference

Purpose:

- Explain the optimization objective without overloading the paper with too many loss terms.
- Clarify that no LLM pairwise reranking is used in the final LTEA method.

Main alignment objective:

```latex
\mathcal{L}_{EA}
```

This can be written as the standard entity alignment ranking / contrastive objective used by the multimodal encoder.

Overall objective, if needed:

```latex
\mathcal{L}
=
\mathcal{L}_{EA}
+
\lambda_{cal}\mathcal{L}_{cal}
+
\lambda_{reg}\mathcal{L}_{reg}
```

Writing caution:

- Keep auxiliary losses lightweight.
- Avoid presenting a large multi-task loss as the central innovation.
- Emphasize stable calibration and routing rather than a complicated loss cocktail.

Inference:

- Entity types are already constructed before inference.
- Visual evidence is calibrated by TCMS.
- Modality-specific evidence scores are routed by DEHR.
- Final ranking is produced by the LTEA evidence score.
- No LLM-based pairwise reranking is included in the final method narrative.

## Current Experimental Conclusions To Align With Method

Use the fixed record:

- `log/phase_logs/PHASE15_FIXED_CKPT_TCMS_DEHR_0520.md`
- `log/phase_logs/phase15_fixed_ckpt_tcms_dehr_0520.json`

Fixed conclusions:

- TCMS alone average Hits@1 gain: +1.477 pp.
- DEHR alone average Hits@1 gain: +3.608 pp.
- TCMS + DEHR average Hits@1 gain: +4.491 pp.
- DEHR is the main performance contributor.
- TCMS is complementary and gives additional improvement when combined with DEHR.
- TCMS is not uniformly positive; FBDB r=0.2 drops when TCMS is used alone.

Method implication:

- Do not claim TCMS is the sole or dominant performance source.
- Present DEHR as the central evidence routing component.
- Present TCMS as a visual evidence quality module that strengthens routing by improving input evidence.

## Planned Contribution Mapping

Contribution 1:

- LLM-guided type construction and LTEA framework.
- Type information is used as an explicit condition for evidence calibration and routing.

Contribution 2:

- Type-grounded visual evidence calibration.
- Distinguishes modality availability from modality validity.
- Refines unreliable visual evidence before final evidence routing.

Contribution 3:

- Type-conditioned multimodal evidence routing.
- Routes image, attribute, relation, and structure evidence according to entity type.
- Demonstrates robust gains across FBDB and FBYG under multiple training ratios.

## Open Decisions

These items still need to be finalized later:

- Whether final formulas should use source-side type only or average source/target type routing.
- Whether to explicitly describe routing as score-level in the final paper if the implementation is partially fusion-level.
- How much detail to include about TCMS gate, residual, and type memory.
- Whether to include type memory in the main method or move it to implementation details.
- Final abbreviation expansion for LTEA.
- Final figure layout and module names.

