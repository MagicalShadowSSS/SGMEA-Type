# Related Papers And Borrowed Ideas

## 1. Purpose

This document records the key papers currently relevant to the project's next-stage directions,
and clarifies what exact ideas should be borrowed from each paper.

The goal is not to copy any single method directly.
Instead, this document serves as a map:

- which paper is most relevant to which route
- which part of the method is worth borrowing
- which part does **not** fit the current SGMEA project directly


## 2. Current Project Context

At the current stage:

- the main validated route is `M2DSA plain`
- prototype has already been shown to work best as a retrieval-time macro semantic prior
- the next-stage exploratory direction is `Ontology-Driven Visual Prompting`

Therefore the most useful literature is:

- prompt learning for vision-language models
- visual prompt tuning
- conditional prompt generation
- prototype / retrieval-time adaptation methods


## 3. Core Reference Papers

### 3.1 CoOp

Paper:
- `Learning to Prompt for Vision-Language Models`

Main relevance:
- prompt learning for frozen vision-language models

What to borrow:
- how to treat prompts as learnable parameters
- how to keep the large backbone frozen while optimizing only a tiny prompt module
- the general prompt-learning training protocol

What not to directly copy:
- CoOp is class-prompt learning for recognition tasks
- the current project is not standard closed-set classification

Project value:
- useful as the baseline reference for prompt parameterization


### 3.2 CoCoOp

Paper:
- `Conditional Prompt Learning for Vision-Language Models`

Main relevance:
- instance-conditioned prompts

What to borrow:
- the idea that prompt content does not need to be globally fixed
- prompt generation can depend on an external conditioning signal
- this is very close to the project's prototype-conditioned prompt idea

What not to directly copy:
- the conditioning signal in CoCoOp is not prototype / ontology prior
- the project would replace their instance-condition source with the prototype semantic signal

Project value:
- most directly relevant paper for the idea:
  - `prototype -> prompt`


### 3.3 Visual Prompt Tuning (VPT)

Paper:
- `Visual Prompt Tuning`

Main relevance:
- adding learnable prompt tokens directly into the vision transformer

What to borrow:
- how visual prompt tokens are inserted into ViT
- how to keep the visual backbone frozen
- why small prompt tokens can still alter visual attention

What not to directly copy:
- VPT is not designed for ontology-guided cross-lingual entity alignment
- it does not provide a macro semantic controller by itself

Project value:
- this is the closest technical reference for the project's Phase-1 Micro-VPT oracle probe


### 3.4 MaPLe

Paper:
- `MaPLe: Multi-modal Prompt Learning`

Main relevance:
- joint prompt learning on both vision and language sides

What to borrow:
- prompting both modalities in a coordinated way
- the idea that visual and textual prompting can be coupled rather than completely isolated

What not to directly copy:
- current project still needs to validate whether a simpler visual-side prompt already gives useful signal
- MaPLe should be treated as a second-stage upgrade reference, not the first implementation target

Project value:
- useful if the project later upgrades from simple ontology prompts to dual-side prompt control


### 3.5 Proto-Adapter

Paper:
- prototype-based adapter / retrieval adaptation line

Main relevance:
- use of prototype-like high-level structure at adaptation or retrieval time

What to borrow:
- the idea that prototype information does not have to be injected into the backbone to be useful
- it can instead act at retrieval / adaptation time as a decision prior

What not to directly copy:
- the original task setting is not multimodal KG entity alignment
- the exact architecture is not directly transferable

Project value:
- most helpful for framing `M2DSA plain`
- especially useful for writing the narrative that prototype is best used as a retrieval-time prior


## 4. Route-Specific Borrowing Guide

### 4.1 For `M2DSA plain`

Most relevant paper family:
- retrieval-time adaptation / prototype-adapter style work

Borrowed idea:
- high-level semantic priors can be more useful at decision time than as backbone perturbations

How it maps to the project:
- `E_joint` = micro identity evidence
- `P` = macro semantic prior
- final score = dual-space retrieval decision


### 4.2 For `Ontology-Driven Visual Prompting`

Most relevant paper family:
- `VPT`
- `CoCoOp`
- `MaPLe`

Borrowed idea:
- a frozen vision(-language) model can still be steered by a small prompt module
- prompts can be conditioned on an external signal

How it maps to the project:
- replace generic class-condition with prototype / ontology condition
- use macro semantic prior as the prompt controller
- use prompting to clean visual evidence before multimodal fusion


### 4.3 For Phase-1 Oracle Probe

Most relevant technical intuition:
- `VPT` for visual prompt insertion
- `CoCoOp` for conditional prompting logic

Borrowed idea:
- prompt tokens can change attention without full backbone retraining

How it maps to the project:
- test whether ontology-conditioned prompting changes same-vs-cross visual margins
- before building a full prompt pipeline


## 5. Recommended Reading Order

If the project wants to continue along the raw-image route, the recommended reading order is:

1. `VPT`
   - understand the mechanics of visual prompt insertion
2. `CoCoOp`
   - understand conditional prompt generation
3. `MaPLe`
   - understand possible multimodal prompt coupling for future upgrades
4. prototype / retrieval adaptation papers
   - mainly to strengthen the narrative around `M2DSA plain`


## 6. Final Practical Summary

The current best alignment between papers and project directions is:

- `M2DSA plain`
  - borrow the **prototype-as-retrieval-prior** intuition from adapter / retrieval adaptation work

- `ODVP / Micro-VPT`
  - borrow the **visual prompt insertion** mechanism from `VPT`
  - borrow the **conditioned prompt generation** idea from `CoCoOp`
  - treat `MaPLe` as an upgrade reference rather than the first implementation target

This is the most useful and realistic borrowing strategy for the current project stage.
