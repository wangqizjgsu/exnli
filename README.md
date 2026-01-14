# Materials and Methods

This document introduces the EGM-PM framework and ExNLI dataset construction.

## EGM-PM Dual-Model Active Learning Framework

**EGM-PM** is a dual-model active learning framework that comprises three core components:

- **Explanation Generation Model (EGM)**: Takes premise-hypothesis pairs as inputs and generates free-form natural language explanations through fine-tuning
- **Prediction Model (PM)**: Takes both premise-hypothesis pairs and EGM-generated explanations as joint inputs for relation prediction
- **Active Learning Sampler**: Selects the most representative and informative samples from the unlabeled pool for annotation

The framework adopts FLAN-T5 as the base model for both EGM and PM. Through iterative training cycles, the EGM-PM achieves synergistic enhancement of explanation generation and prediction performance, ultimately improving both model performance and interpretability.

## ExNLI Dataset Construction

The **ExNLI (Explainable Natural Language Inference)** dataset is constructed based on XNLI and OCNLI datasets:

- **Balanced Sampling**: Stratified sampling by label from OCNLI to ensure balanced data distribution
- **Multilingual Translation**: Uses NLLB-200 to translate the Chinese subset into English, German, and French
- **Explanation Annotation**: Generates natural language explanations for each premise-hypothesis pair using pre-designed prompt templates
- **Data Splits**: Training/validation sets retain a single explanation, while test sets provide multiple explanations for robustness evaluation

## Experimental Environment

- **GPU**: NVIDIA RTX 3070Ti
- **CUDA**: 12.2
- **Framework**: PyTorch
- Single GPU is sufficient for efficient generation of high-quality natural language explanations

