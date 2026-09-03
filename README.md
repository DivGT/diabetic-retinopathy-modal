# Multimodal & Explainable AI for Early Diabetic Retinopathy Detection

[![Research Status](https://img.shields.io/badge/Status-Active_Development_%2F_Pipeline_Migration-blue.svg)](https://github.com/DivGT)
[![Internship](https://img.shields.io/badge/IIIT_Ranchi-Research_Internship-orange.svg)](https://iiitranchi.ac.in)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers_%26_VLMs-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **Research Project Credential:** `IIITR/CSE/BS/2026/13`  
> **Affiliation:** Indian Institute of Information Technology (IIIT), Ranchi  
> **Author:** Divyanshu Gupta ([Portfolio](https://my-portfolio-divyanshu.netlify.app) | [LinkedIn](https://www.linkedin.com/in/divyanshu-gupta13592/))

---

##  Notice
> **Repository Status:** The core code modules, fine-tuned LoRA weights, and Gradio inference pipelines are actively being migrated into this repository following local workspace reconstruction. Checkpoints and interactive demos will be published to Hugging Face Hub shortly.

---

##  Project Overview
Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness worldwide. While automated classification algorithms exist, clinical adoption remains low due to the **"black box"** nature of standard deep convolutional models.

This project delivers an **Explainable Multimodal Diagnostic Pipeline** for detecting and grading Diabetic Retinopathy from color retinal fundus imaging and scan sequences. By pairing parameter-efficient fine-tuning (QLoRA) on Vision-Language Models (VLMs) and Vision Transformers (ViT) with explicit visual interpretability (attention maps / Grad-CAM), the system outputs both a clinical stage classification and a transparent rationale explaining *why* the decision was made.

---

##  Key Technical Features

* **Multimodal Input Handling:** Accepts single-frame fundus photography (`.jpg`, `.png`) as well as multi-frame retinal video feeds/scans (`.mp4`, `.avi`) via automated temporal keyframe extraction (1 FPS).
* **Clinical 5-Stage Grading:** Classifies retinal scans based on the International Clinical Diabetic Retinopathy (ICDR) scale:
  * **Stage 0:** No Apparent Retinopathy
  * **Stage 1:** Mild Non-Proliferative DR (Microaneurysms)
  * **Stage 2:** Moderate Non-Proliferative DR (Hard Exudates, Cotton Wool Spots)
  * **Stage 3:** Severe Non-Proliferative DR (Venous Beading, Hemorrhages across 4 quadrants)
  * **Stage 4:** Proliferative DR (Neovascularization, Pre-retinal / Vitreous Hemorrhage)
* **Explainable AI (XAI):** Generates self-attention rollout overlays highlighting retinal micro-vascular lesions to verify clinical alignment.
* **Structured Clinical Output:** Generates structured diagnostic summaries detailing:
  1. Identified stage and confidence level
  2. Localized lesion biomarkers
  3. Actionable ophthalmologist / retina specialist referral recommendations

---

##  System Architecture

```text
       Retinal Input (Image or Video Feed)
                        │
                        ▼
      Preprocessing & CLAHE Normalization
                        │
                        ▼
      Vision-Language Backbone (ViT / Qwen2-VL)
                        │
                        ▼
        ┌───────────────┴───────────────┐
        ▼                               ▼
 Attention Heatmap              Structured Clinical Report
(Lesion Localization)          (Stage, Biomarkers, Referral)
