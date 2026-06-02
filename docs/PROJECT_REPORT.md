<!--
================================================================================
 PROJECT REPORT  -  Brain Tumor Detection
 B.Tech 8th-semester academic report (mirrors EduFeedback AI template)
 Owner   : <STUDENT_NAME>
 Status  : Draft - Step 1 (Front Matter) complete
================================================================================
 PLACEHOLDERS TO FILL BEFORE SUBMISSION (search the file for these):
    <STUDENT_NAME_1>     <ROLL_NUMBER_1>
    <STUDENT_NAME_2>     <ROLL_NUMBER_2>
    <STUDENT_NAME_3>     <ROLL_NUMBER_3>
    <STUDENT_NAME_4>     <ROLL_NUMBER_4>       (delete any unused student blocks)
    <UNIVERSITY_NAME>    <UNIVERSITY_CITY>     <UNIVERSITY_STATE>
    <UNIVERSITY_ADDRESS_LINE>
    <DEPARTMENT_NAME>                          (e.g. "Computer Science and Engineering")
    <DEGREE_FULL>        <DEGREE_SHORT>        (e.g. "Bachelor of Technology" / "B.Tech")
    <SPECIALIZATION>                           (e.g. "Computer Science & Engineering")
    <ACADEMIC_YEAR>                            (e.g. "2025-2026")
    <SEMESTER_NUMBER>                          (e.g. "8th")
    <SUBMISSION_MONTH_YEAR>                    (e.g. "May 2026")
    <SUPERVISOR_NAME>    <SUPERVISOR_DESIGNATION>   (e.g. "Dr. ___", "Assistant Professor")
    <HOD_NAME>                                  (HoD / Reviewer)
================================================================================
-->

# Brain Tumor Detection from MRI Images using an Ensemble of CNN, Transfer Learning and Swin Transformer Models

**<SEMESTER_NUMBER> Semester Project Report**

submitted to

**<UNIVERSITY_NAME>**

In partial fulfilment of the requirements
for the award of the degree of

**<DEGREE_FULL>**

in

**<SPECIALIZATION>**

by

**<STUDENT_NAME_1>** (Roll No. <ROLL_NUMBER_1>)
**<STUDENT_NAME_2>** (Roll No. <ROLL_NUMBER_2>)
**<STUDENT_NAME_3>** (Roll No. <ROLL_NUMBER_3>)
**<STUDENT_NAME_4>** (Roll No. <ROLL_NUMBER_4>)

Under the guidance of

**<SUPERVISOR_NAME>**

Department of <DEPARTMENT_NAME>

**<UNIVERSITY_NAME>**

<SUBMISSION_MONTH_YEAR>

---

© <ACADEMIC_YEAR>, <UNIVERSITY_NAME>. All rights reserved.

---

## DECLARATION OF AUTHORSHIP

We hereby declare that the project report entitled **"Brain Tumor Detection from MRI Images using an Ensemble of CNN, Transfer Learning and Swin Transformer Models"** is an authentic record of our own work carried out at the Department of <DEPARTMENT_NAME>, **<UNIVERSITY_NAME>**, during the <SEMESTER_NUMBER> semester of the academic year <ACADEMIC_YEAR> under the supervision of **<SUPERVISOR_NAME>**, <SUPERVISOR_DESIGNATION>.

We further declare that the matter embodied in this project report has not been submitted by us for the award of any other degree or diploma of this or any other Institute / University. All the information has been obtained and presented in accordance with academic rules and ethical conduct. We also declare that, as required by these rules and conduct, we have fully cited and referenced all materials and results that are not original to this work. The dataset used in this project is publicly available under its original license; no patient-identifiable data was collected, generated, or stored.

Place: <UNIVERSITY_ADDRESS_LINE>, <UNIVERSITY_CITY>, <UNIVERSITY_STATE>
Date: <SUBMISSION_MONTH_YEAR>

Signatures:

| | |
|---|---|
| Signature: ____________________ | Signature: ____________________ |
| Name: **<STUDENT_NAME_1>** | Name: **<STUDENT_NAME_2>** |
| Roll No: <ROLL_NUMBER_1> | Roll No: <ROLL_NUMBER_2> |
| | |
| Signature: ____________________ | Signature: ____________________ |
| Name: **<STUDENT_NAME_3>** | Name: **<STUDENT_NAME_4>** |
| Roll No: <ROLL_NUMBER_3> | Roll No: <ROLL_NUMBER_4> |

---

## CERTIFICATE OF RECOMMENDATION

This is to certify that the Dissertation Report entitled **"Brain Tumor Detection from MRI Images using an Ensemble of CNN, Transfer Learning and Swin Transformer Models"** submitted by **<STUDENT_NAME_1>**, **<STUDENT_NAME_2>**, **<STUDENT_NAME_3>**, and **<STUDENT_NAME_4>** to <UNIVERSITY_NAME>, <UNIVERSITY_CITY>, is a record of bonafide project work carried out by them under my supervision and guidance, and is worthy of consideration for the award of the degree of **<DEGREE_FULL> (<DEGREE_SHORT>)** in **<SPECIALIZATION>**.

&nbsp;

**<SUPERVISOR_NAME>**,
Project Supervisor,
<SUPERVISOR_DESIGNATION>, Department of <DEPARTMENT_NAME>,
<UNIVERSITY_NAME>, <UNIVERSITY_CITY>, <UNIVERSITY_STATE>.

&nbsp;

Approved By:

**<HOD_NAME>**,
HoD / Reviewer, Department of <DEPARTMENT_NAME>,
<UNIVERSITY_NAME>, <UNIVERSITY_CITY>, <UNIVERSITY_STATE>.

---

## ACKNOWLEDGEMENT

We would like to first express our sincere gratitude to our project supervisor, **<SUPERVISOR_NAME>**, <SUPERVISOR_DESIGNATION>, Department of <DEPARTMENT_NAME>, <UNIVERSITY_NAME>. Their continued guidance, technical insight, and constructive feedback during every stage of this project — from problem formulation and dataset selection to model design, evaluation discipline, and report preparation — were instrumental in shaping the final outcome. Their willingness to discuss difficult engineering decisions, including the diagnosis of subtle numerical-stability issues during mixed-precision training, helped us approach the work with both rigour and curiosity.

We take this opportunity to express our gratitude to all faculty members of the Department of <DEPARTMENT_NAME> for their support, the encouragement they provided during our coursework, and the foundation they laid in machine learning, computer vision, and software engineering — without which a project of this scope would not have been possible.

We acknowledge the open-source community whose tools made this work feasible: the PyTorch and `timm` library maintainers, the authors of `pytorch-grad-cam`, the Streamlit and FastAPI teams, and Masoud Nickparvar for releasing the *Brain Tumor MRI Dataset* on Kaggle under a permissive license that supports academic research.

Finally, we thank our parents and families for their unceasing encouragement, patience, and support throughout the semester. We also extend our gratitude to peers and well-wishers who, directly or indirectly, contributed to the completion of this thesis.

---

## ABSTRACT

The accurate and timely classification of brain tumors from magnetic resonance imaging (MRI) is a clinically important problem, yet manual interpretation by radiologists is time-consuming, expertise-intensive, and subject to inter-observer variability. Reported inter-rater agreement on tumor *type* lies in the 70–85 % range, while access to expert neuroradiologists remains uneven across geographies. There is a clear case for AI-assisted decision support that is fast, consistent, and — critically — explainable.

This project, **Brain Tumor Detection**, presents an end-to-end deep-learning system that classifies T1-weighted brain MRI slices into four clinical categories — **glioma**, **meningioma**, **pituitary tumor**, and **no tumor** — using an ensemble of three architecturally diverse models. The first model is a **custom convolutional neural network** of approximately 1.2 million parameters, trained from random initialisation as a VGG-style baseline. The second is **EfficientNet-B0** adapted via two-stage transfer learning from ImageNet, totalling about 4.0 million parameters. The third is **Swin-Tiny**, a hierarchical vision transformer of 27.5 million parameters that uses windowed self-attention and is similarly fine-tuned in two stages. The three models' softmax probability outputs are combined by a weighted-average ensemble whose mixing weights were obtained by exhaustive grid search on a held-out validation set.

The system was developed and evaluated on the publicly available Brain Tumor MRI Dataset (Nickparvar, Kaggle, 7,200 images across four classes). A stratified 80/20 split partitions the 5,600 training images into 4,480 training and 1,120 validation samples, while 1,600 testing images remain held out and are evaluated exactly once at the end of all hyperparameter tuning. On this held-out test set, the strongest individual model (EfficientNet-B0) achieves **94.94 % accuracy** and a **macro-averaged ROC-AUC of 0.991**; the three-model ensemble achieves 94.69 % accuracy at 0.991 AUC. We report an honest empirical finding — that ensembling matches but does not exceed the dominant base learner — and discuss it through the lens of the *dominant-model problem* in ensemble learning. Beyond accuracy, the system provides per-prediction **Grad-CAM** heatmaps for all three models, a low-confidence threshold gate that explicitly recommends radiologist review for uncertain cases, and four deployment channels: a **Streamlit** web interface, a **FastAPI** REST backend, **ONNX** exports that yield a 2–3.5× CPU speedup, and a **Docker** container.

The contributions of this work are (i) a fully reproducible training, evaluation, and inference pipeline; (ii) architectural diversity used deliberately as the basis for ensembling; (iii) rigorous test-set discipline that prevents the over-optimism common in medical AI reports; (iv) explainability via Grad-CAM as a clinical-trust mechanism; and (v) the documentation of a negative ensembling result that has practical and pedagogical value.

**Keywords:**

- Brain Tumor Classification
- Magnetic Resonance Imaging (MRI)
- Convolutional Neural Networks
- Transfer Learning
- Vision Transformers (Swin Transformer)
- Ensemble Learning
- Grad-CAM Explainability
- Computer-Aided Diagnosis (CAD)

---

## Contents

**1. Introduction** &nbsp; ............................................................ 1
&emsp; 1.1 Medical Imaging and Computer-Aided Diagnosis &nbsp; ......... 1
&emsp; 1.2 Objectives of the Project &nbsp; ........................................... 2
&emsp;&emsp; 1.2.1 Primary Objective &nbsp; ............................................ 2
&emsp;&emsp; 1.2.2 Secondary Objectives &nbsp; ...................................... 2
&emsp; 1.3 Challenges in Brain Tumor Classification &nbsp; ..................... 3
&emsp; 1.4 Methodologies &nbsp; ........................................................ 4

**2. Literature Survey** &nbsp; ................................................. 7
&emsp; 2.1 Existing Systems and Studies &nbsp; ................................... 7
&emsp; 2.2 State of the Art &nbsp; ........................................................ 8
&emsp; 2.3 Market Research &nbsp; ..................................................... 9
&emsp; 2.4 Research Gap and Limitations of Existing Systems &nbsp; ...... 10
&emsp; 2.5 Motivation for Project Selection &nbsp; ............................... 10

**3. Dataset and Data Preparation** &nbsp; ........................... 13
&emsp; 3.1 Overview &nbsp; ............................................................. 13
&emsp; 3.2 Dataset Source and Acquisition &nbsp; ............................... 13
&emsp; 3.3 Class Categories and Clinical Meaning &nbsp; ...................... 14
&emsp; 3.4 Rationale for Dataset Selection &nbsp; ................................. 16
&emsp; 3.5 Dataset Statistics &nbsp; .................................................. 17
&emsp; 3.6 Exploratory Data Analysis &nbsp; ....................................... 18
&emsp; 3.7 Train / Validation / Test Split Strategy &nbsp; ..................... 19
&emsp; 3.8 Image Preprocessing Pipeline &nbsp; .................................. 20
&emsp; 3.9 Data Augmentation Strategy &nbsp; .................................... 21
&emsp; 3.10 Ethical Considerations and Licensing &nbsp; ........................ 22

**4. Project Planning** &nbsp; ................................................. 25
&emsp; 4.1 Overview &nbsp; ............................................................. 25
&emsp; 4.2 Current Status of the Project &nbsp; ................................... 25
&emsp; 4.3 System Architecture &nbsp; ............................................... 26
&emsp; 4.4 Identified Challenges and System Limitations &nbsp; ............. 27
&emsp; 4.5 Strategy to Address Identified Challenges &nbsp; .................. 27
&emsp; 4.6 Implementation Roadmap &nbsp; ........................................ 28
&emsp; 4.7 Project Scheduling and Gantt Chart &nbsp; .......................... 29

**5. Project Description** &nbsp; ............................................. 33
&emsp; 5.1 Software Model &nbsp; ..................................................... 33
&emsp; 5.2 Software Requirements Specification (SRS) &nbsp; ............... 34
&emsp;&emsp; 5.2.1 Introduction &nbsp; ................................................. 34
&emsp;&emsp; 5.2.2 General Description &nbsp; ...................................... 34
&emsp;&emsp; 5.2.3 Functional Requirements &nbsp; ............................... 35
&emsp;&emsp; 5.2.4 Interface Requirements &nbsp; ................................. 35
&emsp;&emsp; 5.2.5 Performance Requirements &nbsp; ............................ 36
&emsp;&emsp; 5.2.6 Design Constraints &nbsp; ........................................ 36
&emsp;&emsp; 5.2.7 Non-Functional Attributes &nbsp; .............................. 36
&emsp; 5.3 Functional Specification &nbsp; ......................................... 37
&emsp;&emsp; 5.3.1 Deep Learning Models &nbsp; ................................. 37
&emsp;&emsp; 5.3.2 Mathematical Model and Loss Functions &nbsp; ......... 39
&emsp;&emsp; 5.3.3 Ensemble Decision Logic &nbsp; .............................. 40
&emsp;&emsp; 5.3.4 Grad-CAM Explainability Generation &nbsp; ............. 41
&emsp; 5.4 Design Specification &nbsp; ............................................. 42
&emsp;&emsp; 5.4.1 Use-Case Diagrams &nbsp; ...................................... 42
&emsp;&emsp; 5.4.2 Data Flow Diagrams (DFD) &nbsp; ............................ 43
&emsp;&emsp; 5.4.3 Data Dictionary &nbsp; ............................................. 44

**6. Implementation Issues** &nbsp; ..................................... 53
&emsp; 6.1 Dataset Variability and Image Quality &nbsp; ........................ 53
&emsp; 6.2 Model Selection and Hyperparameter Sensitivity &nbsp; ........ 54
&emsp; 6.3 GPU Memory Constraints and Mixed Precision &nbsp; ........... 54
&emsp; 6.4 Class Confusion and Generalization Gap &nbsp; ................... 55
&emsp; 6.5 Clinical Adoption and Explainability Trust &nbsp; .................. 55
&emsp; 6.6 The Dominant-Model Problem in Ensembling &nbsp; .............. 56

**7. Conclusion, Summary and Future Scope** &nbsp; ............... 59
&emsp; 7.1 Summary of the Work &nbsp; ............................................ 59
&emsp; 7.2 Conclusion &nbsp; ........................................................... 60
&emsp; 7.3 Future Scope &nbsp; ........................................................ 61

**Bibliography** &nbsp; ..................................................... 63

---

## List of Figures

| Figure | Caption | Page |
|---|---|---:|
| 1.1 | High-level system flow: input MRI → preprocessing → three parallel models → weighted ensemble → predicted class, confidence, and Grad-CAM heatmaps | 5 |
| 2.1 | Global incidence and mortality trends of brain and central-nervous-system tumors | 9 |
| 2.2 | Comparison of reported brain MRI classification accuracies across recent CNN, transfer-learning, and transformer-based studies | 10 |
| 3.1 | Folder layout and file-count breakdown of the Brain Tumor MRI Dataset | 14 |
| 3.2 | Representative MRI slices, one per class (glioma, meningioma, no tumor, pituitary) | 15 |
| 3.3 | Per-class image counts across the Training, Validation, and Test splits | 17 |
| 3.4 | Distribution of original image dimensions (width × height) across all source images | 18 |
| 3.5 | Per-class pixel-intensity histograms, illustrating substantial overlap and the consequent need for deep feature learning | 19 |
| 3.6 | Visual comparison of an original MRI slice and the same slice after the training-time augmentation pipeline | 22 |
| 4.1 | Layered system architecture of the Brain Tumor Detection platform | 26 |
| 4.2 | Gantt chart of the project schedule across the development semester | 30 |
| 5.1 | Use-case diagram — Radiologist / Clinician role | 42 |
| 5.2 | Use-case diagram — Researcher / Developer role | 42 |
| 5.3 | Use-case diagram — Administrator role | 43 |
| 5.4 | Data Flow Diagram — Model Training pipeline | 43 |
| 5.5 | Data Flow Diagram — Single-image Inference pipeline | 44 |
| 5.6 | Data Flow Diagram — Grad-CAM Explainability generation | 44 |
| 5.7 | Custom CNN architecture — four convolutional blocks followed by global-average pooling and dense head | 38 |
| 5.8 | EfficientNet-B0 transfer-learning architecture with two-stage fine-tuning schedule | 39 |
| 5.9 | Swin-Tiny architecture — patch embedding, four hierarchical stages of shifted-window self-attention | 40 |
| 6.1 | Training and validation loss / accuracy curves for the three base models | 56 |
| 6.2 | Confusion matrices on the held-out test set for each model and the final ensemble | 57 |
| 6.3 | Receiver Operating Characteristic (ROC) curves for the three models and the ensemble | 57 |
| 6.4 | Representative Grad-CAM overlays for each base model on a correctly-classified glioma sample | 58 |
| 6.5 | Final comparison bar chart of per-model and ensemble test-set accuracy, F1, and macro AUC | 58 |

---

## List of Tables

| Table | Caption | Page |
|---|---|---:|
| 3.1 | Per-class image counts in the Training, Validation, and Test splits | 17 |
| 3.2 | Distribution of source image colour modes and pixel-dimension ranges | 18 |
| 3.3 | Image preprocessing transforms applied to all data (training, validation, test, inference) | 20 |
| 3.4 | Training-time data augmentation policy with clinical justification | 21 |
| 4.1 | Phases of the entire project work | 30 |
| 5.1 | Functional requirements of the Brain Tumor Detection system | 35 |
| 5.2 | Non-functional requirements and quality attributes | 37 |
| 5.3 | Architectural summary of the three base models | 38 |
| 5.4 | Two-stage training schedule for EfficientNet-B0 and Swin-Tiny | 40 |
| 5.5 | Loss function and optimizer configuration shared across all base models | 41 |
| 5.6 | Ensemble Configuration schema (`ensemble_config.json`) | 45 |
| 5.7 | Prediction Result schema (returned by `BrainTumorPredictor.predict()`) | 46 |
| 5.8 | Model Checkpoint Registry schema | 47 |
| 5.9 | Training Run Metadata schema (TensorBoard / JSON history) | 48 |
| 5.10 | Image Preprocessing Configuration schema | 49 |
| 5.11 | Inference API Request / Response schema (`/predict` endpoint) | 50 |
| 6.1 | Per-model test-set performance: parameters, accuracy, macro-F1, macro-AUC, training time, and latency | 59 |
| 6.2 | Ensemble grid-search results — top five weight combinations evaluated on the validation set | 60 |
| 6.3 | Per-class precision, recall, and F1-score on the held-out test set | 60 |
| 6.4 | ONNX-runtime versus PyTorch-eager CPU inference latency, with argmax-agreement verification | 61 |

---

<!-- ============================================================================
     STEP 1 - FRONT MATTER COMPLETE
     STEP 1.5 - CHAPTER 3 (Dataset and Data Preparation) WRITTEN OUT OF ORDER
     ON USER REQUEST - delivered before Chapters 1, 2, 4, 5, 6, 7
     ============================================================================ -->

# Chapter 3
# Dataset and Data Preparation

## 3.1 Overview

A well-characterised and reproducible dataset is the foundation of any supervised deep-learning system, and the credibility of every accuracy figure reported in the later chapters of this report rests directly on the quality, balance, and provenance of the data used to train and evaluate the models. This chapter is therefore devoted entirely to the dataset.

The sections that follow describe the source of the data and the manner in which it was obtained, the clinical meaning of the four diagnostic classes the system is required to distinguish, the rationale for selecting this particular dataset over the several available alternatives, the exploratory analysis performed to verify data integrity and characterise variation across the corpus, the deterministic train / validation / test split protocol that underpins the project's commitment to honest evaluation, the image preprocessing transforms applied to every input that enters the model, and the augmentation policy applied selectively at training time to improve generalisation. The chapter closes with a short note on ethical considerations and licensing.

By collecting all data-related decisions in a single chapter, the chapters that follow can focus on what is genuinely model- and system-specific without re-litigating data choices.

## 3.2 Dataset Source and Acquisition

The dataset used throughout this project is the publicly available **Brain Tumor MRI Dataset**, released by **Masoud Nickparvar** on the Kaggle platform. The dataset is freely accessible at the following URL:

> **Kaggle URL:**
> [`https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The collection is itself a curated aggregation of three earlier public brain MRI resources — the **figshare** brain tumor dataset, the **SARTAJ** brain tumor dataset, and the **Br35H** brain tumor detection dataset — re-organised by the author into a single unified four-class taxonomy with a consistent Training / Testing folder layout. Re-aggregation of multiple upstream sources has the desirable side effect of yielding a heterogeneous corpus that spans multiple scanners, intensity profiles, and acquisition protocols.

The dataset is distributed as plain image files (`.jpg`) organised in a two-level `<split>/<class>/` directory layout. No DICOM headers, patient identifiers, scanner metadata, or other forms of protected health information accompany the images; each file is a single 2D axial brain MRI slice anonymised at source. After the per-class balancing performed by the author, the dataset contains a total of **7,200 images** split as 5,600 in the Training folder and 1,600 in the Testing folder, with exactly 1,800 images per class spread across the two folders. The folder layout is shown schematically in Figure 3.1.

> **Figure 3.1 — Folder layout and file-count breakdown of the Brain Tumor MRI Dataset.**
> Each of the four diagnostic classes appears as a sub-directory under `Training/` and `Testing/` respectively. The dataset is class-balanced, with 1,400 training and 400 testing images per class.

## 3.3 Class Categories and Clinical Meaning

The four classes in the dataset correspond to the three most clinically common categories of primary brain tumor and a `no-tumor` control class. A brief medical overview of each is provided below to motivate the difficulty of the four-way classification task and to ground the model's predicted labels in their clinical meaning. Representative examples of each class, drawn directly from the test split, are shown in Figure 3.2.

**Glioma.** Gliomas are neoplasms that arise from the glial cells of the central nervous system — predominantly astrocytes, oligodendrocytes, and ependymal cells — and constitute the most common form of *malignant* primary brain tumor in adults. They are graded by the World Health Organization on a scale from I (least aggressive) to IV (most aggressive), with the highest-grade variant, glioblastoma multiforme, being among the most aggressive of all human cancers. On T1-weighted MRI, gliomas frequently appear as ill-defined intra-axial masses, often with surrounding vasogenic oedema, internal heterogeneity, and a measurable mass-effect on adjacent brain structures. Their visual diversity makes them the single most challenging class for any automated classifier.

**Meningioma.** Meningiomas arise not from brain parenchyma itself but from the arachnoid cap cells of the **meninges** — the three layers of protective membrane that surround the brain and spinal cord. They are the most common *primary* brain tumor overall and are typically benign and slow-growing. On MRI, meningiomas characteristically appear as well-circumscribed, extra-axial masses attached to the dura, often exhibiting the so-called *"dural tail"* enhancement pattern that radiologists use as a confirmatory sign. Despite their distinctive radiological signature on multi-modal imaging, their appearance on a single T1-weighted axial slice can be visually similar to that of a glioma, and the empirical confusion between these two classes is the largest single source of error in our models.

**Pituitary tumor.** Pituitary adenomas are growths arising from the **pituitary gland**, a small endocrine organ situated at the base of the brain in the *sella turcica*, a saddle-shaped bony depression of the sphenoid bone. The overwhelming majority of pituitary tumors are benign, and they are clinically distinguished as much by their disruption of hormone regulation as by their mass effect on surrounding structures. On MRI, pituitary lesions appear within or just superior to the sella turcica, at the very base of the brain — an anatomically constrained location that makes them the easiest of the three tumor types for an automated classifier to identify.

**No tumor.** This class comprises MRI slices with no detectable mass lesion. Its inclusion in the four-way taxonomy is essential rather than incidental: a clinically usable diagnostic system must be capable of confidently identifying a *normal* scan, because false-positive tumor predictions on healthy patients carry their own clinical costs — they precipitate avoidable patient anxiety, unnecessary follow-up imaging, and in some cases unnecessary biopsies.

> **Figure 3.2 — Representative MRI slices, one per class.**
> The four panels show a glioma, a meningioma, a no-tumor scan, and a pituitary tumor respectively. Note the anatomic localisation of the pituitary case at the base of the brain; the central mass-effect typical of glioma; the well-circumscribed extra-axial appearance of meningioma; and the visually clean parenchyma of the no-tumor control.

## 3.4 Rationale for Dataset Selection

Several brain MRI datasets are available in the public domain, including the BraTS challenge collections, the standalone figshare dataset, the IXI dataset, and various single-institution releases. The Nickparvar dataset was selected for this project after weighing each candidate against the following criteria.

1. **Four-class taxonomy aligned with the clinical problem.** Many alternative datasets address only binary "tumor / no-tumor" classification, or focus on volumetric segmentation rather than slice-level classification. The four-class structure of the Nickparvar collection — three named tumor types plus a no-tumor control — matches the radiological triage problem that this project sets out to model.

2. **Class balance.** After the per-class augmentation performed by the original author, the Training folder contains a near-uniform number of images per class. Balanced classes simplify training, remove the need for class-weighted losses (which can themselves introduce subtle confounders), and make accuracy a more informative headline metric.

3. **Sufficient volume for transfer learning.** With approximately 7,200 images, the dataset is large enough to meaningfully fine-tune ImageNet-pretrained backbones such as EfficientNet-B0 and Swin-Tiny without entering the very-low-data regime in which pretrained features cannot be specialised. It is simultaneously small enough that the entire training and evaluation pipeline fits comfortably within the time and VRAM budget of a modest workstation GPU.

4. **Public and permissive licensing.** The dataset is released for academic and research use, contains no patient identifiers, and is unambiguously citable — three properties that are jointly critical for any work that aspires to reproducibility.

5. **Established benchmark.** The dataset has been used as the basis of dozens of peer-reviewed brain-tumor classification studies published since 2021. This provides both implicit external validation of the data's quality and a known competitive baseline against which our results can be situated.

6. **Heterogeneity of source images.** Because the dataset aggregates three upstream collections, the resulting images span multiple scanner manufacturers, intensity profiles, and resolutions. This heterogeneity is a feature rather than a defect — it forces any model trained on the data to be robust to the inter-scanner variation that characterises real clinical deployments.

7. **Format simplicity.** Files are JPEGs in a clean folder structure, which eliminates the operational overhead of DICOM parsing and lets the engineering effort focus on the modelling pipeline rather than the data-ingestion plumbing.

Datasets considered and *not* selected include the BraTS series (which targets volumetric tumor segmentation and provides multi-modal images that exceed the scope of a slice-level classification project) and IXI (which contains primarily healthy-subject scans and is therefore unsuitable for a multi-class tumor classifier).

## 3.5 Dataset Statistics

The full dataset, as used in this project after the balancing performed by the original author, contains the per-class counts shown in Table 3.1. The Training folder is partitioned via stratified random sampling into model-training and validation subsets at an 80 / 20 ratio; the Testing folder is retained as a strictly held-out evaluation set and is touched exactly once, at the end of all hyperparameter tuning, to produce the final reported metrics.

**Table 3.1 — Per-class image counts in the Training, Validation, and Test splits.**

| Split | glioma | meningioma | notumor | pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Training (full) | 1,400 | 1,400 | 1,400 | 1,400 | **5,600** |
| ↳ Train (80 %) | 1,120 | 1,120 | 1,120 | 1,120 | 4,480 |
| ↳ Validation (20 %) | 280 | 280 | 280 | 280 | 1,120 |
| Test (held out) | 400 | 400 | 400 | 400 | **1,600** |
| **Grand total** | **1,800** | **1,800** | **1,800** | **1,800** | **7,200** |

The class distribution across all three splits is depicted in Figure 3.3. The perfect class balance, both before and after the train / validation split, means that a trivial constant-prediction baseline would achieve exactly 25 % accuracy, and any model performing above this floor is genuinely learning class-discriminative features.

> **Figure 3.3 — Per-class image counts across the Training, Validation, and Test splits.**
> Each class contributes exactly the same number of images to each split, ensuring that no class is over- or under-represented relative to the others at any stage of training or evaluation.

## 3.6 Exploratory Data Analysis

Before any model was trained, the dataset was characterised across four dimensions — file integrity, image-mode distribution, pixel-dimension distribution, and per-class pixel-intensity distribution — to identify any anomalies that would require special handling in the preprocessing pipeline. The findings are summarised below and reported quantitatively in Table 3.2.

**File integrity.** Every image file in both splits was opened with the Python Imaging Library; no corrupt files, truncated streams, or unreadable headers were encountered, and consequently no images were excluded from the experiments for technical reasons.

**Image modes.** The dataset is heterogeneous in colour mode. Of the 7,200 images, **2,585 are stored as RGB triplets**, **1,892 as single-channel grayscale** (PIL mode `'L'`), and **3 as four-channel RGBA**. All ImageNet-pretrained backbones expect three-channel inputs, so a uniform conversion to RGB via `PIL.Image.convert('RGB')` is applied at the start of every preprocessing pipeline. Grayscale images are converted by replicating the single intensity channel three times; RGBA images are flattened to RGB by discarding the alpha channel.

**Pixel dimensions.** Image dimensions vary widely across the corpus, from a minimum of **150 × 168 pixels** to a maximum of **1,375 × 1,446 pixels**. This range reflects the multi-source origin of the dataset and the absence of any single canonical scan resolution. All images are square-resized to **224 × 224 pixels** during preprocessing — the canonical input size for both EfficientNet-B0 and Swin-Tiny under their published ImageNet checkpoints, and the resolution at which the custom CNN was designed (see §5.3.1). The distribution of source image dimensions is plotted in Figure 3.4.

**Pixel-intensity distribution.** Per-class pixel-intensity histograms were computed to test whether any class might be identifiable from low-order pixel statistics alone — a property that would make the four-way classification trivially easy and the headline metrics unreliable. The histograms (Figure 3.5) overlap substantially across all four classes, confirming that no class is separable from the others by simple thresholding of pixel values. Deep feature learning is therefore genuinely required, and the accuracy figures reported in Chapter 6 reflect genuine learnt structure rather than a brightness shortcut.

> **Figure 3.4 — Distribution of original image dimensions across all source images.** Width is plotted against height for every file in the dataset; the wide spread motivates the uniform resize to 224 × 224 prior to model input.

> **Figure 3.5 — Per-class pixel-intensity histograms.** Histograms are normalised to unit area. Substantial overlap between classes shows that classification cannot be reduced to a simple thresholding of mean intensity.

**Table 3.2 — Distribution of source image colour modes and pixel-dimension ranges.**

| Property | Value |
|---|---|
| Total image count | 7,200 (5,600 training + 1,600 testing) |
| Colour modes encountered | RGB (2,585), grayscale `L` (1,892), RGBA (3) |
| Minimum dimensions | 150 × 168 pixels |
| Maximum dimensions | 1,375 × 1,446 pixels |
| Median dimensions | approximately 512 × 512 pixels |
| Target dimensions (after resize) | 224 × 224 pixels |
| Corrupt or unreadable files | 0 |

## 3.7 Train / Validation / Test Split Strategy

A central methodological commitment of this project is the separation of the available data into three subsets whose roles are *strictly enforced* throughout the project lifecycle:

- **Training set (4,480 images, 62.2 % of the corpus)** — used for gradient updates during model fitting. Every gradient step ever computed in this project was computed against an image drawn from this subset.
- **Validation set (1,120 images, 15.6 %)** — used for early-stopping signals, for hyperparameter and learning-rate selection, for ensemble-weight grid search, and for any other decision that involves comparing one candidate model against another. The validation set is allowed to leak into model-selection decisions; it is *not* allowed to leak into reported headline metrics.
- **Test set (1,600 images, 22.2 %)** — held out entirely until all hyperparameters and modelling choices have been frozen, then evaluated exactly once to produce the final results that appear in this report. Once the test set has been touched, no further hyperparameter tuning is permitted; doing otherwise would invalidate the test set's role as a proxy for genuinely unseen data.

The train / validation split is performed by `sklearn.model_selection.train_test_split` with `stratify=labels` and `random_state=42`, guaranteeing both class balance across the split and bit-exact reproducibility. The resulting partition is cached as three CSV files at `data/processed/{train,val,test}_split.csv` and version-controlled together with the source code, so that any subsequent experiment automatically inherits the identical split.

This discipline is critical to the credibility of the headline numbers. The validation-to-test accuracy gap observed across the experiments — between four and eight percentage points in absolute terms across all three models — confirms that any system that reported only validation-set metrics would substantially overstate its real-world performance. By explicitly carving out and holding out the test set, the project commits in advance to the higher and harder honest number.

## 3.8 Image Preprocessing Pipeline

A single deterministic preprocessing transform is applied to validation and test images, and to every image processed at inference time. The training pipeline applies the same deterministic transform *and additionally* a stochastic augmentation stage that is described in §3.9. The deterministic pipeline consists of the four steps listed in Table 3.3, applied in the order in which they appear.

**Table 3.3 — Image preprocessing transforms applied to all data.**

| Step | Transform | Parameters / Notes |
|---|---|---|
| 1 | **Convert to RGB** | `PIL.Image.convert('RGB')` — coerces grayscale (`L`), RGB, and RGBA inputs to a uniform 3-channel representation |
| 2 | **Resize** | Target 224 × 224 pixels; bilinear interpolation; aspect-ratio distortion preferred to letter-boxing because brain MRI scans are already approximately square |
| 3 | **Convert to tensor** | Output `float32` tensor; range [0, 1]; channels-first layout |
| 4 | **Normalise (ImageNet statistics)** | Per-channel mean = [0.485, 0.456, 0.406], standard deviation = [0.229, 0.224, 0.225] |

Several of these decisions deserve brief justification.

The conversion to RGB at step 1 is necessary because both EfficientNet-B0 and Swin-Tiny were pretrained on three-channel natural images and their first-layer convolutional kernels expect three input channels. Replicating the grayscale channel three times rather than expanding the kernel to single-channel input preserves the value of the pretrained weights without any modification.

The choice of 224 × 224 at step 2 is dictated by the published ImageNet checkpoints used for transfer learning. Using a different target resolution would invalidate the pretrained positional embeddings (in the case of Swin) and would force a re-initialisation of position-sensitive parameters that would discard a non-trivial fraction of the value of the pretrained representation.

The use of ImageNet mean and standard deviation at step 4 is essential for the same reason: the pretrained backbones expect inputs to lie in this normalised distribution, and any other normalisation would produce inputs that the pretrained weights do not expect.

## 3.9 Data Augmentation Strategy

Augmentation is applied *only* to training images and *only* at training time. Validation and test images are never augmented, so every accuracy figure reported in Chapter 6 reflects model performance on unmodified inputs. The augmentation policy is deliberately conservative — augmentations that would produce clinically implausible images, such as large rotations, vertical flips that invert cranio-caudal anatomy, or drastic colour shifts, are explicitly excluded.

The full augmentation policy is summarised in Table 3.4. Each transform is accompanied by the clinical rationale that justifies its inclusion.

**Table 3.4 — Training-time data augmentation policy.**

| Augmentation | Probability / Range | Clinical justification |
|---|---|---|
| Random horizontal flip | p = 0.5 | The brain is approximately bilaterally symmetric on axial slices; a mirrored image remains anatomically plausible. |
| Random rotation | ±15° | Patients are not always perfectly aligned with the scanner gantry; small head-tilt rotations are common in real acquisitions. |
| Random affine — translate | ±5 % of image side | Simulates field-of-view variation between scanners and patient positioning. |
| Random affine — scale | ±5 % | Simulates minor magnification and zoom differences. |
| Colour jitter — brightness | ±0.20 | Simulates scanner-gain variation across acquisitions. |
| Colour jitter — contrast | ±0.20 | Simulates intensity-window variation. |
| Random erasing | p = 0.25, area 2–10 % | Randomly blanks a rectangular patch, forcing the model to reason about global structure rather than fixate on any single region. |

Three augmentations were considered and *deliberately excluded*. Vertical flips would invert cranio-caudal anatomy and produce images that no real scanner ever generates. Aggressive rotations of more than approximately 30 degrees would produce images outside the distribution of any realistic scan protocol. Hue jitter would distort intensity-encoded pathology in MRI, where pixel intensities already carry diagnostic meaning, and was found in pilot experiments to degrade rather than improve validation accuracy. Cutout patches larger than 10 % of image area were observed empirically to harm validation accuracy by occluding the tumor region itself in a non-trivial fraction of training samples, defeating the purpose of the training step.

Applied together, the augmentation policy roughly trebles the effective size of the training corpus without any additional data collection, while keeping every augmented image within the distribution that a real radiologist would consider plausible. Figure 3.6 shows a side-by-side comparison of an original training image and a randomly drawn augmented version of the same image.

> **Figure 3.6 — Visual comparison of an original MRI slice and the same slice after the training-time augmentation pipeline.**
> Note that the tumor structure remains clearly visible; augmentation alters scanner-style nuisance variables (orientation, brightness, contrast, position) without disturbing the diagnostic content.

## 3.10 Ethical Considerations and Licensing

The dataset is publicly distributed by its original author for academic and research use, and no patient-identifiable information is present in any file. No additional patient data was collected, generated, or annotated during this project. All experimental work was performed on the publicly released images alone.

The output of the trained system is intended for academic demonstration and educational use only and is not represented as a medical device, a diagnostic tool, or a substitute for the judgement of a qualified radiologist. Any future clinical deployment of work derived from this project would necessarily require institutional ethics review, regulatory approval under the applicable jurisdiction, and prospective validation on multi-institutional patient data — none of which falls within the scope of the present academic study.

The user-facing interface described in Chapter 5 explicitly surfaces this limitation: the Streamlit dashboard displays a persistent advisory that the system is for academic use only and is not a substitute for radiologist diagnosis, and the low-confidence threshold gate is configured by default to recommend radiologist review for any prediction whose softmax confidence falls below 0.85.

<!-- ============================================================================
     END OF STEP 1 + INSERTED CHAPTER 3
     Next: Step 2 - Chapter 1 (Introduction)
     ============================================================================ -->
