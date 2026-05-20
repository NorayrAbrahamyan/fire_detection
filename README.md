# 🔥 Automated Two-Stage Environmental Fire & Smoke Detection System

## Overview & Abstract
This repository contains a production-grade, end-to-end computer vision and deep learning framework engineered for the autonomous, real-time spatial localization and multi-class classification of environmental hazards—specifically open-terrain wildfires, structural brush fires, and particulate smoke plumes—from static or continuous surveillance camera matrices.

In contrast to computationally expensive, brute-force localized scanning pipelines, this project architects a highly optimized **Two-Stage Object Detection Paradigm**. The framework decouples the processing overhead into two specialized operational stages:
1. **Stage 1 (Region Proposal):** Utilizes an unsupervised, color-textured clustering heuristic via the **Selective Search (SS)** algorithm to isolate non-overlapping regions of interest (RoIs).
2. **Stage 2 (Deep Feature Classification):** Deploys a structurally modified and highly fine-tuned **ResNet18 Convolutional Neural Network (CNN)** backbone to run accelerated class inference on the extracted bounding box candidates.

By separating localization from classification and providing dedicated benchmarking pipelines, the engine establishes a granular diagnostic architecture. This documentation provides exhaustive mathematical, architectural, and operational overviews of the framework to ensure complete scientific reproducibility.

---

## Project Architecture & File System Mapping

The codebase layout is strictly organized according to professional software engineering and machine learning deployment standards. Below is the full directory tree corresponding exactly to the validated directory structures of the local project workspace:

```text
fire_detection/
├── data/                          # Centralized Data Matrix Repository
│   ├── train/                     # Training split for the primary object localization bounding boxes
│   │   ├── images/                # Full-frame raw landscape source imagery used for training
│   │   └── labels/                # Ground Truth annotation files (bounding box coordinates)
│   ├── test/                      # Unseen validation imagery containing full-scale raw landscape scenes
│   │   ├── images/                # Full-frame test images used for final pipeline evaluation
│   │   └── labels/                # Ground Truth annotation files for test metric extraction
│   ├── valid/                     # Validation split for hyperparameter tuning and early stopping
│   │   ├── images/                # Full-frame validation images
│   │   └── labels/                # Ground Truth validation annotation files
│   └── classifier_data/           # Isolated image patch repository cropped specifically for the CNN training
│       ├── train/                 # Cropped sub-frames utilized to train the isolated ResNet18 classifier
│       │   ├── background/        # Negative control samples (clouds, atmospheric fog, dense foliage, sun glare)
│       │   ├── fire/              # Positive cropped patches containing active thermal combustion and open flames
│       │   └── smoke/             # Positive cropped patches containing ascending smoke columns and particulate plumes
│       └── valid/                 # Cropped sub-frames utilized for classifier validation
│           ├── background/        # Negative control validation patches
│           ├── fire/              # Positive fire validation patches
│           └── smoke/             # Positive smoke validation patches
├── models/                        # Serialized Model Artifacts & Heavy Binary Storage
│   └── best_model.pth             # Stored state-dictionary weights of the top-performing fine-tuned ResNet18
├── predictions/                   # Visual Output Target Directory for single-image visual inference test routines
├── detected_errors/               # Deep Quality Assurance (QA) and Error Diagnostic Logging
│   └── false_positives/           # Automated image dumps highlighting system misclassifications and false alarms
├── src/                           # Monolithic Core Source Code Engine Repository
│   ├── detect.py                  # Streamlined localized prediction deployment engine
│   ├── train.py                   # Model optimization loops, Cross-Entropy loss computation, backprop, and checkpointing
│   ├── test.py                    # Static inference wrapper executing Selective Search proposal mapping visually
│   ├── evaluate.py                # End-to-end integrated pipeline performance benchmark across test scenes
│   ├── evaluate_classifier.py     # Isolated upper-bound classifier validation utilizing pre-cropped Ground Truth boxes
│   ├── model.py                   # Neural network topology definitions and custom FC head graph architecture
│   └── prepare_data.py            # Preprocessing script executing data splitting and automated patch cropping
├── requirements.txt               # Locked third-party Python package environment dependency manifest
├── .gitignore                     # Exclusion rules preventing heavy datasets/caches from pushing to version control
└── README.md                      # Documentation
