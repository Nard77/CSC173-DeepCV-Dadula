Project Title: Detecting Accidents and Crowd Events in CCTV Footage Using Deep Learning
CSC173 Intelligent Systems Final Project
Mindanao State University - Iligan Institute of Technology
Student: Bernard Deone B. Dadula, 2022-3990
Semester: AY 2025-2026 Sem 1


Abstract:
This project addresses the problem of automatically detecting dangerous events, such as accidents and abnormal crowd behavior such as fights or panicked running,
in CCTV surveillance video to support faster and more reliable incident response. The system will use a small public video anomaly or violence-detection dataset,
complemented if needed by a limited number of privacy-safe CCTV-style clips, labeled into two classes: normal and anomalous. A pretrained deep spatio-temporal model,
such as a CNN-based feature extractor followed by an LSTM or a lightweight 3D CNN, will be fine-tuned for binary anomaly classification on short video clips,
using basic preprocessing and augmentation to mitigate the small-data setting. Within a one-week development window, the project will implement a minimal end-to-end pipeline covering data preparation,
model fine-tuning, validation, and evaluation with metrics like accuracy, precision, recall, F1-score, and approximate inference time per clip.

Table of Contents:
Introduction
Related Work
Methodology
Experiments & Results
Discussion
Ethical Considerations
Conclusion
Installation
References

Introduction
Problem Statement:
  This project primarily addresses the challenges of automatically detecting dangerous events, such as accidents and abnormal crowd behavior like fights, panicked running, and stampedes,
from live CCTV surveillance video in real time. Traditional monitoring heavily relies on human operators in which can easily miss some critical incidents due to fatigue, lack of sleep,
limited attention, or the increasing number of cameras deployed in public and campus environments. An accurate, low-latency deep learning system can help flag suspicious or hazardous events as they unfold,
allowing security personnel in places like MSU-IIT or the city of Iligan to respond more quickly and effectively. By focusing on locally relevant scenarios and constraints,
the project aims to demonstrate how modern computer vision can directly contribute to public safety and incident prevention.

Objectives
- To Fine-tune an existing deep video anomaly or violence-detection model like pretrained CNN+LSTM or 3D CNN on a small, curated subset of CCTV-style clips containing accidents or unusual crowd events

- To Implement a minimal but complete pipeline: data preparation for example frame extraction or clip selection, basic resizing, model fine-tuning, and evaluation on a held-out test set using simple metrics such as accuracy, precision, recall, and F1-score

- To Build a lightweight demo script that processes offline video files and overlays anomaly scores or “normal/anomalous” labels on each clip, approximating real-time behavior on your available hardware

Related Work
[Paper 1: Real-world Anomaly Detection in Surveillance Videos[1]]
[Paper 2: Video Anomaly Detection [2]]
[Paper 3: Real-Time Violence Detection in Surveillance Videos Using Deep Learning Approach [3]]
Gap: Emphasizes “normal vs anomalous” CCTV behavior with concrete safety events like accidents, fights, or panic.

Methodology
Dataset
Source: Public video anomaly / violence dataset, e.g., a small subset of UCF-Crime or a Kaggle “violence in surveillance video” dataset, with around 50–150 short CCTV‑style clips labeled as normal or anomalous.
Split: 70% train, 15% validation, 15% test at the clip level, ensuring that clips from the same original video do not appear in multiple splits to avoid leakage.
Preprocessing: 
- Extract fixed-length clips from each video.
- Resize frames to a consistent resolution and normalize pixel values.
- Apply light augmentation such as random horizontal flips, random cropping, and small brightness/contrast changes to improve robustness under a small dataset
Architecture
Model Diagram

Backbone: [e.g., CSPDarknet53]
Head: [e.g., YOLO detection layers]
Hyperparameters: Table below
Parameter	Value
Batch Size	16
Learning Rate	0.01
Epochs	100
Optimizer	SGD
Training Code Snippet
train.py excerpt model = YOLO('yolov8n.pt') model.train(data='dataset.yaml', epochs=100, imgsz=640)

Experiments & Results
Metrics
Model	mAP@0.5	Precision	Recall	Inference Time (ms)
Baseline (YOLOv8n)	85%	0.87	0.82	12
Ours (Fine-tuned)	92%	0.94	0.89	15
Training Curve

Demo
 [Video: CSC173_YourLastName_Final.mp4] [web:41]

Discussion
Strengths: [e.g., Handles occluded trash well]
Limitations: [e.g., Low-light performance]
Insights: [e.g., Data augmentation boosted +7% mAP] [web:25]
Ethical Considerations
Bias: Dataset skewed toward plastic/metal; rural waste underrepresented
Privacy: No faces in training data
Misuse: Potential for surveillance if repurposed [web:41]
Conclusion
[Key achievements and 2-3 future directions, e.g., Deploy to Raspberry Pi for IoT.]

Installation
Clone repo: git clone https://github.com/yourusername/CSC173-DeepCV-YourLastName
Install deps: pip install -r requirements.txt
Download weights: See models/ or run download_weights.sh [web:22][web:25]
requirements.txt: torch>=2.0 ultralytics opencv-python albumentations

References
[1] Jocher, G., et al. "YOLOv8," Ultralytics, 2023.
[2] Deng, J., et al. "ImageNet: A large-scale hierarchical image database," CVPR, 2009. [web:25]

GitHub Pages
View this project site: https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/ [web:32]
