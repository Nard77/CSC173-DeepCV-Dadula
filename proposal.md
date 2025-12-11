CSC173 Deep Computer Vision Project Proposal

Student: Bernard Deone B. Dadula, 2022-3990.

Date: December 12, 2025

1. Project Title:

Detecting Accidents and Crowd Events in CCTV Footage Using Deep Learning

2. Problem Statement:

This project primarily addresses the challenges of automatically detecting dangerous events, such as accidents and abnormal crowd behavior like fights, panicked running, and stampedes, 
from live CCTV surveillance video in real time. Traditional monitoring heavily relies on human operators in which can easily miss some critical incidents due to fatigue, lack of sleep, limited attention, 
or the increasing number of cameras deployed in public and campus environments. An accurate, low-latency deep learning system can help flag suspicious or hazardous events as they unfold, 
allowing security personnel in places like MSU-IIT or the city of Iligan to respond more quickly and effectively. By focusing on locally relevant scenarios and constraints, 
the project aims to demonstrate how modern computer vision can directly contribute to public safety and incident prevention.

3. Objectives:

- To Fine-tune an existing deep video anomaly or violence-detection model like pretrained CNN+LSTM or 3D CNN on a small, curated subset of CCTV-style clips containing accidents or unusual crowd events

- To Implement a minimal but complete pipeline: data preparation for example frame extraction or clip selection, basic resizing, model fine-tuning, and evaluation on a held-out test set using simple metrics such as accuracy, precision, recall, and F1-score

- To Build a lightweight demo script that processes offline video files and overlays anomaly scores or “normal/anomalous” labels on each clip, approximating real-time behavior on your available hardware

4. Dataset Plan:

Source: Subset of UCF-Crime dataset and maybe some small violence dataset
Classes: Normal which as the name suggest normal events happening in our everyday life and Anomalous in which unusual events like fights, accidents, and people running are included.
Acquisition: Download the dataset from Kaggle then select and trim a manageable subset of clips or we can add some locally recorded CCTV footage that are also privacy-free which means there will be no identifyable faces.

5. Technical Approach:

Architecture sketch
Model: Pretrained CNN + LSTM or a lightweight 3D CNN fine-tuned for binary “normal vs anomalous” classification on short video clips
Framework: PyTorch, using existing torchvision/video models or an open-source anomaly-detection implementation as a starting point to reduce setup time.
Hardware: Google Colab (free GPU) as the primary training environment, with inference and demo scripts runnable on a standard laptop CPU for offline video testing.

6. Expected Challenges & Mitigations:

Challenge: Limited training time and computing resources.
Solution: Use pretrained backbones, freeze early layers, keep models small, and train on short fixed-length clips to reduce computation.
