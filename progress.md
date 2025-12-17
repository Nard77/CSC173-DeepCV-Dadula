#CSC173 Deep Computer Vision Project Progress Report

Student: [Bernard Deone B. Dadula], [2022-3990]

Date: [12/17/2025]

Repository: https://github.com/yourusername/CSC173-DeepCV-Dadula

##📊 Current Status
Milestone	            Status	         Notes

Baseline Evaluation	⏳ Pending	Training ongoing

##1. Dataset Progress

Total Videos: [400]

Train/Val/Test split: [60%/20%/20%]

Classes implemented: [Normal (Everyday CCTV Activities), Anomalous (Armed Robbery, Assault, Weaponized)]

Preprocessing applied: Resize(244x244) resolution, RGB conversion, Normalization with ImageNet and Basic augmentation (flip, rotate, brightness)

##2. Training Progress
Training Curves (so far) Loss Curve mAP Curve

Current Metrics:

Metric	Train	Val
Loss	[63%]	[52%]

Accuracy	[63%]	[84%]

Precision	[65%]	[90%]

Recall	[61%]	[72%]

##3. Challenges Encountered & Solutions

Issue                 Status      Resolution

No GPU in Colab       ✅ Fixed    Switched runtime to GPU (Runtime → Change runtime type → GPU); verified CUDA available in notebook.

Limited dataset size  ⏳ Ongoing  Started with a small subset of Kaggle CCTV videos; using data augmentation (flip + brightness/contrast jitter) to increase diversity.

Training instability  ⏳ Ongoing  Using pretrained ResNet as frozen feature extractor and a small LSTM; monitoring loss/accuracy and planning to tune learning rate and epochs.

Slow training on CPU  ✅ Avoided  Ensured GPU is enabled; kept clip_len and batch_size small enough to fit within Colab’s limits.


##4. Next Steps (Before Final Submission)

 ⏳Complete training (50 more epochs)

 ⏳Hyperparameter tuning (learning rate, augmentations)

 ⏳Baseline comparison (vs. original pre-trained model)

 ⏳Record 5-min demo video

 ⏳Write complete README.md with results

