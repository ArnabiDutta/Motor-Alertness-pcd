# Motor-Alertness-pcd

Detecting twitching and tremoring in point cloud human representations. This README outlines the complete innovation and development pipeline.

---

##  Pipeline Overview

1. **Calibration using Koide3**

   * Ensure accurate alignment of sensors using [Koide's Calibration Tool](https://github.com/koide3/hdl_graph_slam)

2. **Convert PCD to Depth Image**

   * Project 3D point cloud into 2.5D depth format for visualization or input to RGB-D models.

3. **Segmentation**

   * Segment human body from background or label body parts using:

     * Clustering
     * [SMPL fitting](https://smpl.is.tue.mpg.de/)
     * [Body-part segmentation networks](https://arxiv.org/abs/1903.06806)

---

##  Tremor Detection Approaches

### 🔵 1st Approach: Scene Flow of 3D Keypoints

#### Pipeline:

```
Point Cloud Stream (t0, t1, t2...)
   ↓
Keypoint Estimation (e.g., PoseNet3D, SMPL, custom landmark tracker)
   ↓
Extract Keypoint Coordinates per Frame
   ↓
Scene Flow Estimation (e.g., PointPWC-Net, FlowNet3D)
   ↓
Sample Scene Flow at Keypoint Locations
   ↓
Build Temporal Trajectories (Pos, Vel, Acc, Flow)
   ↓
Detect Twitch/Tremor: frequency analysis, anomaly scores, pattern recognition
```

#### Scene Flow Estimation Resources:

* [PointPWC-Net GitHub](https://github.com/DylanWusee/PointPWC)
* [FlowNet3D](https://github.com/xingyul/flownet3d)

#### Feature Engineering:

* Estimate 3D motion vectors per point
* Compute statistics: mean, variance, entropy
* Tremors show periodic energy in peri-articular regions

---

### 🟠 2nd Approach: 3D Pose Estimation & Keypoint Tracking

#### 1. Extracting 3D Keypoints

##### a. Direct 3D Pose Estimation

* [PoseNet3D](https://arxiv.org/abs/1811.11742)
* [VNect](https://gvv.mpi-inf.mpg.de/projects/VNect/)
* [LiftPose3D](https://arxiv.org/abs/2008.10551)
* [VideoPose3D](https://arxiv.org/abs/1811.11742)
* Point-cloud methods: [PointNet++](https://arxiv.org/abs/1706.02413), [PoseNet++](https://arxiv.org/abs/2012.05371)
* RGB-D: [MediaPipe Holistic 3D](https://google.github.io/mediapipe/solutions/holistic.html), OpenPose + depth lifting

##### b. Fitting Parametric Body Models

* [SMPL](https://smpl.is.tue.mpg.de/) or [SMPL-X](https://smpl-x.is.tue.mpg.de/)

#### 2. Tracking Keypoints

* Apply **Kalman** or **Savitzky-Golay filters** for smoothing
* Track per-joint time-series: `joint_pos[t] = (x, y, z)`

#### 3. Motion Features

* Displacement, velocity, acceleration, **jerk**
* Tremor: rhythmic, small-magnitude
* Twitch: sharp, high-jerk events

#### 4. Frequency Analysis

* FFT / STFT of `(x, y, z)` signals
* Detect dominant frequency in:

  * Parkinsonian tremor: 4–6 Hz
  * Essential tremor: 6–12 Hz
* Compute:

  * Power Spectral Density (PSD)
  * Signal Entropy

#### 5. Statistical Descriptors

* Mean, variance
* RMS velocity/acceleration
* Jitter Index = high-freq / low-freq energy

#### 6. Spatial-Temporal Consistency

* Correlate neighboring joints (e.g., hands)
* Normalize positions w\.r.t. root (pelvis/spine)

#### 7. ML Models

* Feature-based: Random Forest, SVM, MLP
* Sequence: LSTM, GRU, 1D CNN, Transformers

#### 8. Advanced Innovations

* **Joint symmetry analysis**: mirror trajectory & phase correlation
* **Region-based alerts**: per-joint abnormality score
* **Hybrid system**: fuse IMU + pose + point cloud motion

---

### 🟢 3rd Approach: Local Oscillations as Bio-Inspired Fields

#### 1. Segment Point Cloud into Body Regions

* Using [SMPL](https://smpl.is.tue.mpg.de/) or movement-based clustering
* Each segment = local oscillator

#### 2. Extract Oscillation Signals

* Compute centroid trajectory
* Monitor local point jitter & displacement vector

#### 3. Model Inter-Region Synchronization

* Hilbert transform → compute phase
* Use Kuramoto-style analysis:

  * Coherence = normal
  * Desynchronization = tremor/twitch

#### 4. Burst Detection

* Use peak detection and jerk metrics
* Frequency burst = tremor
* Sharp spike = twitch
* Apply STFT or wavelet transform

#### 5. Oscillation Map Visualization

* Color-code body regions:

  * Frequency, amplitude
  * Phase coherence with neighbors
* Highlight discordant oscillations

---

## 🔗 References & Resources

* [Koide3 Calibration GitHub](https://github.com/koide3/hdl_graph_slam)
* [PointPWC-Net for Scene Flow](https://github.com/DylanWusee/PointPWC)
* [FlowNet3D for 3D Motion Estimation](https://github.com/xingyul/flownet3d)
* [SMPL Body Model](https://smpl.is.tue.mpg.de/)
* [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
* [LiftPose3D](https://github.com/akhilpm/LiftPose3D)
* [Kuramoto Oscillator Background](https://en.wikipedia.org/wiki/Kuramoto_model)
* [Eulerian Video Magnification (for inspiration)](https://people.csail.mit.edu/mrub/vidmag/)

---
