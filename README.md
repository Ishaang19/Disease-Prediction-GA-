# 🧬 GA-DFS: Genetic Algorithm-Aided Deep Feature Selection for Chest X-Ray Disease Classification

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

## 📌 Project Overview

This project is about building an intelligent system that classifies **chest X-ray diseases** using deep learning. We explored multiple ANN architectures for **feature extraction** and enhanced model performance using a **Genetic Algorithm (GA)** for optimal feature selection.

The dataset used in this project is the [Kaggle Chest X-ray Dataset](https://www.kaggle.com/datasets/xhlulu/vinbigdata-chest-xray-resized-png-256x256/data), which contains X-ray images categorized as normal or showing signs of different respiratory diseases.
This dataset is a subset mini version of the original one [Kaggle Chest X-ray Dataset](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
The key objective was to identify the most relevant features and eliminate redundancy, resulting in **faster**, **more accurate**, and **efficient** predictions.

---

## 🤖 Models Explored

We experimented with 4 major CNN-based architectures:
- ✅ **DenseNet121**
- ✅ **VGG16**
- ✅ **ResNet50**
- ✅ **EfficientNetB3**

Out of these, **VGG16** and **DenseNet121** were further combined with a **Genetic Algorithm-based feature selection** approach to create 2 more powerful variants. These variants used an **Artificial Neural Network (ANN)** for classification after GA feature selection, not CNN.

> 🔢 **Total Models Evaluated:** 6  
> 🏆 **Best Performer:** `DenseNet + GA`

Detailed metrics and evaluation visuals are available in the `results/` folder.

---

## 📂 Repository Structure
├── Documents/ # Project diagrams and designs
│ ├── Data Flow Diagram.png
│ ├── GA Flowchart.png
│ ├── Model Architecture.png
│ └── Readme
│
├── results/ # Output metrics and visualizations
│ ├── Metrics/
│ └── Results metrics graph.png
│
├── DenseNet121.py
├── DenseNet121 + GeneticAlgo.py
├── EfficientNetB3.py
├── ResNet50.py
├── VGG16 + GeneticAlgo.py
├── VGGGA.py
│
├── LICENSE
├── README.md



---

## 🧬 Genetic Algorithm Integration

- We used GA to **select the most relevant features** from the deep features extracted by CNNs.
- These selected features were passed through a lightweight **ANN-based classifier** for final disease prediction.
- This approach significantly improved the model's interpretability and reduced computational complexity.

---

## 👨‍💻 Team Members

This was a group project carried out as part of our B.Tech curriculum at **Guru Gobind Singh Indraprastha University (GGSIPU)**.

- **Ishaan Garg**  
  [LinkedIn](https://www.linkedin.com/in/gargishaan/)

- **Manayu**  
  [LinkedIn](https://www.linkedin.com/in/manayu-kaushik-b32418332/)

- **Ansh**  
  [LinkedIn](https://www.linkedin.com/in/ansh-gaur-35b3b0314/)

---

## 📄 Research Paper

Our work has been accepted at a reputed international conference:

> **"GA-DFS: A Genetic Algorithm-Aided Deep Feature Selection Framework for Chest X-Ray Disease Classification"**  
> Presented at **ICDAM 2025** (to be published in proceedings)

📌 [Link to paper or conference site](#)

---

## 🚀 How to Run

# Install dependencies
# Run a model (example: DenseNet + GA)

## 📜 License
This project is licensed under the MIT License.


### 📫 Feel free to fork, contribute, or reach out for collaborations!
