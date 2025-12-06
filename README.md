Machine Learning Pipeline & Model Selection Analysis
Academic Project | Ben-Gurion University of the Negev

📌 Project Overview
Developed within the framework of the Machine Learning curriculum at Ben-Gurion University, this project translates theoretical algorithmic concepts into a practical, end-to-end data processing pipeline.

The core objective of this repository is not merely implementation, but comparative analysis. The project focuses on evaluating different machine learning methodologies learned in class to determine the optimal strategy for a given high-dimensional dataset. It demonstrates the decision-making process behind selecting the right model based on data characteristics, bias-variance trade-offs, and performance metrics.

🚀 Key Technical Components
The workflow is modular, simulating a production-grade environment from data acquisition to validation:

Algorithmic Decision Making: The system implements multiple classification approaches to validate theoretical assumptions. A key part of the work involved analyzing the strengths and weaknesses of each method to select the most robust solver for the specific problem domain.

Advanced Feature Engineering (TF-IDF): To handle unstructured or noisy data, I implemented TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This technique transforms raw inputs into weighted feature vectors, emphasizing unique signals while filtering out redundancy.

Dimensionality Reduction (PCA): Addressing the "Curse of Dimensionality," the project utilizes Principal Component Analysis (PCA).

Optimization: Compressed the feature space to improve training efficiency and reduce overfitting.

Visualization: Projected complex data into 3D space (pca_3d_dataset.csv) to visually inspect cluster separability and decision boundaries.

🛠️ Technology Stack
Core: Python 3.x

Data Science: Pandas, NumPy

Machine Learning: Scikit-Learn (PCA, TF-IDF, Model Validators)

Workflow: Git, Data Serialization
