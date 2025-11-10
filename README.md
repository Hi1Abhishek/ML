

===============================
MACHINE LEARNING LAB ASSIGNMENTS
===============================

Assignment 1: Implementation of Python Basic Libraries
------------------------------------------------------
1. Implement basic functions of Statistics and Maths using Numpy and Scipy required for ML.
   a) Demonstrate usage of methods such as floor(), ceil(), sqrt(), isqrt(), gcd().
   b) Demonstrate usage of array attributes (ndim, shape, size) and methods (sum(), mean(), sort(), sin(), etc.).
   c) Demonstrate usage of det() and eig() for determinant and eigenvalues/eigenvectors of a matrix.
   d) Convert a 1D list into 2D and 3D matrices using NumPy.
   e) Use numpy.random.Generator to generate random matrices.
   f) Find the determinant of a matrix using SciPy.
   g) Find eigenvalues and eigenvectors of a matrix using SciPy.

Assignment 2: Implementation of Pandas and Matplotlib
------------------------------------------------------
1. Implement Python Libraries for ML application such as Pandas and Matplotlib.
   a) Create a Series using pandas and display it.
   b) Access the index and values of a Series.
   c) Compare a NumPy array with a Pandas Series.
   d) Define Series objects with individual indices.
   e) Access single value of a Series.
   f) Load datasets in a DataFrame using pandas.
   g) Use different methods in Matplotlib for visualization.

Assignment 3: Creation and Loading Different Types of Datasets
--------------------------------------------------------------
1. Create datasets using pandas (dictionary, list, NumPy array, or external files).
2. Load datasets using sklearn.datasets (Iris, Digits, Diabetes).
3. Load datasets in Google Colab.
4. Write a program to compute Mean, Median, Mode, Variance, Standard Deviation using datasets.
5. Demonstrate various data preprocessing techniques:
   - Reshaping data
   - Filtering data
   - Merging data
   - Handling missing values
   - Feature normalization (Min-Max, Scalar, etc.)

Assignment 4: Implementing Neural Networks
------------------------------------------
1. Design and implement the following neural networks:
   a) AND classifier for binary input.
   b) OR classifier for binary input.
   c) NAND classifier for binary input.
   d) XOR classifier for binary input and comment on its inability to classify data.
   e) Sequential dense neural network for Iris dataset (tune hyperparameters: learning rate, architecture, epochs).
   f) Sequential dense neural network for Diabetes dataset (tune hyperparameters).
   g) Sequential dense neural network for Heart dataset (tune hyperparameters).

Assignment 5: Find-S and Candidate Elimination Algorithm
--------------------------------------------------------
1. Implement the Find-S Algorithm on provided dataset to induce hypotheses.
2. Implement the Candidate Elimination Algorithm to list and refine hypotheses.

Assignment 6: Implementation of Naïve Bayes Classifier
------------------------------------------------------
1. Implement Gaussian Naïve Bayes Classifier using /kaggle/input/adult-dataset/adult.csv.
   - Handle missing values and categorical variables.
   - One-hot encode categorical features.
   - Perform feature scaling using RobustScaler.
   - Train using GaussianNB and evaluate with accuracy, confusion matrix, and classification report.
   - Check for overfitting and underfitting.
2. Implement Multinomial Naïve Bayes Classifier for document classification (Technology, Sports, Politics, Entertainment).
3. Implement Naïve Bayes Optimal Classifier on given dataset.

Assignment 7: Random Forest Algorithm
-------------------------------------
1. Implement Random Forest algorithm on MNIST dataset.
2. Implement Random Forest algorithm on Mental Health dataset.
3. Implement Random Forest algorithm on Customer Default Payment dataset:
   - Import and load dataset.
   - Drop ID column.
   - Analyze missing data.
   - Check target variable balance.
   - Downsample data.
   - One-hot encode independent variables.
   - Split dataset into train and test.
   - Train and evaluate model (accuracy & confusion matrix).
   - Perform hyperparameter tuning (n_estimators, max_depth, etc.) and display results.

Assignment 8: k-Nearest Neighbors (k-NN)
----------------------------------------
1. Predict Sugar of Diabetic Patient given BMI and Age using k=3.
2. Given dataset:

   Brightness | Saturation | Class
   ------------|-------------|------
   40 | 20 | Red
   50 | 50 | Blue
   60 | 90 | Blue
   10 | 25 | Red
   70 | 70 | Blue
   60 | 10 | Red
   25 | 80 | Blue

   Predict class for (Brightness=20, Saturation=35).
3. Apply k-NN on the Iris dataset and evaluate performance.

===============================
END OF ASSIGNMENTS
===============================


Excellent — before I begin: your PDF (`ML lab book final.pdf`) is **a complete practical guide for the Machine Learning (ML) course MSCOSDSC305**, and it includes **8 main assignments** that cover the full workflow of ML — from basic Python libraries to advanced algorithms like Neural Networks, Naïve Bayes, Random Forest, and k-NN.

Below is a **comprehensive explanation of all the concepts** from the PDF, grouped assignment-wise.
This explanation is written in an **exam + viva + understanding style** — so you can study it directly.

---

## 🧩 **Assignment 1 – Python Basic Libraries**

### Concepts:

You learn the **foundation of Python math and numerical operations** for ML.

#### 🔹 Libraries:

* **Math:** For basic mathematical operations.
* **NumPy:** For handling arrays, matrices, and numerical computations.
* **SciPy:** For advanced scientific calculations (determinants, eigenvalues, optimization, etc.).

#### 🔹 Key Concepts:

1. **Math functions:**

   * `floor(x)` → Largest integer ≤ x
   * `ceil(x)` → Smallest integer ≥ x
   * `sqrt(x)` → Square root
   * `isqrt(n)` → Integer square root
   * `gcd(a, b)` → Greatest common divisor

2. **NumPy arrays:**

   * Attributes: `ndim`, `shape`, `size`
   * Methods: `sum()`, `mean()`, `sort()`, `sin()`, etc.
   * Reshaping arrays: `reshape()` converts 1D → 2D or 3D.

3. **Matrix operations:**

   * Determinant: `np.linalg.det()`
   * Eigenvalues/vectors: `np.linalg.eig()`
   * Element-wise operations and matrix multiplication: `@` or `np.dot()`

4. **Random number generation:**

   * Using `np.random.default_rng(seed)` → reproducible random data.
   * Distributions: Uniform, Normal, Integers.

5. **SciPy advanced operations:**

   * `scipy.linalg.det()` → determinant
   * `scipy.linalg.eig()` → eigen decomposition.

📘 *Summary:*
This assignment helps you understand **how ML depends on numerical computing**, and how to manipulate matrices efficiently.

---

## 📊 **Assignment 2 – Pandas and Matplotlib**

### Concepts:

You learn **data manipulation** and **visualization**.

#### 🔹 Pandas:

* Data Structures:

  * `Series` → 1D labeled array.
  * `DataFrame` → 2D table (rows × columns).
* Creating Series and DataFrames.
* Accessing elements by **index** and **labels**.
* Comparison between NumPy arrays and Pandas Series.
* Loading datasets from **CSV, Excel**, etc.
* Methods: `head()`, `describe()`, `info()`, `groupby()`, `value_counts()`.

#### 🔹 Matplotlib:

* Visualization library to plot data.
* Common plots:

  * Line plot → `plt.plot()`
  * Bar chart → `plt.bar()`
  * Histogram → `plt.hist()`
  * Scatter plot → `plt.scatter()`
  * Pie chart → `plt.pie()`

📘 *Summary:*
These tools allow you to **load, explore, clean, and visualize data** before training models.

---

## 📁 **Assignment 3 – Creating and Loading Datasets**

### Concepts:

You explore **data creation, importing, and preprocessing**.

#### 🔹 Dataset creation:

* Using **Pandas**:

  * From dictionary, list of lists, or list of dictionaries.
* From **NumPy arrays** with column names.
* From **external files** like `.csv` or `.xlsx`.

#### 🔹 Loading datasets:

* From `sklearn.datasets`: `load_iris()`, `load_digits()`, `load_diabetes()`.
* Uploading datasets in **Google Colab** via `files.upload()`.

#### 🔹 Descriptive statistics:

Compute **Mean, Median, Mode, Variance, Standard Deviation** using `numpy` or `pandas`.

#### 🔹 Data preprocessing:

* **Reshaping** data → `reshape()`
* **Filtering** data → logical conditions
* **Merging** → `pd.merge()`
* **Handling missing values** → `fillna()`, `dropna()`
* **Normalization** → scale values between 0–1 (Min-Max, Z-score)

📘 *Summary:*
This forms the **data preparation step** in ML pipelines.

---

## 🧠 **Assignment 4 – Neural Networks**

### Concepts:

Covers the **fundamentals of Artificial Neural Networks (ANNs)**.

#### 🔹 Structure:

* **Input layer** → takes features.
* **Hidden layers** → perform transformations.
* **Output layer** → produces predictions.

#### 🔹 Types of Neural Networks:

* **Feedforward Neural Network:** Data flows in one direction.
* **Feedback / Recurrent Network:** Has loops (used in time series).

#### 🔹 Key Terms:

* **Neuron:** Small computational unit that applies an activation function.
* **Activation Functions:**

  * Sigmoid → converts input to range [0, 1]
  * Tanh → [-1, 1]
  * ReLU → max(0, x)
  * Softmax → output probabilities.
* **Epoch:** One complete training cycle.
* **Learning Rate:** Controls how much weights update.
* **Cost Function:** Measures prediction error (e.g., Mean Squared Error, Cross-Entropy).
* **Gradient Descent:** Optimization algorithm to reduce cost.
* **Weights & Biases:** Parameters learned during training.

#### 🔹 Neural network design tasks:

* Build models for **AND, OR, NAND, XOR** logic gates.
* Build classifiers for **Iris**, **Diabetes**, and **Heart** datasets.
* Perform **hyperparameter tuning** (learning rate, epochs, layers).

📘 *Summary:*
This assignment gives insight into how ML models learn through layers and weight updates.

---

## 🧮 **Assignment 5 – Find-S and Candidate Elimination Algorithms**

### Concepts:

These are **concept learning algorithms** used in classical AI.

#### 🔹 Find-S Algorithm:

* Starts with the **most specific hypothesis**.
* Only considers **positive examples**.
* Gradually generalizes hypothesis to include all positive instances.

Example:
Initial hypothesis = `{ϕ, ϕ, ϕ, ϕ}`
After positive example = `{Sunny, Warm, ?, ?, ?}`

#### 🔹 Candidate Elimination Algorithm (CEA):

* Maintains two sets:

  * **S (Specific boundary)** → most specific hypotheses.
  * **G (General boundary)** → most general hypotheses.
* Refines both S and G using positive and negative examples.
* The intersection gives the **Version Space** (set of consistent hypotheses).

📘 *Summary:*
These algorithms demonstrate **how a model “learns” concept boundaries** from examples — foundational to machine learning logic.

---

## 📈 **Assignment 6 – Naïve Bayes Classifiers**

### Concepts:

You study **probabilistic classification** using Bayes’ Theorem.

#### 🔹 Bayes Theorem:

[
P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}
]

* **P(A|B):** Probability of A given B (posterior)
* **P(B|A):** Likelihood
* **P(A):** Prior probability
* **P(B):** Evidence

#### 🔹 Types of Naïve Bayes:

1. **Gaussian NB** → For continuous numeric data (uses normal distribution).
2. **Multinomial NB** → For text classification (word counts).
3. **Bernoulli NB** → For binary features (spam filtering).

#### 🔹 Key Steps:

* Handle missing values
* One-hot encoding
* Feature scaling
* Model training with `GaussianNB()`
* Evaluate with accuracy, confusion matrix, precision, recall, F1-score.

📘 *Summary:*
Naïve Bayes is a **fast and effective classifier** based on probability and independence assumptions.

---

## 🌲 **Assignment 7 – Random Forest Algorithm**

### Concepts:

You learn **ensemble learning** using multiple decision trees.

#### 🔹 Ensemble Idea:

Combines results from many models → improves accuracy.

#### 🔹 Working Steps:

1. Take random samples of dataset (bootstrapping).
2. Build decision trees on each subset.
3. Combine predictions using **majority voting** or **averaging**.

#### 🔹 Types:

* **Bagging:** Random sampling + independent trees.
* **Boosting:** Sequential training (each new model corrects errors of previous).

#### 🔹 Key Parameters:

* `n_estimators` → number of trees.
* `max_depth` → tree depth.
* `min_samples_split` → minimum data to split a node.
* `random_state` → reproducibility.

#### 🔹 Applications:

Used for classification (like credit default prediction) and regression tasks.

📘 *Summary:*
Random Forest reduces overfitting and provides **high accuracy and robustness**.

---

## ⚙️ **Assignment 8 – k-Nearest Neighbors (k-NN)**

### Concepts:

A **distance-based supervised algorithm** for classification or regression.

#### 🔹 How it Works:

1. Choose **K** (number of neighbors).
2. Calculate distance (Euclidean, Manhattan, or Minkowski).
3. Select K nearest points.
4. Predict based on **majority class** (classification) or **average** (regression).

#### 🔹 Distance Formula:

[
\text{Euclidean Distance} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
]

#### 🔹 Key Characteristics:

* **Lazy learner** → No training phase.
* **Non-parametric** → No assumptions about data distribution.

#### 🔹 Pros:

* Simple, intuitive.
* Works well for small datasets.

#### 🔹 Cons:

* Slow for large datasets.
* Sensitive to scale and noise.

📘 *Summary:*
k-NN predicts output by “voting” among the nearest data points.

---

## 🎓 **Overall Learning Outcomes**

By completing all assignments, you will:
✅ Understand and implement core ML algorithms.
✅ Work with real-world datasets using Python.
✅ Learn data preprocessing and model evaluation.
✅ Be ready for applied ML projects.

---

Would you like me to generate this **full concept explanation as a formatted PDF or .docx (Word)** file for study or submission?

