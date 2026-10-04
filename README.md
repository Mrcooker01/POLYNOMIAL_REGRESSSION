# Polynomial Regression

A Machine Learning implementation of **Polynomial Regression** using `scikit-learn` to predict **CO2 emissions** of vehicles based on their **engine size**.

---

## 📌 About the Project

This project is part of a Machine Learning exercise that uses a **degree-2 polynomial regression** model to capture the non-linear relationship between engine size and CO2 emissions.

**Dataset:** `FuelConsumption.csv` — contains fuel consumption ratings and estimated CO2 emissions for light-duty vehicles in Canada.

---

## 🎯 Objectives

- Implement Polynomial Regression using `PolynomialFeatures`
- Train a linear model on polynomial-transformed features
- Evaluate the model with statistical metrics
- Visualize real data and the regression curve

---

## 🛠 Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Load and process the dataset |
| `numpy` | Numerical operations and arrays |
| `scikit-learn` | PolynomialFeatures and LinearRegression models |
| `matplotlib` | Plotting charts |
| `pylab` | Plotting charts |

---

## 📂 Project Structure

```

polynomial_regression/
│
├── polynomial.py # Main script
├── FuelConsumption.csv # Dataset (download separately)
├── README.md # This file
└── .gitignore

```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Mrcooker01/POLYNOMIAL_REGRESSION.git
cd POLYNOMIAL_REGRESSION
```

2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

3. Download the dataset

Download FuelConsumption.csv and place it in the project root:

```bash
curl -o FuelConsumption.csv "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
```

4. Run the script

```bash
python polynomial.py
```

---

🧠 How the Model Works

1. Load the data

```python
loader = pd.read_csv("FuelConsumption.csv")
observer = loader[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
```

2. Split into train and test sets

```python
msk = np.random.rand(len(loader)) < 0.8
train = observer[msk]
test = observer[~msk]
```

3. Transform features into polynomial form

```python
poly = PolynomialFeatures(degree=2)
poly_train_x = poly.fit_transform(train_x)
```

4. Train the linear model

```python
lin = linear_model.LinearRegression()
lin.fit(poly_train_x, train_y)
```

5. Model equation

The final model equation is:

```
y = θ₀ + θ₁·x + θ₂·x²
```

Where x is engine size and y is CO2 emissions.

---

📊 Evaluation Metrics

The model is evaluated using the following metrics:

Metric Description
MAE (Mean Absolute Error) Average of absolute errors
RSS (Residual Sum of Squares) Sum of squared residuals
R² Score Coefficient of determination (closer to 1 is better)

---

📈 Sample Output

· Scatter plot of actual data (blue = train, green = test)
· Polynomial regression curve (degree-2)
· Display of model coefficients and intercept
· Evaluation metrics printed in the terminal

---

🔍 Why Polynomial Regression?

Simple linear regression only models linear relationships, but the relationship between engine size and CO2 emissions may be non-linear. By adding higher-degree terms (x², x³, etc.), the model can better follow the curvature in the data and achieve higher accuracy.

