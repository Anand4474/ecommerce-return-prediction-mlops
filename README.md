# E-Commerce Product Return Prediction — MLOps CI/CD

A complete assignment project using the supplied `Section7_Q4_Ecommerce_Product_Return_CICD.csv` dataset. It trains two classifiers, selects the best model using test ROC-AUC, serves predictions through FastAPI, tests the application, builds a Docker image, and publishes the image to GitHub Container Registry (GHCR) from GitHub Actions.

## 1. Dataset and target

- Dataset file: `data/ecommerce_returns.csv` (the exact CSV supplied with the question)
- Target: `Return_Status` (`Returned` = 1, `Not Returned` = 0)
- `Order_ID` is removed because it is an identifier, not a useful predictive feature.
- Missing numeric values are filled with the training-set median.
- Missing categorical values are filled with the most frequent training-set value.
- Categorical values are one-hot encoded; numeric values are standardized.
- Engineered features:
  - `Delivery_Delay_Days = max(Actual_Delivery_Days - Expected_Delivery_Days, 0)`
  - `Previous_Return_Frequency = Previous_Returns / Previous_Orders` (safe denominator; capped to [0, 1])
- Models compared: Logistic Regression and Random Forest.
- Model selection: highest ROC-AUC on a held-out stratified test set (20%).

**Important:** `Damaged_Package_Flag` is retained because it is an order-level input feature available at prediction time. Do not send `Return_Status` or `Order_ID` to the prediction endpoint.

## 2. Run locally (Windows / VS Code)

Install Python 3.11 (recommended), Git, and Docker Desktop.

Open PowerShell in the extracted project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
pytest -q
uvicorn src.api:app --reload
```

Open:
- API home: http://127.0.0.1:8000/
- Swagger interactive API: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

Example prediction request (Swagger `POST /predict`):

```json
{
  "Product_Category": "Electronics",
  "Product_Price_INR": 24999,
  "Discount_Percent": 15,
  "Expected_Delivery_Days": 4,
  "Actual_Delivery_Days": 6,
  "Customer_Rating_PreOrder": 3.5,
  "Previous_Returns": 2,
  "Previous_Orders": 8,
  "Payment_Method": "UPI",
  "Damaged_Package_Flag": "No",
  "Package_Size": "Medium",
  "Promotion_Active": "Yes"
}
```

The API returns a predicted class, estimated return probability, and a simple Low/Medium/High risk label. A probability is a model estimate, not a guarantee.

## 3. Google Colab — every step

1. Open https://colab.research.google.com/ and select **New notebook**.
2. Upload this ZIP to the Colab session using the Files (folder) panel, or upload the CSV separately.
3. Run this cell to extract the project ZIP (change the filename if Colab assigned a different name):

```python
from google.colab import files
uploaded = files.upload()  # select ecommerce_return_mlops.zip
```

4. Extract and enter the project directory:

```python
import zipfile, os
zip_name = next(name for name in uploaded if name.endswith(".zip"))
with zipfile.ZipFile(zip_name, "r") as z:
    z.extractall("/content")
os.chdir("/content/ecommerce_return_mlops")
print(os.getcwd())
```

5. Install dependencies:

```python
!pip install -q -r requirements.txt
```

6. Check that the dataset is present:

```python
import pandas as pd
df = pd.read_csv("data/ecommerce_returns.csv")
print(df.shape)
display(df.head())
print(df.isna().sum())
print(df["Return_Status"].value_counts())
```

7. Train and compare both models:

```python
!python -m src.train
```

8. Inspect evaluation metrics and confirm that the model artifact exists:

```python
import json, os
print(open("artifacts/metrics.json").read())
print("Model saved:", os.path.exists("artifacts/best_model.joblib"))
```

9. Run automated tests:

```python
!pytest -q
```

10. Start the API inside Colab:

```python
import subprocess, time
api_process = subprocess.Popen(
    ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT
)
time.sleep(3)
print("API process running:", api_process.poll() is None)
```

11. Test `/health` and `/predict` from the same Colab runtime:

```python
import requests
print(requests.get("http://127.0.0.1:8000/health").json())

sample = {
    "Product_Category": "Electronics",
    "Product_Price_INR": 24999,
    "Discount_Percent": 15,
    "Expected_Delivery_Days": 4,
    "Actual_Delivery_Days": 6,
    "Customer_Rating_PreOrder": 3.5,
    "Previous_Returns": 2,
    "Previous_Orders": 8,
    "Payment_Method": "UPI",
    "Damaged_Package_Flag": "No",
    "Package_Size": "Medium",
    "Promotion_Active": "Yes"
}
response = requests.post("http://127.0.0.1:8000/predict", json=sample)
print(response.status_code, response.json())
```

12. When finished, stop the API:

```python
api_process.terminate()
```

**Colab note:** Colab is excellent for preprocessing, model training, evaluation, and API smoke tests. The GitHub Actions workflow and GHCR publishing run on GitHub after you push the repository; Colab does not replace GitHub Actions. A Colab runtime is temporary, so download or push any artifacts you need to keep.

## 4. GitHub repository and automated CI/CD

Create a new empty GitHub repository, for example `ecommerce-return-mlops`. Do not add a README from GitHub if you already have this project README.

In a terminal opened in this project folder:

```bash
git init
git add .
git commit -m "Initial e-commerce return prediction MLOps project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ecommerce-return-mlops.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. The workflow at `.github/workflows/ci-cd.yml` starts automatically on a push to `main` and on pull requests.

### What the workflow does

1. Checks out the repository.
2. Sets up Python 3.11 and installs dependencies.
3. Trains the model only if the model artifact is absent.
4. Compiles Python source and test files.
5. Runs pytest.
6. Builds the Docker image.
7. On pushes to `main`, authenticates to GHCR using the automatic `GITHUB_TOKEN`.
8. Publishes `ghcr.io/OWNER/REPOSITORY:latest` and a commit-SHA tag.

If package publishing is denied, open GitHub repository **Settings → Actions → General → Workflow permissions** and allow read and write permissions. The workflow also declares `packages: write`. No personal access token is needed for this workflow.

### Verify workflow

- Open the GitHub repository → **Actions**.
- Open the latest **ML CI-CD** run.
- Confirm all steps are green.
- Open the repository owner's GitHub profile → **Packages** and locate the container package.
- If the package is private, change package visibility only if you intend it to be public.

## 5. Docker build and run

After training locally:

```bash
docker build -t ecommerce-return-api:latest .
docker run --rm -p 8000:8000 ecommerce-return-api:latest
```

Visit http://127.0.0.1:8000/docs.

To pull the GHCR image after the GitHub Actions workflow succeeds:

```bash
docker pull ghcr.io/YOUR-USERNAME/ecommerce-return-mlops:latest
docker run --rm -p 8000:8000 ghcr.io/YOUR-USERNAME/ecommerce-return-mlops:latest
```

Use the exact lowercase owner/repository path shown in GitHub Packages. If the package is private, authenticate with `docker login ghcr.io` first.

## 6. Evidence/screenshots to submit

Capture your own screenshots from your actual run; do not submit fabricated outputs.

1. Dataset loaded in Colab: shape, head, null counts.
2. Feature engineering/training cell completed.
3. Metrics JSON showing Logistic Regression and Random Forest evaluation.
4. `pytest -q` showing tests passed.
5. FastAPI `/docs` page and successful `/health`.
6. Successful `/predict` response.
7. GitHub repository showing project files and CSV.
8. GitHub Actions run showing successful validation, tests, and Docker build/push.
9. GitHub Packages page showing the published container.
10. Docker container running and API `/docs` accessible.

## 7. Viva / explanation

- **What is MLOps?** Practices that combine ML development and operations to make training, testing, packaging, deployment, and monitoring repeatable.
- **Why imputation?** Most estimators cannot accept missing values; imputation preserves rows and learns fill values from training data only.
- **Why one-hot encoding?** It converts categories into numeric indicator columns without inventing an ordinal relationship.
- **Why Logistic Regression and Random Forest?** They provide two different classification approaches: a linear baseline and a non-linear ensemble.
- **Why ROC-AUC?** It measures ranking quality across classification thresholds and is useful when class balance may be uneven.
- **Why stratified split?** It preserves approximately the same target-class proportions in train and test sets.
- **What is FastAPI?** A Python framework for serving model predictions through HTTP endpoints.
- **What is Docker?** A packaging technology that bundles application code and runtime dependencies into a reproducible image.
- **What is CI/CD?** Continuous Integration validates code and tests; Continuous Delivery/Deployment packages and publishes the application automatically.
- **What is GHCR?** GitHub Container Registry, used to store Docker/OCI images.
- **Why remove Order_ID?** It is a unique identifier and is not a generalizable customer/order behavior signal.
- **What is data leakage?** Using information unavailable at prediction time or target-derived information during training. The target and order identifier are removed from model inputs.

## 8. Project structure

```text
ecommerce_return_mlops/
├── .github/workflows/ci-cd.yml
├── artifacts/                  # trained model and metrics generated by training
├── data/ecommerce_returns.csv  # supplied dataset
├── src/
│   ├── features.py
│   ├── train.py
│   └── api.py
├── tests/
│   ├── test_features.py
│   └── test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```
