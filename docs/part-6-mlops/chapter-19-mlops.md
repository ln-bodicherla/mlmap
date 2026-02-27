# Chapter 19: MLOps and Experiment Management

## Learning Objectives

By the end of this chapter, you will be able to:

1. Articulate the MLOps lifecycle and identify the maturity level of an organization's ML operations.
2. Implement experiment tracking using MLflow, Weights & Biases, and DVC.
3. Design and orchestrate ML pipelines with Airflow, Prefect, Dagster, Kubeflow, and Metaflow.
4. Leverage Ray for distributed training, hyperparameter tuning, and model serving.
5. Build monitoring systems that detect data drift, concept drift, and model performance degradation.
6. Construct CI/CD pipelines for ML that include automated testing, model validation, and deployment strategies.
7. Containerize ML workloads with Docker and orchestrate them on Kubernetes with GPU support.

---

## 19.1 The MLOps Lifecycle

Machine learning in production is fundamentally different from machine learning in research. A model that achieves state-of-the-art accuracy on a benchmark means nothing if it cannot be deployed, monitored, and maintained reliably. MLOps---the discipline of applying DevOps principles to machine learning systems---exists precisely to bridge this gap.

### 19.1.1 From Development to Production

The MLOps lifecycle encompasses five interconnected phases:

**Development.** Data scientists explore data, engineer features, and train candidate models. This phase is inherently experimental: dozens or hundreds of configurations are tried, most are discarded, and the few promising ones advance. The critical requirement here is reproducibility---the ability to recreate any experiment, with the same data, code, and hyperparameters, months or years later.

**Staging.** Before a model reaches users, it must pass through a staging environment that mirrors production as closely as possible. Here, integration tests verify that the model can receive real-shaped inputs, produce outputs in the expected format, and meet latency and throughput requirements. Shadow deployments---where the candidate model processes live traffic without serving its predictions to users---reveal issues that synthetic tests miss.

**Production.** The model is deployed behind an endpoint that serves predictions. This is where the rubber meets the road: the model must handle variable traffic, degrade gracefully under load, and recover from failures without human intervention. Blue-green deployments, canary releases, and A/B tests are standard strategies for managing the risk of new model versions.

**Monitoring.** Once in production, the model is subject to forces that erode its performance. The distribution of incoming data shifts. User behavior changes. Upstream data pipelines break silently. Monitoring systems must detect these problems before they cause business impact, and they must do so with a low false-positive rate lest alert fatigue set in.

**Retraining.** When monitoring signals cross predefined thresholds, the model must be retrained---ideally through an automated pipeline that triggers data collection, feature engineering, training, evaluation, and deployment without human intervention. The cycle then begins anew.

These five phases form a continuous loop, not a linear sequence. The best MLOps systems treat this loop as infrastructure, not as a set of ad hoc scripts cobbled together by whichever engineer happens to be available.

### 19.1.2 MLOps Maturity Levels

Google's MLOps maturity model (Google, 2021) provides a useful framework for assessing where an organization stands:

**Level 0: Manual Process.** Everything is done by hand. Data scientists train models in notebooks, export them as pickle files, and hand them off to engineers for deployment. There is no version control for data or models, no automated testing, and no monitoring. This is where most organizations begin, and many never leave.

**Level 1: ML Pipeline Automation.** The training pipeline is automated end-to-end. Data ingestion, feature engineering, training, evaluation, and model registration all happen without manual intervention. However, the pipeline itself is still built and updated by hand. Continuous training is achieved---when the pipeline runs on a schedule or in response to a trigger, a new model is produced automatically.

**Level 2: CI/CD Pipeline Automation.** Not only is the training pipeline automated, but so is the process of building, testing, and deploying the pipeline itself. Changes to feature engineering code, training logic, or pipeline configuration are tested in CI, validated in staging, and rolled out to production through CD. This level requires significant investment in testing infrastructure, but it pays dividends in velocity and reliability.

**Level 3: Full MLOps.** The system is self-healing. Monitoring detects performance degradation, triggers automated retraining, evaluates the new model against the current production model, and promotes it if it passes all quality gates. Human intervention is required only for novel failure modes or strategic decisions about which models to develop. Few organizations reach this level, but it represents the aspiration of the discipline.

### 19.1.3 Technical Debt in ML Systems

In their landmark paper, Sculley et al. (2015) argued that machine learning systems have a special capacity for accumulating technical debt. The ML code---the part that trains and evaluates models---is typically a small fraction of the total system. Surrounding it is a vast infrastructure of data collection, feature extraction, configuration, serving, and monitoring code. The authors identified several categories of ML-specific technical debt:

**Entanglement.** Changing anything changes everything. CACE (Changing Anything Changes Everything) is the ML analog of spaghetti code. If you change one input feature, the optimal values of all other features, all hyperparameters, and all downstream models may shift.

**Hidden Feedback Loops.** A model that influences the data it is trained on creates a feedback loop that is difficult to detect and dangerous to ignore. Recommendation systems are the canonical example: the model recommends items, users interact with the recommendations, and those interactions become training data for the next version of the model.

**Undeclared Consumers.** When a model's predictions are consumed by other systems without explicit contracts, changes to the model can have cascading effects that no one anticipated. This is the ML analog of an unstable API.

**Data Dependencies.** ML systems depend on data that is produced by other systems, often outside the ML team's control. These dependencies are harder to track than code dependencies because they are not captured by package managers or build systems.

**Pipeline Jungles.** As ML systems evolve, the data pipelines that feed them grow organically, producing a tangle of joins, scrapes, and transformations that is difficult to understand, test, or modify.

The MLOps tools and practices described in this chapter are, in large part, responses to these forms of technical debt. Experiment tracking combats entanglement by making it possible to compare the effects of changes. Pipeline orchestration tames pipeline jungles. Monitoring detects hidden feedback loops. CI/CD enforces contracts between producers and consumers.

---

## 19.2 Experiment Tracking with MLflow

MLflow (Zaharia et al., 2018) is an open-source platform for managing the ML lifecycle. It provides four components: Tracking (for experiment logging), Projects (for reproducible runs), Models (for packaging and deployment), and a Model Registry (for managing model versions). We focus here on Tracking and the Model Registry, as they are the most widely adopted.

### 19.2.1 Architecture

MLflow Tracking is organized around three concepts:

- **Experiments** are named collections of runs that correspond to a particular modeling task (e.g., "customer-churn-prediction").
- **Runs** are individual executions of training code. Each run records parameters, metrics, artifacts, and metadata.
- **The Tracking Server** is an HTTP service that stores run data. It can use a file store (for local development), a SQL database (for teams), or a remote tracking server (for enterprise deployments). Artifact storage is typically backed by S3, GCS, Azure Blob, or a shared filesystem.

### 19.2.2 Complete Example

The following example demonstrates the full MLflow workflow: configuring the tracking server, logging an experiment, registering a model, and transitioning it through stages.

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import load_breast_cancer
import numpy as np

# Configure the tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast-cancer-classification")

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Define hyperparameters
params = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "min_samples_split": 10,
}

# Start a run
with mlflow.start_run(run_name="gbm-v1") as run:
    # Log parameters
    mlflow.log_params(params)

    # Train model
    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    mlflow.log_metrics(metrics)

    # Log the model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="BreastCancerClassifier",
    )

    # Log additional artifacts
    feature_importance = np.argsort(model.feature_importances_)[::-1][:10]
    with open("top_features.txt", "w") as f:
        for idx in feature_importance:
            f.write(f"{data.feature_names[idx]}: {model.feature_importances_[idx]:.4f}\n")
    mlflow.log_artifact("top_features.txt")

    print(f"Run ID: {run.info.run_id}")
    print(f"Metrics: {metrics}")
```

### 19.2.3 Model Registry and Stages

The Model Registry provides a centralized hub for managing model versions. Each registered model can have multiple versions, and each version can be in one of four stages: `None`, `Staging`, `Production`, or `Archived`.

```python
client = MlflowClient()

# Transition model to Staging
client.transition_model_version_stage(
    name="BreastCancerClassifier",
    version=1,
    stage="Staging",
)

# After validation, promote to Production
client.transition_model_version_stage(
    name="BreastCancerClassifier",
    version=1,
    stage="Production",
)

# Load the production model for serving
production_model = mlflow.pyfunc.load_model(
    model_uri="models:/BreastCancerClassifier/Production"
)

# Make predictions using the pyfunc interface
predictions = production_model.predict(X_test)
```

### 19.2.4 The Pyfunc Interface and REST API

MLflow's `pyfunc` interface provides a unified API for loading and serving models regardless of their underlying framework. Any model logged with MLflow---whether scikit-learn, PyTorch, TensorFlow, or a custom framework---can be loaded as a `pyfunc` and invoked with the same `predict()` method.

The MLflow Tracking Server also exposes a REST API that allows programmatic interaction from any language or tool:

```bash
# Search for runs with accuracy > 0.95
curl -X POST http://localhost:5000/api/2.0/mlflow/runs/search \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_ids": ["1"],
    "filter": "metrics.accuracy > 0.95",
    "order_by": ["metrics.f1_score DESC"],
    "max_results": 10
  }'
```

This REST API is particularly useful for building custom dashboards, integrating with CI/CD systems, or querying experiment results from non-Python environments.

---

## 19.3 Weights & Biases

Weights & Biases (W&B) (Biewald, 2020) is a commercial experiment tracking platform that has gained wide adoption in the research community. While MLflow is the standard for open-source, self-hosted experiment tracking, W&B offers a richer user interface, more sophisticated visualization, and features like hyperparameter sweeps and collaborative reports that go beyond what MLflow provides out of the box.

### 19.3.1 Core Concepts

W&B is organized around four primitives:

- **Runs** are individual experiments, analogous to MLflow runs. Each run records configuration, metrics, system metrics (GPU utilization, memory usage), code, and artifacts.
- **Artifacts** are versioned datasets, models, or other files. W&B tracks the lineage of artifacts---which run produced them and which runs consumed them---enabling full reproducibility.
- **Sweeps** are hyperparameter searches. W&B supports grid search, random search, and Bayesian optimization (using a Tree-structured Parzen Estimator). Sweeps can be distributed across multiple machines with a single command.
- **Tables** are interactive data visualizations that allow you to log, query, and visualize structured data (e.g., model predictions alongside ground truth labels).

### 19.3.2 Integration with PyTorch and Hugging Face

W&B integrates deeply with both PyTorch and the Hugging Face Transformers library:

```python
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

# Initialize a W&B run
wandb.init(
    project="sentiment-analysis",
    config={
        "model_name": "distilbert-base-uncased",
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 16,
        "warmup_steps": 500,
        "weight_decay": 0.01,
    },
)
config = wandb.config

# Load and tokenize data
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    config.model_name, num_labels=2
)

# Training arguments with W&B integration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=config.epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    warmup_steps=config.warmup_steps,
    weight_decay=config.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="wandb",  # Enable W&B logging
    load_best_model_at_end=True,
)

# The Trainer automatically logs to W&B
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()
wandb.finish()
```

### 19.3.3 Hyperparameter Sweeps

W&B Sweeps automate hyperparameter optimization with a declarative configuration:

```yaml
# sweep_config.yaml
program: train.py
method: bayes  # bayes, grid, or random
metric:
  name: eval/f1
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-3
  batch_size:
    values: [8, 16, 32]
  warmup_steps:
    distribution: int_uniform
    min: 0
    max: 1000
  weight_decay:
    distribution: uniform
    min: 0.0
    max: 0.3
early_terminate:
  type: hyperband
  min_iter: 3
```

```python
import wandb

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="sentiment-analysis")

# Run the sweep agent (can run on multiple machines)
wandb.agent(sweep_id, function=train, count=50)
```

The Bayesian optimization strategy uses results from completed runs to model the relationship between hyperparameters and the objective metric, focusing exploration on the most promising regions of the search space. The `early_terminate` configuration implements Hyperband-style early stopping, killing underperforming runs to save compute.

### 19.3.4 Advantages Over MLflow

W&B offers several advantages over MLflow for teams that can accept a managed service:

1. **Richer Visualization.** W&B's UI supports custom panels, parallel coordinates plots, parameter importance analysis, and interactive data tables out of the box. MLflow's UI, while functional, is comparatively spartan.
2. **System Metrics.** W&B automatically logs GPU utilization, memory usage, CPU load, and other system metrics, which is invaluable for debugging performance bottlenecks.
3. **Collaborative Reports.** W&B Reports allow teams to create interactive documents that combine text, charts, and live experiment data---useful for model review meetings and stakeholder communication.
4. **Sweeps.** While MLflow requires external tools (e.g., Optuna, Hyperopt) for hyperparameter optimization, W&B provides built-in sweep functionality with distributed execution.

The primary disadvantage is that W&B is a commercial product. While it offers a free tier for individuals and academics, enterprises must pay for the service or host it on-premises. MLflow, being fully open-source, can be self-hosted at no cost beyond infrastructure.

---

## 19.4 DVC (Data Version Control)

DVC (Data Version Control) extends Git to handle the large files that are ubiquitous in ML: datasets, model weights, and intermediate artifacts. While Git tracks code changes efficiently, it was never designed for binary files measured in gigabytes or terabytes. DVC fills this gap with a git-like interface that data scientists already understand.

### 19.4.1 Core Concepts

DVC works by replacing large files with small `.dvc` metadata files that are tracked by Git. The actual data is stored in a configured remote (S3, GCS, Azure Blob, HDFS, or even a local directory). When a collaborator clones the repository and runs `dvc pull`, DVC downloads the data from the remote.

```bash
# Initialize DVC in an existing Git repository
dvc init

# Configure a remote storage
dvc remote add -d myremote s3://my-bucket/dvc-store

# Track a large dataset
dvc add data/training_data.parquet

# The above creates data/training_data.parquet.dvc and adds the data
# file to .gitignore. Commit the .dvc file to Git:
git add data/training_data.parquet.dvc data/.gitignore
git commit -m "Track training data with DVC"

# Push the data to remote storage
dvc push
```

### 19.4.2 Pipeline Stages

DVC pipelines define reproducible ML workflows in a `dvc.yaml` file:

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/
    outs:
      - data/prepared/
    params:
      - prepare.split_ratio
      - prepare.seed

  featurize:
    cmd: python src/featurize.py
    deps:
      - src/featurize.py
      - data/prepared/
    outs:
      - data/features/
    params:
      - featurize.max_features
      - featurize.ngram_range

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/features/
    outs:
      - models/model.pkl
    params:
      - train.n_estimators
      - train.learning_rate
    metrics:
      - metrics/scores.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/features/
    metrics:
      - metrics/eval.json:
          cache: false
    plots:
      - metrics/confusion_matrix.csv:
          x: predicted
          y: actual
```

```bash
# Run the full pipeline
dvc repro

# Compare metrics across Git branches
dvc metrics diff

# Run experiments with different parameters
dvc exp run --set-param train.n_estimators=500
dvc exp run --set-param train.learning_rate=0.05

# Compare experiments
dvc exp show
```

DVC's strength lies in its integration with Git. Every experiment is a Git commit (or a lightweight Git reference), which means that you can use all of Git's branching, merging, and diffing capabilities to manage your experiments. For teams that are already Git-fluent, this is a lower cognitive overhead than learning an entirely new experiment tracking system.

---

## 19.5 Workflow Orchestration with Apache Airflow

Apache Airflow is the most widely adopted workflow orchestration platform in data engineering and, by extension, in MLOps. Originally developed at Airbnb in 2014, it provides a Python-based framework for defining, scheduling, and monitoring workflows.

### 19.5.1 Core Concepts

**DAGs (Directed Acyclic Graphs)** are the central abstraction. A DAG is a collection of tasks with dependencies between them. The acyclicity constraint ensures that the workflow has a well-defined execution order and will terminate.

**Operators** are templates for tasks. Airflow provides built-in operators for common operations:

- `PythonOperator`: executes a Python callable.
- `BashOperator`: executes a bash command.
- `KubernetesPodOperator`: runs a Docker container in a Kubernetes pod.
- `S3ToRedshiftOperator`, `BigQueryOperator`, etc.: interact with specific services.

**Sensors** are operators that wait for a condition to be met before proceeding. `S3KeySensor` waits for a file to appear in S3; `ExternalTaskSensor` waits for a task in another DAG to complete.

**XCom (Cross-Communication)** allows tasks to exchange small amounts of data. A task can push a value to XCom, and downstream tasks can pull it. XCom is intended for metadata (run IDs, file paths, metric values), not for large data transfers.

**Connections and Hooks** manage credentials and API clients for external services. Connections are stored in Airflow's metadata database (or a secrets backend like AWS Secrets Manager), and Hooks provide a consistent interface for interacting with the connected service.

### 19.5.2 ML Pipeline DAG Example

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.s3 import S3CopyObjectOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)
from airflow.sensors.s3_key_sensor import S3KeySensor
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["ml-alerts@company.com"],
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=4),
}

dag = DAG(
    "ml_training_pipeline",
    default_args=default_args,
    description="End-to-end ML training pipeline",
    schedule_interval="0 6 * * 1",  # Every Monday at 6 AM
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "training"],
)

# Task 1: Wait for new data
wait_for_data = S3KeySensor(
    task_id="wait_for_data",
    bucket_name="ml-data-lake",
    bucket_key="raw/daily/{{ ds }}/_SUCCESS",
    aws_conn_id="aws_default",
    timeout=3600,
    poke_interval=300,
    dag=dag,
)

# Task 2: Data validation
def validate_data(**context):
    import great_expectations as gx

    data_context = gx.get_context()
    result = data_context.run_checkpoint(
        checkpoint_name="raw_data_checkpoint",
        batch_request={
            "datasource_name": "s3_datasource",
            "data_asset_name": "raw_daily",
            "options": {"path": f"raw/daily/{context['ds']}/"},
        },
    )
    if not result.success:
        raise ValueError("Data validation failed")
    return result.to_json_dict()

validate = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    provide_context=True,
    dag=dag,
)

# Task 3: Feature engineering (runs in Kubernetes for isolation)
feature_engineering = KubernetesPodOperator(
    task_id="feature_engineering",
    name="feature-engineering",
    namespace="ml-pipelines",
    image="ml-registry.company.com/feature-engine:v2.3",
    cmds=["python", "feature_pipeline.py"],
    arguments=["--date={{ ds }}", "--output=s3://ml-features/{{ ds }}/"],
    resources={
        "request_memory": "8Gi",
        "request_cpu": "4",
        "limit_memory": "16Gi",
        "limit_cpu": "8",
    },
    env_vars={"SPARK_DRIVER_MEMORY": "4g"},
    is_delete_operator_pod=True,
    dag=dag,
)

# Task 4: Model training (GPU-accelerated)
model_training = KubernetesPodOperator(
    task_id="model_training",
    name="model-training",
    namespace="ml-pipelines",
    image="ml-registry.company.com/trainer:v1.5",
    cmds=["python", "train.py"],
    arguments=[
        "--features=s3://ml-features/{{ ds }}/",
        "--model-output=s3://ml-models/{{ ds }}/",
        "--experiment=weekly-retrain",
    ],
    resources={
        "request_memory": "32Gi",
        "request_cpu": "8",
        "limit_memory": "64Gi",
        "limit_cpu": "16",
        "request_gpu": "1",
    },
    node_selector={"gpu-type": "nvidia-a100"},
    is_delete_operator_pod=True,
    dag=dag,
)

# Task 5: Model evaluation
def evaluate_model(**context):
    import mlflow

    client = mlflow.tracking.MlflowClient()
    latest_run = client.search_runs(
        experiment_ids=["1"],
        filter_string=f"tags.date = '{context['ds']}'",
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )[0]

    # Compare with production model
    production_model = client.get_latest_versions(
        "ProductionModel", stages=["Production"]
    )
    if production_model:
        prod_f1 = production_model[0].run_id
        prod_run = client.get_run(prod_f1)
        prod_score = float(prod_run.data.metrics.get("f1_score", 0))
        new_score = float(latest_run.data.metrics["f1_score"])

        if new_score > prod_score:
            context["ti"].xcom_push(
                key="promote", value=True
            )
            context["ti"].xcom_push(
                key="model_version", value=latest_run.info.run_id
            )
        else:
            context["ti"].xcom_push(key="promote", value=False)

evaluate = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
)

# Task 6: Conditional model promotion
def promote_model(**context):
    promote = context["ti"].xcom_pull(
        task_ids="evaluate_model", key="promote"
    )
    if promote:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        run_id = context["ti"].xcom_pull(
            task_ids="evaluate_model", key="model_version"
        )
        client.transition_model_version_stage(
            name="ProductionModel",
            version=run_id,
            stage="Production",
        )

promote = PythonOperator(
    task_id="promote_model",
    python_callable=promote_model,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
wait_for_data >> validate >> feature_engineering >> model_training >> evaluate >> promote
```

### 19.5.3 SLAs and Backfilling

Airflow's **SLA** (Service Level Agreement) feature allows you to set expected completion times for tasks. If a task exceeds its SLA, Airflow sends an alert---critical for ML pipelines that must complete before serving windows begin.

**Backfilling** allows you to rerun a DAG for historical dates. If you deploy a new feature engineering pipeline, you can backfill features for the past year:

```bash
airflow dags backfill ml_training_pipeline \
    --start-date 2025-01-01 \
    --end-date 2025-12-31
```

---

## 19.6 Prefect

Prefect is a modern workflow orchestration tool that positions itself as a simpler, more Pythonic alternative to Airflow. Where Airflow requires DAG definitions, scheduler configuration, and a metadata database, Prefect allows you to turn any Python function into a workflow with decorators.

### 19.6.1 Flow and Task Decorators

```python
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import mlflow

@task(
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def load_data(path: str) -> pd.DataFrame:
    """Load and validate training data."""
    df = pd.read_parquet(path)
    assert len(df) > 1000, f"Insufficient data: {len(df)} rows"
    assert df.isnull().sum().sum() < len(df) * 0.05, "Too many nulls"
    return df

@task(log_prints=True)
def train_model(
    df: pd.DataFrame, n_estimators: int = 100, max_depth: int = 10
) -> RandomForestClassifier:
    """Train a Random Forest model."""
    X = df.drop("target", axis=1)
    y = df["target"]
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X, y)
    print(f"Trained model with {n_estimators} estimators")
    return model

@task
def evaluate_model(model, df: pd.DataFrame) -> float:
    """Evaluate the model on test data."""
    X = df.drop("target", axis=1)
    y = df["target"]
    predictions = model.predict(X)
    score = f1_score(y, predictions, average="weighted")
    return score

@flow(name="ml-training-pipeline")
def training_pipeline(
    train_path: str,
    test_path: str,
    n_estimators: int = 100,
    max_depth: int = 10,
):
    """Complete ML training pipeline."""
    # Tasks are called like normal functions
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    model = train_model(train_df, n_estimators, max_depth)
    score = evaluate_model(model, test_df)

    if score > 0.85:
        mlflow.sklearn.log_model(model, "model")
        print(f"Model promoted with score: {score}")
    else:
        print(f"Model rejected with score: {score}")

    return score

# Run locally
if __name__ == "__main__":
    training_pipeline(
        train_path="s3://data/train.parquet",
        test_path="s3://data/test.parquet",
        n_estimators=200,
    )
```

### 19.6.2 Orion Server, Deployments, and Work Queues

Prefect's Orion server (now Prefect Server) provides a UI for monitoring flows, inspecting runs, and managing deployments. **Deployments** define how and where a flow runs:

```python
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect.infrastructure.docker import DockerContainer

deployment = Deployment.build_from_flow(
    flow=training_pipeline,
    name="weekly-training",
    schedule=CronSchedule(cron="0 6 * * 1"),
    infrastructure=DockerContainer(
        image="ml-pipeline:latest",
        image_pull_policy="ALWAYS",
    ),
    work_queue_name="ml-gpu",
    parameters={
        "train_path": "s3://data/train.parquet",
        "test_path": "s3://data/test.parquet",
    },
)
deployment.apply()
```

**Work queues** route deployments to the appropriate infrastructure. A GPU work queue is served by agents running on GPU-equipped machines; a CPU work queue by standard compute instances. **Blocks** provide a unified interface for storing configuration and credentials (database connections, cloud storage paths, API keys).

### 19.6.3 Why Teams Choose Prefect Over Airflow

Several factors drive adoption of Prefect over Airflow:

1. **Simplicity.** Prefect flows are plain Python functions. There is no need to understand DAG serialization, operator templates, or the Airflow scheduler's execution model.
2. **Dynamic Workflows.** Prefect natively supports dynamic workflows where the DAG structure depends on runtime data. In Airflow, this requires workarounds like dynamic task generation or branching operators.
3. **Local Development.** A Prefect flow can be run locally as a normal Python script, with no server or database required. Airflow requires a running scheduler, webserver, and metadata database even for local development.
4. **Modern Architecture.** Prefect's server is built on FastAPI and uses an async-first design. Airflow's architecture, while battle-tested, shows its age in areas like the scheduler's polling model and the reliance on pickle for task serialization.

---

## 19.7 Dagster

Dagster takes a fundamentally different approach to orchestration. Where Airflow and Prefect are task-oriented---they define workflows as sequences of operations---Dagster is asset-oriented. The primary abstraction is the **software-defined asset**: a materialized data artifact with a clear definition, dependencies, and quality expectations.

### 19.7.1 Core Concepts

- **Software-Defined Assets** are functions that produce data artifacts. Dagster tracks the lineage between assets, re-materializes them when their dependencies change, and provides a global view of all assets in the data platform.
- **Ops** are the lower-level building blocks: individual units of computation, analogous to Airflow tasks.
- **Jobs** are collections of ops that execute together.
- **Resources** are shared services (database connections, API clients) that ops and assets can depend on.
- **IO Managers** handle the serialization and deserialization of data between ops. A single pipeline can use different IO managers for different assets (e.g., Parquet for tabular data, pickle for models).
- **Dagit** is Dagster's web UI, providing a graph visualization of assets and their dependencies, real-time execution monitoring, and a type-aware launchpad for parameterized runs.

### 19.7.2 Asset-Oriented ML Pipeline

```python
from dagster import (
    asset,
    AssetIn,
    Output,
    MetadataValue,
    MaterializeResult,
    Definitions,
    define_asset_job,
    ScheduleDefinition,
)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import mlflow

@asset(
    description="Raw customer data from the data warehouse",
    group_name="data_ingestion",
)
def raw_customer_data() -> pd.DataFrame:
    """Load raw customer data."""
    df = pd.read_parquet("s3://data-lake/raw/customers/")
    return df

@asset(
    description="Engineered features for churn prediction",
    group_name="feature_engineering",
)
def churn_features(raw_customer_data: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for churn prediction."""
    df = raw_customer_data.copy()

    # Feature engineering
    df["tenure_months"] = (
        pd.Timestamp.now() - pd.to_datetime(df["signup_date"])
    ).dt.days / 30
    df["avg_monthly_spend"] = df["total_spend"] / df["tenure_months"].clip(lower=1)
    df["support_ticket_rate"] = df["support_tickets"] / df["tenure_months"].clip(lower=1)
    df["engagement_score"] = (
        df["login_frequency"] * 0.3
        + df["feature_usage_breadth"] * 0.4
        + df["session_duration_avg"] * 0.3
    )

    feature_cols = [
        "tenure_months", "avg_monthly_spend", "support_ticket_rate",
        "engagement_score", "contract_type", "payment_method", "churned",
    ]
    return df[feature_cols]

@asset(
    description="Trained churn prediction model",
    group_name="modeling",
)
def churn_model(churn_features: pd.DataFrame) -> Output:
    """Train and evaluate a churn prediction model."""
    X = churn_features.drop("churned", axis=1)
    y = churn_features["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metrics({
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
        })
        mlflow.sklearn.log_model(model, "model")

    return Output(
        model,
        metadata={
            "accuracy": MetadataValue.float(report["accuracy"]),
            "f1_score": MetadataValue.float(report["1"]["f1-score"]),
            "training_samples": MetadataValue.int(len(X_train)),
        },
    )

# Define a job and schedule
training_job = define_asset_job(
    name="churn_training_job",
    selection=["raw_customer_data", "churn_features", "churn_model"],
)

weekly_schedule = ScheduleDefinition(
    job=training_job,
    cron_schedule="0 6 * * 1",  # Every Monday at 6 AM
)

defs = Definitions(
    assets=[raw_customer_data, churn_features, churn_model],
    jobs=[training_job],
    schedules=[weekly_schedule],
)
```

Dagster's asset-oriented approach is particularly well-suited to ML because it mirrors how data scientists think: not as a sequence of tasks to execute, but as a collection of artifacts to produce. The Dagit UI visualizes the entire asset graph, making it easy to understand how data flows from raw sources to deployed models.

---

## 19.8 Kubeflow Pipelines

Kubeflow Pipelines is a Kubernetes-native platform for building and deploying ML workflows. Each step in a Kubeflow pipeline runs as a separate container in a Kubernetes pod, providing strong isolation, reproducibility, and scalability.

### 19.8.1 Components and Pipeline DSL

Kubeflow pipelines are defined using the KFP SDK. **Components** are self-contained units of work, each with defined inputs and outputs:

```python
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"],
)
def preprocess_data(
    raw_data_path: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(raw_data_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_parquet(train_data.path)
    test_df.to_parquet(test_data.path)

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "pyarrow", "joblib"],
)
def train_model(
    train_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100,
    max_depth: int = 5,
):
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    import joblib

    df = pd.read_parquet(train_data.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    model = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth
    )
    model.fit(X, y)

    train_acc = accuracy_score(y, model.predict(X))
    metrics.log_metric("train_accuracy", train_acc)

    joblib.dump(model, model_artifact.path)

@dsl.pipeline(name="ml-training-pipeline")
def ml_pipeline(
    raw_data_path: str = "gs://ml-data/raw/data.parquet",
    n_estimators: int = 200,
    max_depth: int = 5,
):
    preprocess_task = preprocess_data(raw_data_path=raw_data_path)

    train_task = train_model(
        train_data=preprocess_task.outputs["train_data"],
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    # Set GPU resources
    train_task.set_gpu_limit(1)
    train_task.set_memory_limit("16Gi")
    train_task.set_cpu_limit("4")
```

### 19.8.2 Katib for Hyperparameter Optimization

**Katib** is Kubeflow's hyperparameter optimization component. It supports multiple search algorithms (random search, grid search, Bayesian optimization, Hyperband, neural architecture search) and runs trials as separate Kubernetes jobs:

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: churn-model-hpo
spec:
  objective:
    type: maximize
    goal: 0.95
    objectiveMetricName: f1-score
  algorithm:
    algorithmName: bayesianoptimization
  parallelTrialCount: 3
  maxTrialCount: 30
  maxFailedTrialCount: 3
  parameters:
    - name: learning_rate
      parameterType: double
      feasibleSpace:
        min: "0.001"
        max: "0.1"
    - name: n_estimators
      parameterType: int
      feasibleSpace:
        min: "50"
        max: "500"
    - name: max_depth
      parameterType: int
      feasibleSpace:
        min: "3"
        max: "10"
  trialTemplate:
    primaryContainerName: training
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            containers:
              - name: training
                image: ml-training:latest
                command:
                  - "python"
                  - "train.py"
                  - "--lr=${trialParameters.learning_rate}"
                  - "--n-estimators=${trialParameters.n_estimators}"
                  - "--max-depth=${trialParameters.max_depth}"
```

Kubeflow's primary advantage is its Kubernetes-native design. For organizations that have already invested in Kubernetes, Kubeflow Pipelines provides a natural way to run ML workflows alongside other containerized workloads, with all the benefits of Kubernetes' scheduling, scaling, and resource management.

---

## 19.9 Ray

Ray (Moritz et al., 2018) is a unified framework for scaling AI applications. It began as a distributed computing framework at Berkeley's RISELab and has grown into an ecosystem that covers training, tuning, serving, and data processing.

### 19.9.1 Ray Core

Ray Core provides two primitives for distributed computing: **remote functions** and **actors**.

```python
import ray
import numpy as np

ray.init()

# Remote functions: stateless, parallelizable
@ray.remote
def process_partition(data: np.ndarray) -> dict:
    """Process a data partition and return statistics."""
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "count": len(data),
    }

# Launch 10 tasks in parallel
data = np.random.randn(1_000_000)
partitions = np.array_split(data, 10)
futures = [process_partition.remote(p) for p in partitions]
results = ray.get(futures)

# Actors: stateful, long-running
@ray.remote
class ModelServer:
    def __init__(self, model_path: str):
        import joblib
        self.model = joblib.load(model_path)
        self.request_count = 0

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.request_count += 1
        return self.model.predict(features)

    def get_stats(self) -> dict:
        return {"requests_served": self.request_count}

# Create two model server replicas
servers = [ModelServer.remote("model.pkl") for _ in range(2)]

# Route requests round-robin
for i, batch in enumerate(feature_batches):
    server = servers[i % len(servers)]
    prediction = ray.get(server.predict.remote(batch))
```

### 19.9.2 Ray Train

Ray Train provides distributed training for PyTorch, TensorFlow, and other frameworks:

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
import ray.train as train

def train_loop_per_worker(config):
    import torch
    from torch.utils.data import DataLoader

    # Ray handles distributed setup (rank, world_size, etc.)
    model = build_model(config["hidden_size"])
    model = train.torch.prepare_model(model)

    dataset = load_dataset()
    dataloader = DataLoader(dataset, batch_size=config["batch_size"])
    dataloader = train.torch.prepare_data_loader(dataloader)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train.report({"loss": loss.item(), "epoch": epoch})

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        "hidden_size": 256,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 10,
    },
    scaling_config=ScalingConfig(
        num_workers=4,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 4},
    ),
    run_config=RunConfig(name="distributed-training"),
)

result = trainer.fit()
```

### 19.9.3 Ray Tune

Ray Tune integrates with Ray Train for distributed hyperparameter search:

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

search_space = {
    "hidden_size": tune.choice([128, 256, 512]),
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([32, 64, 128]),
    "epochs": 10,
}

scheduler = ASHAScheduler(
    max_t=10,
    grace_period=2,
    reduction_factor=3,
)

tuner = tune.Tuner(
    trainer,
    param_space={"train_loop_config": search_space},
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
        num_samples=50,
        scheduler=scheduler,
        search_alg=OptunaSearch(),
    ),
)

results = tuner.fit()
best_result = results.get_best_result()
print(f"Best config: {best_result.config}")
```

### 19.9.4 Ray Serve

Ray Serve provides scalable model serving with support for model composition:

```python
from ray import serve
import ray
from starlette.requests import Request

@serve.deployment(
    num_replicas=3,
    ray_actor_options={"num_gpus": 1},
)
class SentimentAnalyzer:
    def __init__(self):
        from transformers import pipeline
        self.model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0,
        )

    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        result = self.model(data["text"])
        return {"sentiment": result[0]["label"], "score": result[0]["score"]}

# Deploy
serve.run(SentimentAnalyzer.bind())
```

### 19.9.5 When to Use Ray

Ray is the right choice when you need a unified framework that spans training, tuning, serving, and data processing, especially when workloads are heterogeneous (mixing CPU and GPU tasks) or when you need to scale beyond a single machine. It is not the right choice for simple, single-machine workflows where tools like scikit-learn or a basic Flask server suffice. Ray's strength is its generality: the same cluster can run distributed training in the morning, hyperparameter tuning in the afternoon, and model serving around the clock.

---

## 19.10 Metaflow

Metaflow, developed at Netflix and open-sourced in 2019, provides a flow-based abstraction for data science workflows. Its design philosophy is that data scientists should be able to write production-grade pipelines using familiar Python patterns, without becoming infrastructure experts.

### 19.10.1 Flow-Based Abstraction

```python
from metaflow import FlowSpec, step, Parameter, conda_base, resources, batch

@conda_base(python="3.10", libraries={"scikit-learn": "1.3.0", "pandas": "2.0.0"})
class ChurnPredictionFlow(FlowSpec):
    """End-to-end churn prediction pipeline."""

    data_path = Parameter(
        "data_path",
        help="Path to training data",
        default="s3://data/customers.parquet",
    )

    n_estimators = Parameter(
        "n_estimators",
        help="Number of trees",
        default=200,
        type=int,
    )

    @step
    def start(self):
        """Load and validate data."""
        import pandas as pd

        self.df = pd.read_parquet(self.data_path)
        print(f"Loaded {len(self.df)} rows")

        # Fan out to parallel feature engineering
        self.feature_groups = ["demographic", "behavioral", "financial"]
        self.next(self.engineer_features, foreach="feature_groups")

    @resources(cpu=4, memory=8192)
    @step
    def engineer_features(self):
        """Engineer features for each group (runs in parallel)."""
        import pandas as pd

        group = self.input
        if group == "demographic":
            self.features = self._demographic_features(self.df)
        elif group == "behavioral":
            self.features = self._behavioral_features(self.df)
        elif group == "financial":
            self.features = self._financial_features(self.df)

        self.next(self.join_features)

    @step
    def join_features(self, inputs):
        """Join features from parallel branches."""
        import pandas as pd

        self.df = inputs[0].df
        feature_dfs = [inp.features for inp in inputs]
        self.feature_df = pd.concat(feature_dfs, axis=1)
        self.next(self.train)

    @batch(gpu=1, memory=32768, cpu=8)
    @step
    def train(self):
        """Train the model on AWS Batch with GPU."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score

        X = self.feature_df
        y = self.df["churned"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators, random_state=42
        )
        self.model.fit(X_train, y_train)

        self.f1 = f1_score(y_test, self.model.predict(X_test))
        print(f"F1 Score: {self.f1:.4f}")
        self.next(self.end)

    @step
    def end(self):
        """Finalize the pipeline."""
        print(f"Pipeline complete. F1: {self.f1:.4f}")

if __name__ == "__main__":
    ChurnPredictionFlow()
```

### 19.10.2 Key Decorators

Metaflow's power lies in its decorators:

- `@conda` and `@pip` specify per-step dependencies, ensuring that each step runs in the correct environment.
- `@resources` specifies CPU and memory requirements.
- `@batch` routes a step to AWS Batch, enabling GPU access without managing infrastructure.
- `@retry` adds automatic retry logic.
- `@timeout` sets a maximum execution time.
- `@catch` catches exceptions and allows the pipeline to continue.

The `foreach` parameter on `self.next()` enables parallel fan-out---a pattern that is natural for ML workflows where multiple feature groups, model architectures, or hyperparameter configurations can be explored in parallel.

Metaflow's integration with AWS is its primary differentiator. Flows can seamlessly move between local execution and cloud execution, with data and artifacts automatically managed in S3. The `Client` API allows querying past runs and retrieving artifacts, enabling the kind of retrospective analysis that is essential for debugging production ML systems.

---

## 19.11 Model Monitoring

Deploying a model is not the end of the ML lifecycle---it is the beginning of a new phase in which the model's performance must be continuously verified against the reality of production data. The core challenge is that production data is non-stationary: the distributions that the model was trained on will change, sometimes gradually and sometimes abruptly.

### 19.11.1 Types of Drift

**Data Drift** (also called covariate shift) occurs when the distribution of input features changes. A model trained on summer weather data will encounter data drift when winter arrives. Data drift does not necessarily cause performance degradation---if the model generalizes well, it may handle the new distribution fine---but it is a warning sign that warrants investigation.

**Concept Drift** occurs when the relationship between inputs and outputs changes. The same input features now predict a different outcome. For example, during the COVID-19 pandemic, the relationship between economic indicators and consumer spending changed dramatically. Concept drift almost always causes performance degradation.

**Feature Drift** occurs when individual features change in unexpected ways, often due to upstream pipeline changes (a column that was previously a float becomes a string, a feature that was denominated in dollars is now in euros).

**Prediction Drift** occurs when the distribution of model predictions changes, independent of whether the inputs or the true outcomes have changed. This can indicate data drift, concept drift, or a change in the model's calibration.

### 19.11.2 Statistical Tests for Drift Detection

Several statistical tests are commonly used to detect drift:

**Population Stability Index (PSI).** PSI measures the difference between two distributions by comparing their binned frequencies:

$$\text{PSI} = \sum_{i=1}^{n} (p_i - q_i) \cdot \ln\left(\frac{p_i}{q_i}\right)$$

where $p_i$ and $q_i$ are the proportions in bin $i$ for the reference and current distributions, respectively. A PSI below 0.1 indicates no significant change; 0.1--0.25 indicates moderate change; above 0.25 indicates significant change (Yurdakul, 2018).

**Kolmogorov-Smirnov (KS) Test.** The KS test measures the maximum difference between two cumulative distribution functions. It is distribution-free and works well for continuous features:

$$D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$

**Jensen-Shannon Divergence (JSD).** JSD is a symmetric version of KL divergence, bounded between 0 and 1:

$$\text{JSD}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$. JSD is particularly useful for comparing categorical distributions.

### 19.11.3 Monitoring with Evidently AI

Evidently AI is an open-source library for ML monitoring that generates interactive dashboards and reports:

```python
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
    ClassificationPreset,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnDrift,
)

# Reference data (training set) and current data (production)
reference_data = pd.read_parquet("reference_data.parquet")
current_data = pd.read_parquet("current_production_data.parquet")

column_mapping = ColumnMapping(
    target="churned",
    prediction="prediction",
    numerical_features=["tenure", "monthly_spend", "support_tickets"],
    categorical_features=["contract_type", "payment_method"],
)

# Generate a comprehensive drift report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    TargetDriftPreset(),
    ClassificationPreset(),
])

report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping,
)

# Save as HTML dashboard
report.save_html("monitoring_report.html")

# Programmatic drift tests for CI/CD integration
test_suite = TestSuite(tests=[
    TestNumberOfDriftedColumns(lt=3),
    TestShareOfDriftedColumns(lt=0.3),
    TestColumnDrift(column_name="monthly_spend"),
    TestColumnDrift(column_name="tenure"),
])

test_suite.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping,
)

# Fail the pipeline if drift is detected
result = test_suite.as_dict()
if not result["summary"]["all_passed"]:
    failed_tests = [
        t["name"] for t in result["tests"] if t["status"] == "FAIL"
    ]
    raise ValueError(f"Drift detected! Failed tests: {failed_tests}")
```

### 19.11.4 Alert Configuration

Effective monitoring requires well-configured alerts. The key principle is to minimize false positives while catching genuine degradation:

1. **Set thresholds based on historical variation.** If a metric normally fluctuates within a 2% band, an alert that fires at a 1% deviation will produce constant noise. Use statistical process control charts or rolling percentiles to establish baselines.
2. **Use multi-level alerts.** A warning alert (e.g., PSI > 0.1) triggers investigation; a critical alert (e.g., PSI > 0.25 or model accuracy drop > 5%) triggers automatic retraining or rollback.
3. **Combine multiple signals.** A single drifted feature may be benign; simultaneous drift in multiple features, combined with prediction distribution changes, is a stronger signal.
4. **Monitor upstream data.** Many "model" failures are actually data pipeline failures. Monitor data freshness, schema compliance, and volume alongside model metrics.

---

## 19.12 CI/CD for ML

Continuous Integration and Continuous Deployment (CI/CD) for ML extends traditional software CI/CD with ML-specific concerns: data validation, model quality gates, and performance benchmarking.

### 19.12.1 Testing Pyramid for ML

**Unit Tests.** Test individual functions: feature engineering logic, data transformations, custom loss functions. These tests should be fast (seconds) and deterministic.

```python
# test_features.py
import pytest
import pandas as pd
import numpy as np
from features import compute_recency_features

def test_recency_features_basic():
    """Test that recency features are computed correctly."""
    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "last_purchase_date": pd.to_datetime([
            "2025-01-01", "2025-06-01", "2025-12-01"
        ]),
    })
    result = compute_recency_features(df, reference_date=pd.Timestamp("2025-12-31"))
    assert result["days_since_purchase"].tolist() == [364, 213, 30]

def test_recency_features_handles_nulls():
    """Test that null dates produce null recency."""
    df = pd.DataFrame({
        "user_id": [1, 2],
        "last_purchase_date": [pd.Timestamp("2025-01-01"), pd.NaT],
    })
    result = compute_recency_features(df, reference_date=pd.Timestamp("2025-12-31"))
    assert pd.isna(result["days_since_purchase"].iloc[1])
```

**Integration Tests.** Test that pipeline components work together: data loading, feature engineering, training, and serving produce correct results when connected.

**Model Validation Tests.** Test that a trained model meets minimum quality standards:

```python
# test_model_quality.py
import pytest
from sklearn.metrics import f1_score, roc_auc_score

def test_model_meets_minimum_f1(trained_model, test_data):
    """Model must achieve at least 0.80 F1 on the test set."""
    X, y = test_data
    predictions = trained_model.predict(X)
    f1 = f1_score(y, predictions, average="weighted")
    assert f1 >= 0.80, f"F1 score {f1:.4f} below minimum threshold 0.80"

def test_model_not_overfit(trained_model, train_data, test_data):
    """Train/test F1 gap must be less than 0.10."""
    X_train, y_train = train_data
    X_test, y_test = test_data

    train_f1 = f1_score(y_train, trained_model.predict(X_train), average="weighted")
    test_f1 = f1_score(y_test, trained_model.predict(X_test), average="weighted")

    gap = train_f1 - test_f1
    assert gap < 0.10, f"Overfit detected: train F1={train_f1:.4f}, test F1={test_f1:.4f}"

def test_model_prediction_latency(trained_model, single_sample):
    """Single prediction must complete in under 50ms."""
    import time

    start = time.perf_counter()
    trained_model.predict(single_sample)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 50, f"Prediction latency {elapsed_ms:.1f}ms exceeds 50ms threshold"
```

### 19.12.2 GitHub Actions Workflow for ML

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline CI/CD

on:
  push:
    branches: [main]
    paths:
      - "src/**"
      - "configs/**"
      - "tests/**"
  pull_request:
    branches: [main]

env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  AWS_REGION: us-east-1

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt

      - name: Run unit tests
        run: pytest tests/unit/ -v --tb=short

      - name: Run integration tests
        run: pytest tests/integration/ -v --tb=short

      - name: Lint and type check
        run: |
          ruff check src/
          mypy src/ --ignore-missing-imports

  train-and-evaluate:
    needs: test
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Train model
        run: |
          python src/train.py \
            --config configs/production.yaml \
            --experiment-name "ci-cd-training"

      - name: Evaluate model
        run: |
          python src/evaluate.py \
            --model-uri "runs:/${{ env.RUN_ID }}/model" \
            --test-data "s3://ml-data/test/" \
            --output metrics.json

      - name: Model quality gate
        run: pytest tests/model_quality/ -v --tb=short

      - name: Register model
        if: success()
        run: |
          python src/register_model.py \
            --run-id "${{ env.RUN_ID }}" \
            --model-name "production-model" \
            --stage "Staging"

  deploy:
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Deploy canary (10% traffic)
        run: |
          python src/deploy.py \
            --model-name "production-model" \
            --stage "Staging" \
            --canary-weight 10

      - name: Monitor canary (30 minutes)
        run: |
          python src/monitor_canary.py \
            --duration 1800 \
            --error-rate-threshold 0.01 \
            --latency-p99-threshold 200

      - name: Promote to full traffic
        if: success()
        run: |
          python src/deploy.py \
            --model-name "production-model" \
            --stage "Production" \
            --canary-weight 100
```

### 19.12.3 Canary Deployments and A/B Testing

**Canary deployments** route a small percentage of traffic to the new model while the majority continues to be served by the current production model. If the canary model's error rate, latency, or business metrics deviate from acceptable ranges, the deployment is automatically rolled back.

**A/B testing** is a more rigorous approach: users are randomly assigned to the control (current model) or treatment (new model) group, and the difference in business outcomes is measured with statistical significance. A/B tests require more traffic and time than canary deployments but provide stronger evidence that the new model is genuinely better.

---

## 19.13 Containerization for ML

Docker containers provide the reproducibility and isolation that ML workloads require. A container packages the model, its dependencies, and the serving code into a single artifact that runs identically on a developer's laptop and in a production Kubernetes cluster.

### 19.13.1 Multi-Stage Builds

Multi-stage builds reduce image size by separating the build environment from the runtime environment:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy only the installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 19.13.2 GPU Support with NVIDIA Container Toolkit

For GPU-accelerated inference, use NVIDIA's base images:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

CMD ["python3", "-m", "src.serve"]
```

The NVIDIA Container Toolkit (`nvidia-container-toolkit`) must be installed on the host machine. It provides the `nvidia` runtime that exposes GPU devices to containers:

```bash
# Run with GPU access
docker run --gpus all -p 8080:8080 ml-serving:latest

# Run with specific GPUs
docker run --gpus '"device=0,1"' -p 8080:8080 ml-serving:latest
```

### 19.13.3 Image Size Optimization

ML Docker images can easily reach 10GB or more. Strategies for reducing size include:

1. **Use slim base images.** `python:3.10-slim` is 120MB vs. 900MB for `python:3.10`.
2. **Multi-stage builds.** Build tools (compilers, headers) are needed only during `pip install` and can be discarded afterward.
3. **Avoid caching pip downloads.** Always use `--no-cache-dir`.
4. **Minimize layers.** Combine related `RUN` commands with `&&`.
5. **Use `.dockerignore`.** Exclude test files, documentation, `.git`, and other unnecessary files from the build context.

---

## 19.14 Kubernetes for ML

Kubernetes is the de facto standard for container orchestration. For ML workloads, it provides resource management, scaling, and scheduling capabilities that are essential for running training jobs, inference services, and data pipelines at scale.

### 19.14.1 Core Abstractions for ML

**Pods** are the smallest deployable unit. An ML inference pod typically contains the model server container and optional sidecar containers (e.g., for logging or metrics collection).

**Deployments** manage the desired state of a set of pods. For ML serving, a Deployment ensures that the specified number of model server replicas are running at all times.

**Services** provide stable network endpoints for accessing pods. A ClusterIP Service exposes the model server within the cluster; a LoadBalancer Service exposes it externally.

### 19.14.2 Autoscaling for ML Workloads

**Horizontal Pod Autoscaler (HPA)** scales the number of pods based on CPU utilization, memory usage, or custom metrics:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: requests_per_second
        target:
          type: AverageValue
          averageValue: 100
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

**KEDA (Kubernetes Event-Driven Autoscaling)** extends HPA with the ability to scale based on external event sources (Kafka queue depth, SQS message count, Prometheus metrics, etc.). For ML batch processing, KEDA can scale workers based on the number of pending inference requests in a message queue:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: ml-batch-processor
spec:
  scaleTargetRef:
    name: batch-inference
  pollingInterval: 15
  cooldownPeriod: 300
  minReplicaCount: 0  # Scale to zero when idle
  maxReplicaCount: 50
  triggers:
    - type: aws-sqs-queue
      metadata:
        queueURL: https://sqs.us-east-1.amazonaws.com/123456789/ml-inference-queue
        queueLength: "10"
        awsRegion: us-east-1
```

### 19.14.3 GPU Scheduling

Kubernetes supports GPU scheduling through device plugins. The NVIDIA device plugin advertises GPU resources to the scheduler:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-model-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gpu-model-serving
  template:
    metadata:
      labels:
        app: gpu-model-serving
    spec:
      containers:
        - name: model-server
          image: ml-registry.company.com/model-server:v2.0
          resources:
            requests:
              memory: "16Gi"
              cpu: "4"
              nvidia.com/gpu: 1
            limits:
              memory: "32Gi"
              cpu: "8"
              nvidia.com/gpu: 1
          ports:
            - containerPort: 8080
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      nodeSelector:
        gpu-type: "nvidia-a100"
```

A key constraint of Kubernetes GPU scheduling is that GPUs cannot be shared between pods (without specialized tools like NVIDIA MPS or MIG). Each pod that requests a GPU gets exclusive access to one or more whole GPUs. This means that GPU utilization optimization is critical: an inference model that uses only 20% of a GPU's memory and compute is wasting 80% of an expensive resource.

**Multi-Instance GPU (MIG)**, available on NVIDIA A100 and H100 GPUs, addresses this by partitioning a single physical GPU into multiple isolated instances. Each MIG instance has its own memory, cache, and compute resources, and appears as a separate GPU to Kubernetes. This allows multiple models to share a single physical GPU without interference.

---

## Exercises

1. **Experiment Tracking Comparison.** Set up MLflow and W&B to track the same experiment (training a classifier on a dataset of your choice). Compare the two platforms on ease of use, visualization quality, and collaboration features. Write a brief report summarizing your findings.

2. **DVC Pipeline.** Create a DVC pipeline for a text classification task with stages for data preparation, tokenization, training, and evaluation. Use `dvc exp run` with different hyperparameters and compare results with `dvc exp show`.

3. **Airflow vs. Prefect.** Implement the same ML pipeline in both Airflow and Prefect. The pipeline should: (a) load data from a file, (b) validate it, (c) train a model, (d) evaluate it, and (e) register it if it meets a quality threshold. Compare the developer experience.

4. **Drift Detection System.** Using Evidently AI, build a monitoring system that: (a) generates a reference profile from training data, (b) computes drift statistics for new batches of data, (c) triggers an alert if PSI exceeds 0.25 for any feature.

5. **Kubernetes Deployment.** Containerize a model serving application with Docker (multi-stage build). Deploy it to a Kubernetes cluster with an HPA that scales based on CPU utilization. Test it with a load generator and observe the scaling behavior.

6. **CI/CD Pipeline.** Create a GitHub Actions workflow that: (a) runs unit tests on every PR, (b) trains a model on merge to main, (c) evaluates the model against a quality gate, and (d) deploys it to a staging environment if it passes.

7. **Dagster Asset Pipeline.** Rewrite the Airflow DAG from Exercise 3 as a Dagster asset pipeline. Discuss the advantages and disadvantages of the asset-oriented approach.

8. **Ray Distributed Training.** Using Ray Train, implement distributed data-parallel training for a PyTorch model across 4 workers. Compare training throughput (samples/second) with 1, 2, and 4 workers.

---

## References

Apache Airflow Documentation. (2024). *Apache Airflow*. https://airflow.apache.org/docs/

Biewald, L. (2020). *Experiment Tracking with Weights and Biases*. Weights & Biases. https://www.wandb.com/

Dagster Documentation. (2024). *Dagster: The Data Orchestration Platform*. https://docs.dagster.io/

DVC Documentation. (2024). *Data Version Control*. https://dvc.org/doc

Google. (2021). MLOps: Continuous delivery and automation pipelines in machine learning. *Google Cloud Architecture Center*. https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

Kubeflow Documentation. (2024). *Kubeflow Pipelines*. https://www.kubeflow.org/docs/components/pipelines/

Metaflow Documentation. (2024). *Metaflow: A Framework for Real-Life Data Science*. https://docs.metaflow.org/

Moritz, P., Nishihara, R., Wang, S., Tumanov, A., Liaw, R., Liang, E., Elibol, M., Yang, Z., Paul, W., Jordan, M. I., & Stoica, I. (2018). Ray: A Distributed Framework for Emerging AI Applications. In *Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)* (pp. 561--577).

Prefect Documentation. (2024). *Prefect: The Easiest Way to Orchestrate and Observe Your Data Pipelines*. https://docs.prefect.io/

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden Technical Debt in Machine Learning Systems. In *Advances in Neural Information Processing Systems 28 (NeurIPS 2015)* (pp. 2503--2511).

Yurdakul, B. (2018). Statistical Properties of Population Stability Index. *Western Michigan University ScholarWorks*.

Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F., & Zuber, C. (2018). Accelerating the Machine Learning Lifecycle with MLflow. *IEEE Data Engineering Bulletin*, 41(4), 39--45.
