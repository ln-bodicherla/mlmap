# Chapter 21: Cloud Platforms for ML

## Learning Objectives

By the end of this chapter, you will be able to:

1. Evaluate the three major cloud platforms (AWS, GCP, Azure) for ML workloads using a structured decision framework.
2. Design and deploy end-to-end ML workflows on AWS SageMaker, including training, serving, and pipeline orchestration.
3. Build ML solutions on GCP Vertex AI and leverage TPUs for large-scale training.
4. Implement ML workloads on Azure ML with integration into the broader Microsoft ecosystem.
5. Provision and manage ML infrastructure using Terraform and CloudFormation.
6. Design multi-cloud architectures when business requirements demand them.
7. Optimize cloud costs for ML workloads using spot instances, reserved capacity, and auto-scaling strategies.

---

## 21.1 Choosing a Cloud Platform

The choice of cloud platform is one of the most consequential decisions an ML team makes. Migrating between platforms is expensive---often prohibitively so---because of data gravity, API dependencies, and organizational knowledge. The decision should therefore be made deliberately, based on a structured evaluation of several factors.

### 21.1.1 Decision Framework

**Existing Infrastructure.** The most common reason for choosing a cloud platform is that the organization already uses it. If the company's data is in S3, its applications run on EC2, and its developers know IAM, choosing AWS for ML eliminates integration friction. Choosing GCP in this situation means maintaining two cloud platforms, two billing systems, and two sets of operational expertise.

**Team Expertise.** The best platform is the one your team knows. An ML team staffed with ex-Googlers will be productive on GCP immediately; asking them to learn AWS adds months of ramp-up time. This consideration is often underweighted in platform evaluations.

**Specific Service Needs.** Some platforms have clear advantages for specific workloads:
- **TPUs** are only available on GCP. If your workload benefits from TPU architecture (large-scale language model training, JAX-based workflows), GCP is the natural choice.
- **Azure OpenAI Service** provides managed access to GPT-4 and related models with enterprise security. Organizations that need these specific models with enterprise compliance should consider Azure.
- **AWS Bedrock** provides managed access to Claude, Llama, and other foundation models with enterprise features like guardrails and knowledge bases.
- **AWS SageMaker** has the broadest feature set for traditional ML (feature store, model monitoring, pipelines, A/B testing).

**Cost.** Cloud pricing is notoriously complex, and the cheapest platform depends on the workload. In general: GCP tends to be cheaper for compute-intensive batch workloads (due to sustained-use discounts and preemptible VM pricing); AWS tends to have the most flexible pricing options; Azure tends to offer the best enterprise licensing deals (especially for organizations with Microsoft Enterprise Agreements).

**Compliance.** Regulated industries (healthcare, financial services, government) may have specific compliance requirements that constrain the platform choice. All three major providers have FedRAMP, HIPAA, SOC 2, and ISO certifications, but the specific services that are certified vary, and the compliance documentation is more mature for some providers than others.

---

## 21.2 AWS for ML

Amazon Web Services is the most widely adopted cloud platform, with the broadest set of ML services. AWS's ML strategy centers on SageMaker, a comprehensive platform that covers the entire ML lifecycle.

### 21.2.1 SageMaker

SageMaker provides a unified environment for building, training, and deploying ML models. Its major components include:

**Training Jobs.** SageMaker manages the infrastructure for training: provisioning instances, loading data, running the training code, saving model artifacts, and tearing down instances when training completes.

```python
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define the training job
estimator = PyTorch(
    entry_point="train.py",
    source_dir="src/",
    role=role,
    instance_count=4,
    instance_type="ml.p4d.24xlarge",  # 8x A100 GPUs per instance
    framework_version="2.1.0",
    py_version="py310",
    distribution={
        "torch_distributed": {
            "enabled": True,
        }
    },
    hyperparameters={
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "model_name": "bert-base-uncased",
    },
    output_path=f"s3://{session.default_bucket()}/models/",
    max_run=86400,  # 24 hours max
    checkpoint_s3_uri=f"s3://{session.default_bucket()}/checkpoints/",
    use_spot_instances=True,
    max_wait=172800,  # 48 hours max wait for spot
    tags=[
        {"Key": "project", "Value": "sentiment-analysis"},
        {"Key": "team", "Value": "ml-engineering"},
    ],
)

# Define input channels
train_input = TrainingInput(
    s3_data="s3://ml-data/train/",
    content_type="application/x-parquet",
    s3_data_type="S3Prefix",
    distribution="ShardedByS3Key",  # Distribute data across instances
)

validation_input = TrainingInput(
    s3_data="s3://ml-data/validation/",
    content_type="application/x-parquet",
)

# Launch training
estimator.fit(
    inputs={
        "train": train_input,
        "validation": validation_input,
    },
    job_name="sentiment-bert-v2",
    wait=False,  # Don't block; monitor asynchronously
)
```

**Endpoints.** SageMaker supports three endpoint types:

- **Real-time endpoints** provide synchronous inference with sub-second latency. They support auto-scaling, A/B testing (traffic splitting between model variants), and data capture for monitoring.
- **Serverless endpoints** provide on-demand inference without managing infrastructure. They scale to zero when idle and are cost-effective for sporadic traffic patterns. Cold start latency (typically 1--5 seconds) is the tradeoff.
- **Asynchronous endpoints** handle long-running inference requests. The client submits a request and receives a reference ID; the result is written to S3 when processing completes. Ideal for batch-like inference and large payloads.

```python
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Deploy a real-time endpoint with A/B testing
model_a = PyTorchModel(
    model_data="s3://models/model-a/model.tar.gz",
    role=role,
    framework_version="2.1.0",
    py_version="py310",
    entry_point="inference.py",
)

model_b = PyTorchModel(
    model_data="s3://models/model-b/model.tar.gz",
    role=role,
    framework_version="2.1.0",
    py_version="py310",
    entry_point="inference.py",
)

from sagemaker.model import Model
from sagemaker.predictor import Predictor

# Create endpoint with two production variants
endpoint_name = "sentiment-endpoint"

model_a_variant = model_a.deploy(
    initial_instance_count=2,
    instance_type="ml.g5.xlarge",
    endpoint_name=endpoint_name,
    variant_name="model-a",
    initial_weight=90,  # 90% of traffic
    wait=True,
)

# Add model B as a second variant (10% traffic)
model_b.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name=endpoint_name,
    variant_name="model-b",
    initial_weight=10,  # 10% of traffic
    update_endpoint=True,
)

# Configure auto-scaling
import boto3

asg_client = boto3.client("application-autoscaling")

# Register the variant as a scalable target
asg_client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/model-a",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=2,
    MaxCapacity=20,
)

# Define a target-tracking scaling policy
asg_client.put_scaling_policy(
    PolicyName="sentiment-scaling-policy",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/model-a",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 1000,  # Target: 1000 invocations per instance per minute
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60,
    },
)
```

**SageMaker Pipelines.** SageMaker Pipelines provides a purpose-built CI/CD system for ML workflows:

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    RegisterModel,
    ConditionStep,
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.processing import ScriptProcessor

# Pipeline parameters
input_data = ParameterString(name="InputData", default_value="s3://ml-data/raw/")
min_accuracy = ParameterFloat(name="MinAccuracy", default_value=0.85)

# Step 1: Data processing
processor = ScriptProcessor(
    image_uri="123456789.dkr.ecr.us-east-1.amazonaws.com/ml-processor:latest",
    role=role,
    instance_count=1,
    instance_type="ml.m5.2xlarge",
)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input",
        ),
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
        ),
        sagemaker.processing.ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
        ),
    ],
    code="src/preprocess.py",
)

# Step 2: Training
training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
        ),
    },
)

# Step 3: Evaluation
eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        sagemaker.processing.ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/output/evaluation",
        ),
    ],
    code="src/evaluate.py",
    property_files=[
        sagemaker.workflow.properties.PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation.json",
        ),
    ],
)

# Step 4: Conditional model registration
condition = ConditionGreaterThanOrEqualTo(
    left=sagemaker.workflow.functions.JsonGet(
        step_name="EvaluateModel",
        property_file="EvaluationReport",
        json_path="metrics.accuracy",
    ),
    right=min_accuracy,
)

register_step = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.g5.xlarge", "ml.g5.2xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name="SentimentModels",
    approval_status="PendingManualApproval",
)

condition_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[condition],
    if_steps=[register_step],
    else_steps=[],
)

# Create and run the pipeline
pipeline = Pipeline(
    name="sentiment-training-pipeline",
    parameters=[input_data, min_accuracy],
    steps=[processing_step, training_step, eval_step, condition_step],
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
```

**SageMaker Feature Store.** Provides both an offline store (S3-backed, for training data generation) and an online store (low-latency, for real-time serving). Feature groups define the schema and storage configuration.

**SageMaker Experiments.** Tracks training experiments with automatic logging of parameters, metrics, and artifacts. Integrates with SageMaker Pipelines for end-to-end lineage tracking.

### 21.2.2 AWS Bedrock

AWS Bedrock is a managed service for accessing foundation models from Anthropic (Claude), Meta (Llama), Amazon (Titan), and others. It provides:

```python
import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Invoke Claude via Bedrock
response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Analyze the following customer feedback and classify the sentiment as positive, negative, or neutral. Also extract key themes.\n\nFeedback: The product quality is excellent but the shipping took way too long. Customer support was helpful when I finally reached them."
            }
        ],
        "temperature": 0.0,
    }),
)

result = json.loads(response["body"].read())
print(result["content"][0]["text"])
```

**Knowledge Bases for RAG.** Bedrock Knowledge Bases provide managed Retrieval-Augmented Generation: you upload documents, Bedrock chunks and embeds them, stores the embeddings in a vector database (OpenSearch Serverless or Pinecone), and handles retrieval and generation automatically.

**Guardrails.** Bedrock Guardrails provide configurable content filtering, topic blocking, PII redaction, and grounding checks. They can be applied to any model invocation with a single configuration.

**Agents.** Bedrock Agents enable foundation models to take actions by calling APIs and processing results. You define the actions (as OpenAPI schemas), and the agent orchestrates multi-step reasoning and execution.

### 21.2.3 SageMaker HyperPod

SageMaker HyperPod provides managed distributed training clusters with built-in fault tolerance. It automatically detects and replaces faulty nodes, resumes training from the latest checkpoint, and provides Slurm-based cluster management. HyperPod is designed for large-scale training jobs (foundation model pre-training, large-scale fine-tuning) where hardware failures are common and downtime is expensive.

### 21.2.4 Supporting AWS Services

The following AWS services are frequently used alongside SageMaker:

- **S3:** Object storage for datasets, model artifacts, and checkpoints. The de facto data lake storage layer.
- **Glue:** Serverless ETL service and data catalog. Glue Crawlers automatically discover and catalog datasets; Glue Jobs run Spark-based transformations.
- **EMR (Elastic MapReduce):** Managed Spark, Hive, and Presto clusters for large-scale data processing.
- **Lambda:** Serverless compute for lightweight inference, event-driven model triggers, and glue code between services.
- **ECR (Elastic Container Registry):** Docker container registry for custom training and inference images.
- **ECS/EKS:** Container orchestration. ECS is AWS-native; EKS provides managed Kubernetes.
- **SQS/SNS:** Message queuing and notification for asynchronous workflows.
- **CloudWatch:** Monitoring, logging, and alerting for all AWS services.
- **AWS Batch:** Managed batch computing for running containerized batch jobs at scale.

---

## 21.3 GCP for ML

Google Cloud Platform's ML strategy leverages Google's deep expertise in AI research. Its primary ML service, Vertex AI, provides a unified platform for the ML lifecycle, while BigQuery and TPUs provide unique capabilities not available on other clouds.

### 21.3.1 Vertex AI

Vertex AI is GCP's unified ML platform, covering training, serving, pipelines, and model management:

```python
from google.cloud import aiplatform

aiplatform.init(
    project="my-gcp-project",
    location="us-central1",
    staging_bucket="gs://my-ml-bucket/staging/",
)

# Custom training job
job = aiplatform.CustomContainerTrainingJob(
    display_name="sentiment-training",
    container_uri="us-docker.pkg.dev/my-project/ml/trainer:latest",
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
)

model = job.run(
    model_display_name="sentiment-model-v2",
    replica_count=4,
    machine_type="n1-highmem-16",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=2,
    args=[
        "--epochs=10",
        "--batch_size=64",
        "--learning_rate=1e-4",
    ],
    environment_variables={
        "WANDB_API_KEY": "***",
        "MLFLOW_TRACKING_URI": "https://mlflow.company.com",
    },
)

# Deploy to an endpoint
endpoint = aiplatform.Endpoint.create(display_name="sentiment-endpoint")

endpoint.deploy(
    model=model,
    deployed_model_display_name="sentiment-v2",
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=10,
    traffic_percentage=100,
)

# Make predictions
prediction = endpoint.predict(
    instances=[{"text": "This product is amazing!"}]
)
print(prediction.predictions)
```

**AutoML.** Vertex AI AutoML automatically trains and tunes models for tabular, image, text, and video data. For tabular data, it combines feature engineering, model selection, and hyperparameter tuning. AutoML is useful for establishing baselines and for teams that lack deep ML expertise.

**Vertex AI Pipelines.** Built on Kubeflow Pipelines, Vertex AI Pipelines provides managed pipeline execution on GCP infrastructure:

```python
from kfp.v2 import dsl, compiler
from google.cloud import aiplatform
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "google-cloud-bigquery"],
)
def extract_data(
    project: str,
    query: str,
    output_dataset: Output[Dataset],
):
    from google.cloud import bigquery
    import pandas as pd

    client = bigquery.Client(project=project)
    df = client.query(query).to_dataframe()
    df.to_parquet(output_dataset.path)

@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"],
)
def train_model(
    train_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 200,
):
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import joblib

    df = pd.read_parquet(train_data.path)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = GradientBoostingClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics.log_metric("accuracy", accuracy_score(y_test, y_pred))
    metrics.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))

    joblib.dump(model, model_artifact.path)

@dsl.pipeline(name="ml-training-pipeline")
def ml_pipeline(
    project: str = "my-gcp-project",
    n_estimators: int = 200,
):
    extract_task = extract_data(
        project=project,
        query="SELECT * FROM `my_project.features.customer_features`",
    )
    train_task = train_model(
        train_data=extract_task.outputs["output_dataset"],
        n_estimators=n_estimators,
    )

# Compile and submit
compiler.Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path="pipeline.json",
)

pipeline_job = aiplatform.PipelineJob(
    display_name="weekly-training",
    template_path="pipeline.json",
    pipeline_root="gs://my-ml-bucket/pipeline_root/",
)
pipeline_job.submit()
```

**Matching Engine.** Vertex AI Matching Engine is a managed vector similarity search service. It supports approximate nearest neighbor search with high throughput and low latency, making it suitable for recommendation systems, semantic search, and RAG applications.

**Vertex AI Feature Store.** Provides managed feature serving with both offline (BigQuery-backed) and online (Bigtable-backed) stores. Features are defined as feature groups and served through a unified API.

### 21.3.2 BigQuery and BigQuery ML

BigQuery is Google's serverless data warehouse. **BigQuery ML** allows training and deploying ML models using SQL syntax:

```sql
-- Create a logistic regression model
CREATE OR REPLACE MODEL `my_project.ml_models.churn_model`
OPTIONS(
    model_type='BOOSTED_TREE_CLASSIFIER',
    input_label_cols=['churned'],
    max_iterations=100,
    learn_rate=0.1,
    num_parallel_tree=5,
    data_split_method='AUTO_SPLIT',
    enable_global_explain=TRUE
) AS
SELECT
    tenure_months,
    monthly_spend,
    support_tickets,
    contract_type,
    payment_method,
    churned
FROM `my_project.features.customer_features`
WHERE feature_date >= '2025-01-01';

-- Evaluate the model
SELECT *
FROM ML.EVALUATE(MODEL `my_project.ml_models.churn_model`);

-- Make predictions
SELECT
    customer_id,
    predicted_churned,
    predicted_churned_probs
FROM ML.PREDICT(
    MODEL `my_project.ml_models.churn_model`,
    (SELECT * FROM `my_project.features.customer_features` WHERE feature_date = CURRENT_DATE())
);

-- Explain predictions (feature importance)
SELECT *
FROM ML.EXPLAIN_PREDICT(
    MODEL `my_project.ml_models.churn_model`,
    (SELECT * FROM `my_project.features.customer_features` WHERE customer_id = 'C001'),
    STRUCT(3 AS top_k_features)
);

-- Export the model to Vertex AI for serving
CREATE OR REPLACE MODEL `my_project.ml_models.churn_model_export`
OPTIONS(model_type='BOOSTED_TREE_CLASSIFIER', model_registry='vertex_ai')
AS SELECT * FROM ...;
```

BigQuery ML supports linear regression, logistic regression, k-means clustering, matrix factorization, time series (ARIMA_PLUS), boosted trees, deep neural networks, and even imported TensorFlow models. For data analysts who know SQL but not Python, BigQuery ML provides a remarkably accessible entry point to ML.

**Federated Queries.** BigQuery can query data in Cloud Storage (Parquet, ORC, Avro), Cloud SQL, Cloud Spanner, and other BigQuery datasets without copying data. This is useful for ML workflows that need to join warehouse data with data lake files.

### 21.3.3 TPUs

Tensor Processing Units (TPUs) are Google's custom ASICs designed for ML workloads. They use a **systolic array** architecture optimized for matrix multiplication---the core operation in neural network training and inference.

**Architecture.** Each TPU chip contains two TensorCores, each with a 128x128 systolic array of multiply-accumulate units, 16 GB of HBM (High Bandwidth Memory), and a vector processing unit. TPU Pods connect up to 4,096 chips with a high-speed interconnect, providing exaflop-scale compute.

**When TPUs Beat GPUs.** TPUs excel at:
- Large-scale language model training (pre-training, fine-tuning)
- Workloads that are primarily matrix multiplication (transformers, CNNs)
- JAX/Flax-based workflows (which compile to XLA, TPU's native instruction set)

GPUs are better for:
- Small-scale experiments (TPUs have a minimum efficient scale)
- Workloads with irregular control flow or sparse operations
- PyTorch-centric workflows (although PyTorch/XLA is improving, the native experience is still better on GPUs)

**JAX/Flax for TPU Programming:**

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# Define a model using Flax
class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic=False):
        # Multi-head self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
        )(x, x)
        attn_output = nn.Dropout(rate=self.dropout_rate)(
            attn_output, deterministic=deterministic
        )
        x = nn.LayerNorm()(x + attn_output)

        # Feed-forward
        ff_output = nn.Dense(self.hidden_dim * 4)(x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dense(self.hidden_dim)(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate)(
            ff_output, deterministic=deterministic
        )
        x = nn.LayerNorm()(x + ff_output)
        return x

class TextClassifier(nn.Module):
    vocab_size: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    num_classes: int

    @nn.compact
    def __call__(self, x, deterministic=False):
        x = nn.Embed(self.vocab_size, self.hidden_dim)(x)
        for _ in range(self.num_layers):
            x = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
            )(x, deterministic=deterministic)
        x = jnp.mean(x, axis=1)  # Global average pooling
        x = nn.Dense(self.num_classes)(x)
        return x

# Initialize model and optimizer
model = TextClassifier(
    vocab_size=32000,
    hidden_dim=768,
    num_heads=12,
    num_layers=6,
    num_classes=2,
)

key = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
params = model.init(key, dummy_input)

# Create training state
tx = optax.adamw(learning_rate=1e-4, weight_decay=0.01)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
)

# JIT-compiled training step (compiles to XLA for TPU execution)
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply(params, batch["input_ids"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["labels"]
        ).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch["labels"])
    return state, {"loss": loss, "accuracy": accuracy}

# Use pmap for multi-device training (across TPU cores)
@jax.pmap
def distributed_train_step(state, batch):
    return train_step(state, batch)
```

**XLA Compilation.** XLA (Accelerated Linear Algebra) is the compiler that translates high-level operations into efficient machine code for TPUs (and GPUs). JAX uses XLA natively; TensorFlow supports XLA through `tf.function(jit_compile=True)`; PyTorch supports XLA through `torch_xla`.

### 21.3.4 Supporting GCP Services

- **Cloud Storage:** Object storage (equivalent to S3). The data lake foundation on GCP.
- **Dataflow:** Managed Apache Beam execution. Auto-scaling, serverless, supports both batch and streaming.
- **Dataproc:** Managed Spark and Hadoop clusters on GCP.
- **Cloud Run:** Serverless container execution. Useful for lightweight model serving and event-driven inference.
- **Pub/Sub:** Managed message queue (similar to Kafka). Integrates with Dataflow for streaming pipelines.
- **Vertex AI Workbench:** Managed JupyterLab notebooks with integration into GCP services.

---

## 21.4 Azure for ML

Microsoft Azure's ML strategy leverages its enterprise relationships and the Microsoft ecosystem (Office 365, Teams, Power BI, GitHub, VS Code). Azure ML is a comprehensive platform, and Azure OpenAI Service provides unique access to OpenAI models with enterprise features.

### 21.4.1 Azure Machine Learning

Azure ML is organized around a **workspace** that contains all resources: compute, data, models, experiments, and endpoints.

```python
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import (
    Environment,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

# Connect to workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="ml-rg",
    workspace_name="ml-workspace",
)

# Define a training job
training_job = command(
    code="./src",
    command="python train.py "
            "--data ${{inputs.training_data}} "
            "--epochs ${{inputs.epochs}} "
            "--learning_rate ${{inputs.lr}} "
            "--output ${{outputs.model}}",
    inputs={
        "training_data": Input(
            type="uri_folder",
            path="azureml://datastores/ml_data/paths/training/",
        ),
        "epochs": 10,
        "lr": 1e-4,
    },
    outputs={
        "model": Output(type="mlflow_model"),
    },
    environment=Environment(
        image="mcr.microsoft.com/azureml/pytorch-2.1-cuda12.1:latest",
        conda_file="./environment.yml",
    ),
    compute="gpu-cluster",  # Pre-created compute cluster
    instance_count=4,
    distribution={
        "type": "PyTorch",
        "process_count_per_instance": 1,
    },
    experiment_name="sentiment-analysis",
    display_name="bert-fine-tuning-v2",
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(training_job)
print(f"Job URL: {returned_job.studio_url}")

# Register the trained model
model = Model(
    path=f"azureml://jobs/{returned_job.name}/outputs/model",
    name="sentiment-model",
    description="BERT-based sentiment classifier",
    type="mlflow_model",
)
registered_model = ml_client.models.create_or_update(model)

# Deploy to a managed online endpoint
endpoint = ManagedOnlineEndpoint(
    name="sentiment-endpoint",
    auth_mode="key",
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="bert-v2",
    endpoint_name="sentiment-endpoint",
    model=registered_model.id,
    instance_type="Standard_NC4as_T4_v3",  # T4 GPU
    instance_count=2,
    environment=Environment(
        image="mcr.microsoft.com/azureml/pytorch-2.1-cuda12.1:latest",
        inference_config={
            "liveness_route": {"port": 8080, "path": "/health"},
            "readiness_route": {"port": 8080, "path": "/ready"},
            "scoring_route": {"port": 8080, "path": "/predict"},
        },
    ),
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

# Set 100% traffic to the new deployment
endpoint.traffic = {"bert-v2": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

**Azure ML Pipelines.** Azure ML supports pipeline definitions using the SDK v2:

```python
from azure.ai.ml.dsl import pipeline

@pipeline(
    compute="cpu-cluster",
    description="ML training pipeline",
)
def training_pipeline(
    training_data: Input,
    test_data: Input,
    learning_rate: float = 1e-4,
):
    preprocess_step = preprocess_component(
        input_data=training_data,
    )

    train_step = train_component(
        training_data=preprocess_step.outputs.processed_data,
        learning_rate=learning_rate,
    )
    train_step.compute = "gpu-cluster"

    evaluate_step = evaluate_component(
        model=train_step.outputs.model,
        test_data=test_data,
    )

    return {
        "model": train_step.outputs.model,
        "metrics": evaluate_step.outputs.metrics,
    }

# Submit the pipeline
pipeline_job = training_pipeline(
    training_data=Input(path="azureml://datastores/ml_data/paths/train/"),
    test_data=Input(path="azureml://datastores/ml_data/paths/test/"),
)
returned_pipeline = ml_client.jobs.create_or_update(pipeline_job)
```

**Responsible AI Dashboard.** Azure ML provides built-in tools for model interpretability, fairness assessment, error analysis, and causal inference. The Responsible AI dashboard aggregates these tools into a single interface that can be attached to any registered model.

### 21.4.2 Azure OpenAI Service

Azure OpenAI Service provides managed access to OpenAI models (GPT-4, GPT-4o, o1, DALL-E, Whisper) with enterprise features:

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="your-azure-openai-key",
    api_version="2024-06-01",
    azure_endpoint="https://your-resource.openai.azure.com",
)

# Chat completion
response = client.chat.completions.create(
    model="gpt-4o",  # This is the deployment name, not the model name
    messages=[
        {
            "role": "system",
            "content": "You are a customer support analyst. Classify the sentiment of customer feedback.",
        },
        {
            "role": "user",
            "content": "The product arrived damaged and customer support was unhelpful.",
        },
    ],
    temperature=0.0,
    max_tokens=256,
)

print(response.choices[0].message.content)

# Embeddings
embedding_response = client.embeddings.create(
    model="text-embedding-3-large",
    input=["Customer complaint about billing"],
)
embedding = embedding_response.data[0].embedding
```

**Content Filtering.** Azure OpenAI includes configurable content filters that detect and block harmful content in both inputs and outputs. Filters cover hate speech, sexual content, violence, and self-harm, each configurable at four severity levels.

**Enterprise Deployment Patterns.** Azure OpenAI runs within the customer's Azure subscription, meaning that data does not leave the Azure boundary. This is critical for regulated industries. Models can be deployed in specific Azure regions for data residency compliance. Private endpoints and virtual network integration provide network-level isolation.

### 21.4.3 Azure Synapse Analytics

Azure Synapse unifies data warehousing (dedicated SQL pools), big data (Spark pools), and data integration (pipelines) into a single workspace:

- **Dedicated SQL Pools:** MPP (massively parallel processing) data warehouse. Best for structured analytical queries.
- **Serverless SQL Pools:** On-demand SQL queries over data lake files (Parquet, CSV, JSON) without provisioning infrastructure.
- **Spark Pools:** Managed Apache Spark for data engineering, data science, and ML.
- **Pipelines:** Data integration and orchestration (based on Azure Data Factory).

### 21.4.4 Supporting Azure Services

- **Blob Storage / Data Lake Storage Gen2:** Object storage for data lakes. ADLS Gen2 adds hierarchical namespace and fine-grained access control.
- **Azure Databricks:** Managed Databricks platform on Azure. Provides Unity Catalog, Delta Live Tables, and MLflow integration.
- **Azure Functions:** Serverless compute for event-driven workloads.
- **AKS (Azure Kubernetes Service):** Managed Kubernetes for container orchestration.
- **Cosmos DB:** Multi-model, globally distributed database. Useful for feature stores (low-latency key-value access) and application data.

---

## 21.5 Infrastructure as Code

Infrastructure as Code (IaC) is the practice of defining and managing infrastructure through machine-readable configuration files rather than manual processes. For ML workloads, IaC is essential because ML infrastructure is complex (compute clusters, storage, networking, IAM, endpoints) and must be reproducible across environments (development, staging, production).

### 21.5.1 Terraform

Terraform, by HashiCorp, is the most widely adopted IaC tool. It uses a declarative language (HCL---HashiCorp Configuration Language) to define infrastructure across any cloud provider.

**Core Concepts:**

- **Providers** are plugins that interface with cloud APIs (AWS, GCP, Azure, Kubernetes, etc.).
- **Resources** are infrastructure objects (EC2 instances, S3 buckets, IAM roles).
- **Data Sources** query existing infrastructure (look up an AMI ID, find a VPC).
- **Modules** are reusable packages of related resources.
- **State** is a JSON file that maps Terraform configuration to real-world resources. State management (local, S3 backend, Terraform Cloud) is critical for team collaboration.
- **Workspaces** enable multiple state files for the same configuration (e.g., dev, staging, prod).

**ML Infrastructure Example: SageMaker Endpoint with Auto-Scaling**

```hcl
# terraform/main.tf

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "ml-terraform-state"
    key            = "ml-inference/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "model_data_url" {
  description = "S3 URL of the model artifacts"
  type        = string
}

variable "instance_type" {
  description = "SageMaker instance type for inference"
  type        = string
  default     = "ml.g5.xlarge"
}

variable "min_instance_count" {
  description = "Minimum number of instances"
  type        = number
  default     = 2
}

variable "max_instance_count" {
  description = "Maximum number of instances"
  type        = number
  default     = 20
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "sagemaker-inference-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "s3_read_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

# SageMaker Model
resource "aws_sagemaker_model" "ml_model" {
  name               = "sentiment-model-${var.environment}"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    image          = "763104351884.dkr.ecr.${var.aws_region}.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
    model_data_url = var.model_data_url
    environment = {
      SAGEMAKER_PROGRAM = "inference.py"
      MODEL_NAME        = "sentiment-bert-v2"
    }
  }

  tags = {
    Environment = var.environment
    Project     = "sentiment-analysis"
    ManagedBy   = "terraform"
  }
}

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "ml_endpoint_config" {
  name = "sentiment-endpoint-config-${var.environment}"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.ml_model.name
    initial_instance_count = var.min_instance_count
    instance_type          = var.instance_type
    initial_variant_weight = 1.0
  }

  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = 10
    destination_s3_uri          = "s3://ml-monitoring/data-capture/${var.environment}/"

    capture_options {
      capture_mode = "Input"
    }
    capture_options {
      capture_mode = "Output"
    }
  }

  tags = {
    Environment = var.environment
    Project     = "sentiment-analysis"
    ManagedBy   = "terraform"
  }
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "ml_endpoint" {
  name                 = "sentiment-endpoint-${var.environment}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.ml_endpoint_config.name

  tags = {
    Environment = var.environment
    Project     = "sentiment-analysis"
    ManagedBy   = "terraform"
  }
}

# Auto-scaling Target
resource "aws_appautoscaling_target" "sagemaker_target" {
  max_capacity       = var.max_instance_count
  min_capacity       = var.min_instance_count
  resource_id        = "endpoint/${aws_sagemaker_endpoint.ml_endpoint.name}/variant/primary"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

# Auto-scaling Policy
resource "aws_appautoscaling_policy" "sagemaker_scaling_policy" {
  name               = "sentiment-scaling-${var.environment}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker_target.resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
    target_value       = 1000
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

# CloudWatch Alarms for monitoring
resource "aws_cloudwatch_metric_alarm" "model_latency_alarm" {
  alarm_name          = "sentiment-model-latency-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "Average"
  threshold           = 500000  # 500ms in microseconds
  alarm_description   = "Model latency exceeds 500ms"
  alarm_actions       = [aws_sns_topic.ml_alerts.arn]

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.ml_endpoint.name
    VariantName  = "primary"
  }
}

resource "aws_sns_topic" "ml_alerts" {
  name = "ml-alerts-${var.environment}"
}

# Outputs
output "endpoint_name" {
  value = aws_sagemaker_endpoint.ml_endpoint.name
}

output "endpoint_arn" {
  value = aws_sagemaker_endpoint.ml_endpoint.arn
}
```

```bash
# Deploy the infrastructure
terraform init
terraform plan -var-file="production.tfvars"
terraform apply -var-file="production.tfvars"

# production.tfvars
# aws_region         = "us-east-1"
# model_data_url     = "s3://ml-models/sentiment-bert-v2/model.tar.gz"
# instance_type      = "ml.g5.xlarge"
# min_instance_count = 2
# max_instance_count = 20
# environment        = "production"
```

### 21.5.2 CloudFormation

AWS CloudFormation is AWS's native IaC service. Templates are written in YAML or JSON and define AWS resources declaratively.

**Template Anatomy:**

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: ML Inference Infrastructure

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]

  ModelDataUrl:
    Type: String
    Description: S3 URL of model artifacts

  InstanceType:
    Type: String
    Default: ml.g5.xlarge

Resources:
  SageMakerModel:
    Type: AWS::SageMaker::Model
    Properties:
      ModelName: !Sub "sentiment-model-${Environment}"
      ExecutionRoleArn: !GetAtt SageMakerRole.Arn
      PrimaryContainer:
        Image: !Sub "${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/ml-inference:latest"
        ModelDataUrl: !Ref ModelDataUrl

  SageMakerEndpointConfig:
    Type: AWS::SageMaker::EndpointConfig
    Properties:
      EndpointConfigName: !Sub "sentiment-config-${Environment}"
      ProductionVariants:
        - VariantName: primary
          ModelName: !GetAtt SageMakerModel.ModelName
          InitialInstanceCount: 2
          InstanceType: !Ref InstanceType
          InitialVariantWeight: 1.0

  SageMakerEndpoint:
    Type: AWS::SageMaker::Endpoint
    Properties:
      EndpointName: !Sub "sentiment-endpoint-${Environment}"
      EndpointConfigName: !GetAtt SageMakerEndpointConfig.EndpointConfigName

Outputs:
  EndpointName:
    Value: !GetAtt SageMakerEndpoint.EndpointName
    Export:
      Name: !Sub "${AWS::StackName}-EndpointName"
```

**Comparison to Terraform:**

| Aspect | Terraform | CloudFormation |
|---|---|---|
| **Multi-Cloud** | Yes (primary advantage) | AWS only |
| **Language** | HCL (purpose-built) | YAML/JSON |
| **State Management** | Explicit (must manage) | Managed by AWS |
| **Drift Detection** | terraform plan | Drift detection feature |
| **Modularity** | Modules, workspaces | Nested stacks, StackSets |
| **Community** | Large, multi-cloud | AWS-focused |
| **Learning Curve** | Moderate | Moderate |

For AWS-only environments, CloudFormation is a reasonable choice because AWS manages state and provides tighter integration with AWS services. For multi-cloud environments or teams that use multiple infrastructure providers, Terraform is the clear choice.

---

## 21.6 Multi-Cloud Strategies

Multi-cloud---running workloads across two or more cloud providers---is a topic that generates strong opinions. The arguments in favor and against are both compelling.

### 21.6.1 When Multi-Cloud Makes Sense

**Vendor Lock-In Mitigation.** Organizations in regulated industries may require the ability to migrate away from a provider within a defined timeframe. Multi-cloud provides an escape hatch, even if it is rarely exercised.

**Best-of-Breed Services.** Some organizations use GCP for BigQuery and TPU training, AWS for SageMaker serving and S3 storage, and Azure for OpenAI models. This approach gets the best service from each provider but significantly increases operational complexity.

**Compliance and Data Residency.** Some jurisdictions require data to remain within specific geographic boundaries. If no single provider has presence in all required regions, multi-cloud may be necessary.

**Acquisitions.** When companies merge, they often inherit different cloud platforms. Multi-cloud management becomes a practical necessity, not a strategic choice.

### 21.6.2 Challenges

**Networking.** Interconnecting cloud providers requires VPN tunnels, dedicated interconnects (e.g., AWS Direct Connect to GCP Cloud Interconnect), or public internet transit. Latency, bandwidth, and cost are all worse than intra-cloud networking.

**IAM Complexity.** Each cloud provider has its own identity and access management system. Managing permissions across AWS IAM, GCP IAM, and Azure AD is a significant operational burden. Tools like HashiCorp Vault can provide a unified secrets management layer, but identity federation remains complex.

**Data Gravity.** Data is expensive to move between clouds. A dataset stored in S3 costs nothing to access from SageMaker but incurs significant egress charges when accessed from Vertex AI. This "data gravity" tends to pull workloads toward the cloud where the data resides.

### 21.6.3 Multi-Cloud Tools

**Terraform** (HashiCorp, 2023) is the most widely used multi-cloud IaC tool. A single Terraform configuration can provision resources across multiple clouds.

**Pulumi** provides IaC using general-purpose programming languages (Python, TypeScript, Go, C#). For ML teams that prefer Python over HCL, Pulumi can be more approachable:

```python
import pulumi
import pulumi_aws as aws
import pulumi_gcp as gcp

# AWS: S3 bucket for model artifacts
model_bucket = aws.s3.Bucket(
    "model-artifacts",
    versioning=aws.s3.BucketVersioningArgs(
        enabled=True,
    ),
)

# GCP: Vertex AI training on TPUs
training_job = gcp.vertex.AiCustomJob(
    "tpu-training",
    display_name="large-model-training",
    job_spec=gcp.vertex.AiCustomJobJobSpecArgs(
        worker_pool_specs=[
            gcp.vertex.AiCustomJobJobSpecWorkerPoolSpecArgs(
                replica_count=1,
                machine_spec=gcp.vertex.AiCustomJobJobSpecWorkerPoolSpecMachineSpecArgs(
                    machine_type="ct5lp-hightpu-4t",
                    accelerator_type="TPU_V5_LITEPOD",
                    accelerator_count=4,
                ),
                container_spec=gcp.vertex.AiCustomJobJobSpecWorkerPoolSpecContainerSpecArgs(
                    image_uri="us-docker.pkg.dev/my-project/ml/trainer:latest",
                ),
            ),
        ],
    ),
    region="us-central1",
    project="my-gcp-project",
)

pulumi.export("model_bucket_name", model_bucket.id)
```

---

## 21.7 Cost Management

Cloud ML workloads are expensive. A single A100 GPU instance on AWS (`ml.p4d.24xlarge`) costs approximately $37 per hour. A 4-instance distributed training job running for 24 hours costs over $3,500. Multiply this by the number of experiments, hyperparameter searches, and failed runs, and the costs add up quickly. Effective cost management is not an afterthought---it is a core competency for any ML engineering team.

### 21.7.1 Spot and Preemptible Instances

**Spot instances** (AWS) and **preemptible VMs** (GCP) provide unused compute capacity at 60--90% discounts. The tradeoff is that the instance can be reclaimed with short notice (2 minutes on AWS, 30 seconds on GCP).

For training workloads, spot instances are highly effective because training is inherently resumable from checkpoints. The strategy is straightforward:

1. Checkpoint frequently (every N steps or every M minutes).
2. Use spot instances for training.
3. When the instance is reclaimed, automatically restart from the latest checkpoint.

```python
# SageMaker with spot instances
estimator = PyTorch(
    entry_point="train.py",
    instance_type="ml.p4d.24xlarge",
    instance_count=4,
    use_spot_instances=True,          # Enable spot
    max_wait=172800,                   # Max 48 hours waiting for capacity
    max_run=86400,                     # Max 24 hours of training
    checkpoint_s3_uri="s3://checkpoints/training/",  # Checkpoint location
    # SageMaker automatically resumes from checkpoint on spot interruption
)
```

**Cost savings by instance type (approximate):**

| Instance Type | On-Demand ($/hr) | Spot ($/hr) | Savings |
|---|---|---|---|
| ml.p4d.24xlarge (8x A100) | $37.69 | $11.31 | 70% |
| ml.g5.12xlarge (4x A10G) | $7.09 | $2.13 | 70% |
| ml.p3.16xlarge (8x V100) | $28.15 | $8.45 | 70% |
| n1-standard-8 + 1x T4 (GCP) | $1.69 | $0.51 | 70% |

Spot instances are not appropriate for real-time serving endpoints, where instance reclamation would cause service disruptions. For serving, use on-demand or reserved instances.

### 21.7.2 Reserved Instances and Savings Plans

For serving workloads that run 24/7, **reserved instances** (1-year or 3-year commitments) provide 30--60% discounts over on-demand pricing.

**AWS Savings Plans** provide similar discounts with more flexibility: instead of committing to specific instance types, you commit to a dollar amount of compute usage per hour. This is ideal for ML teams whose instance types may change as they adopt new hardware.

**GCP Committed Use Discounts (CUDs)** provide 1-year or 3-year discounts of 20--57% for committed compute resources.

### 21.7.3 Auto-Scaling Strategies

Auto-scaling is the most impactful cost optimization for serving workloads. The goal is to match capacity to demand: scale up during peak hours, scale down during off-hours, and scale to zero when possible.

**Scheduled Scaling** pre-provisions capacity before anticipated demand spikes:

```python
import boto3

asg_client = boto3.client("application-autoscaling")

# Scale up before peak hours (9 AM)
asg_client.put_scheduled_action(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/sentiment-endpoint/variant/primary",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    ScheduledActionName="scale-up-morning",
    Schedule="cron(0 9 * * ? *)",  # 9 AM daily
    ScalableTargetAction={
        "MinCapacity": 10,
        "MaxCapacity": 20,
    },
)

# Scale down after peak hours (9 PM)
asg_client.put_scheduled_action(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/sentiment-endpoint/variant/primary",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    ScheduledActionName="scale-down-evening",
    Schedule="cron(0 21 * * ? *)",  # 9 PM daily
    ScalableTargetAction={
        "MinCapacity": 2,
        "MaxCapacity": 10,
    },
)
```

**Target-Tracking Scaling** adjusts capacity to maintain a target metric:

```python
# Scale to maintain 70% GPU utilization
asg_client.put_scaling_policy(
    PolicyName="gpu-utilization-tracking",
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/sentiment-endpoint/variant/primary",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "CustomizedMetricSpecification": {
            "MetricName": "GPUUtilization",
            "Namespace": "AWS/SageMaker",
            "Dimensions": [
                {"Name": "EndpointName", "Value": "sentiment-endpoint"},
                {"Name": "VariantName", "Value": "primary"},
            ],
            "Statistic": "Average",
        },
        "TargetValue": 70.0,
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60,
    },
)
```

### 21.7.4 GPU Utilization Optimization

GPUs are the most expensive component of ML infrastructure, and they are often underutilized. Common causes and remedies include:

**Data Loading Bottlenecks.** If the GPU is idle while waiting for data, increase the number of data loading workers, use prefetching, or move data to faster storage (EBS io2 instead of gp3, or local NVMe SSD).

**Small Batch Sizes.** Small batches underutilize GPU compute. Increase batch size until GPU memory is fully utilized. If a single sample is already large (e.g., high-resolution images, long sequences), use gradient accumulation.

**Inefficient Model Architecture.** Some model architectures are not well-suited to GPU execution. Operations like dynamic control flow, irregular memory access patterns, and small tensor operations can leave the GPU idle. Profile with PyTorch Profiler or NVIDIA Nsight Systems to identify bottlenecks.

**Right-Sizing Instances.** If a model uses only 4GB of a 40GB A100's memory, you are paying for 36GB of unused capacity. Consider using a smaller GPU (T4, L4) or sharing the GPU with multiple models using NVIDIA MIG.

### 21.7.5 Cost Monitoring and Alerting

Proactive cost monitoring prevents surprise bills:

```python
import boto3

# Set up a budget alarm
budgets_client = boto3.client("budgets")

budgets_client.create_budget(
    AccountId="123456789012",
    Budget={
        "BudgetName": "ML-Monthly-Budget",
        "BudgetLimit": {
            "Amount": "50000",
            "Unit": "USD",
        },
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
        "CostFilters": {
            "TagKeyValue": [
                "user:project$sentiment-analysis",
            ],
        },
    },
    NotificationsWithSubscribers=[
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 80.0,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [
                {
                    "SubscriptionType": "EMAIL",
                    "Address": "ml-team@company.com",
                },
            ],
        },
        {
            "Notification": {
                "NotificationType": "FORECASTED",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100.0,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [
                {
                    "SubscriptionType": "EMAIL",
                    "Address": "ml-lead@company.com",
                },
            ],
        },
    ],
)
```

Best practices for cost monitoring:

1. **Tag everything.** Every resource should have tags for project, team, environment, and purpose. Tags enable cost allocation and per-project budgeting.
2. **Set up alerts at 50%, 80%, and 100% of budget.** The 50% alert provides early warning; the 80% alert triggers investigation; the 100% alert triggers corrective action.
3. **Review costs weekly.** A weekly cost review meeting (15 minutes) catches anomalies early. Look for instances that were started for experiments but never terminated, endpoints with near-zero traffic, and storage that is growing faster than expected.
4. **Use AWS Cost Explorer / GCP Billing / Azure Cost Management.** These tools provide detailed cost breakdowns by service, resource, and tag. Use them to identify the top cost drivers and target optimization efforts.
5. **Implement automated cleanup.** Write scripts that terminate idle compute resources, delete old model artifacts, and clean up orphaned infrastructure. Schedule these scripts to run daily.

---

## Exercises

1. **SageMaker End-to-End Pipeline.** Build a SageMaker Pipeline that (a) preprocesses data, (b) trains a model, (c) evaluates it, and (d) conditionally registers it if it meets a quality threshold. Deploy the registered model to an endpoint with auto-scaling.

2. **Vertex AI Training on TPU.** Write a JAX/Flax training script and deploy it as a Vertex AI Custom Training Job on TPU v3. Compare training time and cost with an equivalent GPU training job on A100.

3. **BigQuery ML.** Using BigQuery ML, train three different model types (logistic regression, boosted trees, DNN) on the same dataset. Compare their performance using BigQuery's `ML.EVALUATE` and explain predictions using `ML.EXPLAIN_PREDICT`.

4. **Azure ML Pipeline.** Build an Azure ML pipeline that trains a model, evaluates it, and deploys it to a Managed Online Endpoint. Use the Responsible AI Dashboard to analyze the model's fairness and interpretability.

5. **Terraform ML Infrastructure.** Write a Terraform module that provisions a complete ML serving stack: S3 bucket for model artifacts, ECR repository for inference images, SageMaker endpoint with auto-scaling, CloudWatch alarms, and an SNS topic for alerts. Use variable-driven configuration for dev/staging/prod environments.

6. **Cost Optimization Analysis.** For a hypothetical ML workload (specify training hours, GPU type, serving endpoint size and traffic), calculate the monthly cost on AWS, GCP, and Azure. Identify the cheapest option and quantify the savings from using spot instances for training and reserved instances for serving.

7. **Multi-Cloud Architecture.** Design a multi-cloud ML architecture that uses GCP for training (TPUs) and AWS for serving (SageMaker). Describe the data flow, networking, IAM, and CI/CD considerations. Estimate the additional cost of cross-cloud data transfer.

8. **CloudFormation vs. Terraform.** Implement the same ML infrastructure (a SageMaker endpoint with auto-scaling) in both CloudFormation and Terraform. Compare the developer experience, error handling, and state management.

---

## References

AWS Documentation. (2024). *Amazon SageMaker Developer Guide*. https://docs.aws.amazon.com/sagemaker/latest/dg/

AWS Documentation. (2024). *Amazon Bedrock User Guide*. https://docs.aws.amazon.com/bedrock/latest/userguide/

Azure Documentation. (2024). *Azure Machine Learning Documentation*. https://learn.microsoft.com/en-us/azure/machine-learning/

Azure Documentation. (2024). *Azure OpenAI Service Documentation*. https://learn.microsoft.com/en-us/azure/ai-services/openai/

Google Cloud Documentation. (2024). *Vertex AI Documentation*. https://cloud.google.com/vertex-ai/docs

Google Cloud Documentation. (2024). *BigQuery ML Documentation*. https://cloud.google.com/bigquery/docs/bqml-introduction

Google Cloud Documentation. (2024). *Cloud TPU Documentation*. https://cloud.google.com/tpu/docs

HashiCorp. (2023). *Terraform Documentation*. https://developer.hashicorp.com/terraform/docs

Jouppi, N. P., Young, C., Patil, N., Patterson, D., Agrawal, G., Bajwa, R., Bates, S., Bhatia, S., Boden, N., Borber, A., Boyle, R., Cantin, P., Chao, C., Clark, C., Coriell, J., Daley, M., Dau, M., Dean, J., Gelb, B., ... Yoon, D. H. (2017). In-Datacenter Performance Analysis of a Tensor Processing Unit. In *Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA '17)* (pp. 1--12).

Pulumi Documentation. (2024). *Pulumi: Infrastructure as Code in Any Programming Language*. https://www.pulumi.com/docs/

Synapse Documentation. (2024). *Azure Synapse Analytics Documentation*. https://learn.microsoft.com/en-us/azure/synapse-analytics/
