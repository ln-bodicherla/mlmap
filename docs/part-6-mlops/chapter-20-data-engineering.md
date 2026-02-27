# Chapter 20: Data Engineering for ML

## Learning Objectives

By the end of this chapter, you will be able to:

1. Describe the modern data stack and distinguish between data lakes, data warehouses, and lakehouses.
2. Build distributed data processing pipelines with Apache Spark and PySpark, including ML pipelines with Spark MLlib.
3. Design real-time data streaming architectures with Apache Kafka and Apache Flink.
4. Implement lakehouse architectures using Delta Lake, Apache Iceberg, and Databricks.
5. Build transformation pipelines with dbt and understand incremental models and snapshot strategies.
6. Design and deploy feature stores using Feast for consistent feature serving between training and inference.
7. Implement data quality checks with Great Expectations and enforce data contracts across organizational boundaries.
8. Query across heterogeneous data sources using federated query engines like Trino and Apache Beam.

---

## 20.1 The Modern Data Stack

The modern data stack has evolved through three generations, each driven by changing cost structures and workload requirements.

### 20.1.1 From ETL to ELT

In the traditional approach, data was **Extracted** from source systems, **Transformed** on dedicated ETL servers, and **Loaded** into a data warehouse. Transformation happened before loading because warehouse storage was expensive and compute was limited. Tools like Informatica and Talend dominated this era.

The advent of cloud data warehouses---BigQuery, Snowflake, Redshift---inverted the paradigm. Storage became cheap and compute became elastic. The new approach, **ELT**, loads raw data into the warehouse first and transforms it there, using the warehouse's own compute resources. This separation of ingestion from transformation provides several advantages: raw data is always preserved, transformations can be rerun as requirements change, and transformation logic lives in version-controlled SQL rather than in proprietary ETL tools.

### 20.1.2 Data Lake vs. Data Warehouse vs. Lakehouse

**Data Lakes** store raw data in open formats (Parquet, ORC, JSON, CSV) on object storage (S3, GCS, ADLS). They are cheap, flexible, and can store any data type. But they lack ACID transactions, schema enforcement, and query optimization---problems collectively known as the "data swamp" phenomenon.

**Data Warehouses** provide structured, schema-enforced storage with powerful query engines. They excel at analytical queries over structured data. But they are expensive, support only structured data, and create vendor lock-in because data is stored in proprietary formats.

**Lakehouses** combine the best of both worlds: data is stored in open formats on object storage (lake economics and flexibility), but a metadata and transaction layer provides ACID transactions, schema enforcement, and query optimization (warehouse semantics). Delta Lake, Apache Iceberg, and Apache Hudi are the three leading lakehouse table formats. The lakehouse architecture has become the dominant paradigm for modern data platforms (Armbrust et al., 2021).

### 20.1.3 Batch vs. Streaming

**Batch processing** operates on bounded datasets. Data is collected over a period (hourly, daily), processed as a unit, and the results are written to storage. Batch processing is well-suited for training data preparation, feature engineering over historical data, and periodic model retraining.

**Stream processing** operates on unbounded, continuously arriving data. Events are processed as they arrive (or with a small delay), and results are available in near-real-time. Stream processing is essential for real-time feature computation, online fraud detection, and any application where latency matters.

Most production ML systems require both: batch processing for training data and historical features, and stream processing for real-time features and predictions. The **Lambda architecture** runs separate batch and streaming pipelines, merging their results at query time. The **Kappa architecture** processes everything through a streaming pipeline, using the stream's replay capability to handle historical data. The choice between them depends on the complexity of the batch logic and the team's familiarity with streaming frameworks.

### 20.1.4 The Data Platform for ML

A well-designed data platform for ML typically consists of the following layers, each serving a distinct purpose:

**Ingestion Layer.** Data arrives from operational databases (via change data capture tools like Debezium), application events (via Kafka or Pub/Sub), third-party APIs (via scheduled extractors), and file uploads. The ingestion layer must handle schema evolution, exactly-once delivery, and backpressure.

**Storage Layer.** Raw data lands in a data lake (S3, GCS, ADLS) in open formats. A lakehouse table format (Delta Lake, Iceberg) provides ACID transactions and time travel. A data warehouse (Snowflake, BigQuery, Redshift) provides fast analytical queries for structured data.

**Transformation Layer.** dbt, Spark, or Flink transform raw data into clean, validated, feature-ready datasets. The transformation layer enforces data quality and business logic.

**Feature Layer.** A feature store (Feast, Tecton) provides consistent feature access for both training and serving. The feature layer computes and stores features, manages point-in-time correctness, and serves features with low latency.

**Consumption Layer.** ML training pipelines read historical features for model training. Serving endpoints read real-time features for inference. BI tools read aggregated data for dashboards and reports.

The relationship between these layers forms a directed graph: data flows from ingestion to storage to transformation to features to consumption, with monitoring and data quality checks at each boundary. The tools described in this chapter populate each layer; the art of data engineering lies in choosing the right tools and connecting them reliably.

---

## 20.2 Apache Spark and PySpark

Apache Spark (Zaharia et al., 2016) is the most widely used distributed data processing framework. It provides a unified engine for batch processing, stream processing, SQL queries, and machine learning.

### 20.2.1 Architecture

Spark follows a master-worker architecture:

- **The Driver** is the master process. It runs the user's application, constructs the execution plan, and coordinates work across the cluster.
- **Executors** are worker processes that run on cluster nodes. Each executor runs tasks, stores data in memory or on disk, and reports results to the driver.
- **Cluster Managers** (YARN, Mesos, Kubernetes, or Spark Standalone) allocate resources to Spark applications.

When a Spark application runs, the driver translates the user's code into a **DAG (Directed Acyclic Graph)** of stages and tasks. Each stage consists of tasks that can be executed in parallel on different partitions of the data. Stage boundaries occur at **shuffles**, where data must be redistributed across the cluster.

### 20.2.2 RDDs, DataFrames, and Transformations

**Resilient Distributed Datasets (RDDs)** are Spark's original abstraction: distributed collections of objects that can be transformed in parallel. While RDDs provide maximum flexibility, they lack the optimization benefits of higher-level APIs.

**DataFrames** are the modern interface. They represent distributed tables with named columns and known types, enabling Spark's Catalyst optimizer to generate efficient execution plans:

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("ML Feature Engineering") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

# Read data
transactions = spark.read.parquet("s3://data-lake/transactions/")
customers = spark.read.parquet("s3://data-lake/customers/")

# Transformations are lazy — nothing executes until an action is called
customer_features = (
    transactions
    .filter(F.col("transaction_date") >= F.lit("2025-01-01"))
    .groupBy("customer_id")
    .agg(
        F.count("*").alias("transaction_count"),
        F.sum("amount").alias("total_spend"),
        F.avg("amount").alias("avg_transaction_amount"),
        F.stddev("amount").alias("std_transaction_amount"),
        F.max("transaction_date").alias("last_transaction_date"),
        F.countDistinct("merchant_category").alias("unique_categories"),
    )
)

# Window functions for time-based features
window_30d = (
    Window
    .partitionBy("customer_id")
    .orderBy(F.col("transaction_date").cast("long"))
    .rangeBetween(-30 * 86400, 0)  # 30 days in seconds
)

transactions_with_windows = transactions.withColumn(
    "rolling_30d_spend",
    F.sum("amount").over(window_30d)
).withColumn(
    "rolling_30d_count",
    F.count("*").over(window_30d)
)

# Broadcast join for small dimension tables
customer_enriched = customer_features.join(
    F.broadcast(customers),
    on="customer_id",
    how="left",
)

# Action: write the result
customer_enriched.write \
    .mode("overwrite") \
    .partitionBy("signup_year") \
    .parquet("s3://feature-store/customer_features/")
```

### 20.2.3 Key Concepts for Performance

**Lazy Evaluation.** Transformations (filter, map, join, groupBy) are not executed immediately. Spark records them in a DAG and optimizes the entire pipeline before execution. This enables optimizations like predicate pushdown (applying filters as early as possible), column pruning (reading only needed columns), and whole-stage code generation.

**Shuffles and Partitioning.** Shuffles redistribute data across the cluster and are the most expensive operation in Spark. They occur during groupBy, join, distinct, and repartition operations. Minimizing shuffles---through broadcast joins for small tables, proper partitioning, and pre-aggregation---is the most important performance optimization.

**Broadcast Joins.** When one table is small enough to fit in memory on each executor (typically < 100MB, configurable via `spark.sql.autoBroadcastJoinThreshold`), Spark can broadcast it to all executors, eliminating the shuffle of the larger table.

**Adaptive Query Execution (AQE).** Introduced in Spark 3.0, AQE optimizes the query plan at runtime based on actual data statistics. It can coalesce small partitions after shuffles, switch join strategies based on actual table sizes, and optimize skewed joins by splitting large partitions.

### 20.2.4 Spark MLlib Pipeline

Spark MLlib provides distributed implementations of common ML algorithms:

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
)
from pyspark.ml.classification import GradientBoostedTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Prepare features
categorical_cols = ["contract_type", "payment_method"]
numerical_cols = ["tenure_months", "monthly_spend", "support_tickets"]

# Build pipeline stages
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
    for c in categorical_cols
]
encoders = [
    OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_encoded")
    for c in categorical_cols
]

assembler = VectorAssembler(
    inputCols=[f"{c}_encoded" for c in categorical_cols] + numerical_cols,
    outputCol="features_raw",
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True,
)

gbt = GradientBoostedTreeClassifier(
    labelCol="churned",
    featuresCol="features",
    maxIter=100,
)

pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, gbt])

# Hyperparameter tuning with cross-validation
param_grid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [3, 5, 7])
    .addGrid(gbt.stepSize, [0.05, 0.1, 0.2])
    .addGrid(gbt.subsamplingRate, [0.8, 1.0])
    .build()
)

evaluator = BinaryClassificationEvaluator(
    labelCol="churned",
    metricName="areaUnderROC",
)

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=4,
)

# Train and evaluate
train_data, test_data = customer_enriched.randomSplit([0.8, 0.2], seed=42)
cv_model = cv.fit(train_data)

predictions = cv_model.transform(test_data)
auc = evaluator.evaluate(predictions)
print(f"Test AUC: {auc:.4f}")

# Save the best model
cv_model.bestModel.write().overwrite().save("s3://models/churn-gbt/")
```

---

## 20.3 Apache Kafka

Apache Kafka (Kreps et al., 2011) is a distributed event streaming platform that serves as the backbone for real-time data pipelines. In ML systems, Kafka enables real-time feature computation, event-driven model inference, and data integration between heterogeneous systems.

### 20.3.1 Architecture

**Brokers** are the servers that store and serve data. A Kafka cluster consists of multiple brokers for fault tolerance and scalability.

**Topics** are named streams of records. Each topic is divided into **partitions**, which are the unit of parallelism. Records within a partition are ordered and assigned a monotonically increasing **offset**.

**Producers** write records to topics. Records are assigned to partitions based on a key (hash partitioning) or round-robin (if no key is specified).

**Consumer Groups** enable parallel consumption. Each partition is assigned to exactly one consumer in the group, and Kafka tracks each consumer's offset to enable exactly-once processing and fault recovery.

### 20.3.2 Exactly-Once Semantics

Achieving exactly-once semantics in Kafka requires two mechanisms:

**Idempotent Producers.** When `enable.idempotence=true`, the producer assigns a sequence number to each message. The broker deduplicates messages with the same sequence number, preventing duplicates caused by retries.

**Transactional Producers.** Transactions allow atomic writes across multiple partitions and topics. The producer begins a transaction, writes messages, commits (or aborts), and consumers configured with `isolation.level=read_committed` see only committed messages.

```python
from confluent_kafka import Producer, Consumer, KafkaError
import json

# Producer with exactly-once semantics
producer_config = {
    "bootstrap.servers": "kafka-broker:9092",
    "enable.idempotence": True,
    "acks": "all",
    "retries": 10,
    "max.in.flight.requests.per.connection": 5,
}
producer = Producer(producer_config)

def produce_features(user_id: str, features: dict):
    """Produce feature events to Kafka."""
    event = {
        "user_id": user_id,
        "features": features,
        "timestamp": int(time.time() * 1000),
    }
    producer.produce(
        topic="ml-features",
        key=user_id.encode("utf-8"),
        value=json.dumps(event).encode("utf-8"),
        callback=delivery_callback,
    )
    producer.flush()

def delivery_callback(err, msg):
    if err is not None:
        print(f"Delivery failed: {err}")
    else:
        print(f"Delivered to {msg.topic()}[{msg.partition()}]@{msg.offset()}")

# Consumer for real-time ML inference
consumer_config = {
    "bootstrap.servers": "kafka-broker:9092",
    "group.id": "ml-inference-group",
    "auto.offset.reset": "latest",
    "enable.auto.commit": False,
    "isolation.level": "read_committed",
}
consumer = Consumer(consumer_config)
consumer.subscribe(["ml-features"])

# Real-time inference loop
import joblib
model = joblib.load("model.pkl")

while True:
    msg = consumer.poll(timeout=1.0)
    if msg is None:
        continue
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            continue
        raise KafkaError(msg.error())

    event = json.loads(msg.value().decode("utf-8"))
    features = event["features"]

    # Run inference
    prediction = model.predict([list(features.values())])[0]

    # Produce prediction to output topic
    result = {
        "user_id": event["user_id"],
        "prediction": int(prediction),
        "model_version": "v2.3",
        "timestamp": int(time.time() * 1000),
    }
    producer.produce(
        topic="ml-predictions",
        key=event["user_id"].encode("utf-8"),
        value=json.dumps(result).encode("utf-8"),
    )

    # Commit offset after successful processing
    consumer.commit(asynchronous=False)
```

### 20.3.3 Kafka Connect and Schema Registry

**Kafka Connect** provides pre-built connectors for moving data between Kafka and external systems (databases, S3, Elasticsearch, etc.) without writing code. Source connectors ingest data into Kafka; sink connectors write data from Kafka to external systems.

**Schema Registry** stores and manages Avro, Protobuf, or JSON schemas for Kafka topics. It enforces schema evolution rules (backward compatibility, forward compatibility, full compatibility), preventing producers from publishing messages that consumers cannot parse. This is critical for ML feature pipelines, where a schema change in an upstream feature can silently corrupt downstream models.

---

## 20.4 Delta Lake

Delta Lake (Armbrust et al., 2020) is an open-source storage layer that brings ACID transactions to data lakes. Built on top of Apache Spark, it stores data in Parquet format with a transaction log that tracks every change.

### 20.4.1 Key Features

**ACID Transactions.** Delta Lake uses an optimistic concurrency control protocol based on a write-ahead transaction log. Multiple writers can operate concurrently on the same table, and conflicts are detected and resolved automatically.

**Time Travel.** Every committed transaction creates a new version of the table. You can query any historical version:

```python
# Read the table as of a specific version
df = spark.read.format("delta").option("versionAsOf", 42).load("s3://lake/customers/")

# Read the table as of a specific timestamp
df = spark.read.format("delta").option("timestampAsOf", "2025-06-15").load("s3://lake/customers/")
```

This is invaluable for ML reproducibility: you can always recreate the exact training dataset used for a particular model version.

**Z-Ordering.** Z-ordering co-locates related data in the same files, optimizing query performance for common filter patterns:

```python
# Optimize the table with Z-ordering on frequently filtered columns
spark.sql("""
    OPTIMIZE delta.`s3://lake/transactions/`
    ZORDER BY (customer_id, transaction_date)
""")
```

### 20.4.2 The Medallion Architecture

The **Medallion Architecture** (also called the multi-hop architecture) organizes data in three layers:

**Bronze (Raw).** Raw data is ingested as-is from source systems. No transformations are applied. This layer serves as the single source of truth and enables reprocessing when transformation logic changes.

**Silver (Cleaned).** Data is cleaned, deduplicated, and schema-enforced. Invalid records are quarantined for investigation. Common operations include type casting, null handling, and joining with reference data.

**Gold (Aggregated).** Business-level aggregations, feature tables, and analytics-ready datasets. This is where ML feature tables live.

```python
# Bronze: Ingest raw data with Auto Loader
bronze_df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", "s3://lake/_schemas/events/")
    .load("s3://raw-events/")
)

bronze_df.writeStream \
    .format("delta") \
    .option("checkpointLocation", "s3://lake/_checkpoints/bronze_events/") \
    .outputMode("append") \
    .table("bronze.events")

# Silver: Clean and validate
silver_df = (
    spark.readStream
    .table("bronze.events")
    .filter(F.col("event_type").isNotNull())
    .withColumn("event_timestamp", F.to_timestamp("event_timestamp"))
    .withColumn("amount", F.col("amount").cast("double"))
    .dropDuplicates(["event_id"])
)

silver_df.writeStream \
    .format("delta") \
    .option("checkpointLocation", "s3://lake/_checkpoints/silver_events/") \
    .outputMode("append") \
    .table("silver.events")

# Gold: Feature aggregations
gold_features = (
    spark.read.table("silver.events")
    .groupBy("customer_id")
    .agg(
        F.count("*").alias("event_count_all_time"),
        F.sum(F.when(F.col("event_type") == "purchase", F.col("amount"))).alias("total_purchases"),
        F.avg("amount").alias("avg_transaction_amount"),
        F.max("event_timestamp").alias("last_event_timestamp"),
    )
)

gold_features.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold.customer_features")
```

### 20.4.3 Change Data Feed

Delta Lake's **Change Data Feed (CDF)** captures row-level changes (inserts, updates, deletes) as they occur. This enables incremental processing: instead of reprocessing the entire table, downstream consumers can read only the changes since their last checkpoint.

```python
# Enable CDF on a table
spark.sql("ALTER TABLE silver.customers SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# Read only the changes since version 10
changes = (
    spark.read.format("delta")
    .option("readChangeFeed", "true")
    .option("startingVersion", 10)
    .table("silver.customers")
)

# Each row has _change_type: insert, update_preimage, update_postimage, delete
new_and_updated = changes.filter(
    F.col("_change_type").isin("insert", "update_postimage")
)
```

### 20.4.4 Vacuum and Optimization

Delta Lake tables accumulate old data files over time. The `VACUUM` command removes files that are no longer referenced by the transaction log:

```python
# Remove files older than 7 days (default retention period)
spark.sql("VACUUM delta.`s3://lake/transactions/` RETAIN 168 HOURS")
```

The `OPTIMIZE` command compacts small files into larger ones for better query performance:

```python
spark.sql("OPTIMIZE delta.`s3://lake/transactions/`")
```

These maintenance operations should be scheduled regularly (daily or weekly) to prevent performance degradation.

---

## 20.5 Databricks

Databricks, founded by the creators of Apache Spark, provides a unified analytics platform built on the lakehouse architecture. While much of Databricks' functionality can be achieved with open-source tools, its integrated experience and managed infrastructure reduce operational overhead significantly.

### 20.5.1 Unity Catalog

**Unity Catalog** provides centralized data governance across all Databricks workspaces. It introduces a three-level namespace: `catalog.schema.table`. Access control is defined at any level of the hierarchy, and audit logs track every data access.

```sql
-- Create a catalog for the ML team
CREATE CATALOG ml_platform;

-- Grant access to the data science team
GRANT USE CATALOG ON CATALOG ml_platform TO `data-science-team`;

-- Create a schema for feature tables
CREATE SCHEMA ml_platform.features;

-- Grant read access to the feature schema
GRANT SELECT ON SCHEMA ml_platform.features TO `ml-inference-service`;
```

### 20.5.2 Auto Loader

**Auto Loader** incrementally ingests new files as they arrive in cloud storage. It uses file notification (via cloud events) or directory listing to detect new files, and maintains a checkpoint to ensure exactly-once processing:

```python
raw_df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", "/mnt/schemas/events/")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
    .load("/mnt/raw-events/")
)

raw_df.writeStream \
    .format("delta") \
    .option("checkpointLocation", "/mnt/checkpoints/bronze_events/") \
    .option("mergeSchema", "true") \
    .trigger(availableNow=True)  # Process all available files then stop
    .toTable("bronze.events")
```

### 20.5.3 Delta Live Tables

**Delta Live Tables (DLT)** provides a declarative framework for building data pipelines. You define the desired state of each table, and DLT handles execution, dependency resolution, error handling, and data quality:

```python
import dlt
from pyspark.sql import functions as F

@dlt.table(
    comment="Raw events from mobile app",
    table_properties={"quality": "bronze"},
)
def bronze_events():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .load("/mnt/raw-events/")
    )

@dlt.table(
    comment="Cleaned and validated events",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("valid_event_type", "event_type IS NOT NULL")
@dlt.expect_or_drop("valid_amount", "amount >= 0")
@dlt.expect("valid_timestamp", "event_timestamp IS NOT NULL")
def silver_events():
    return (
        dlt.read_stream("bronze_events")
        .withColumn("event_timestamp", F.to_timestamp("event_timestamp"))
        .withColumn("amount", F.col("amount").cast("double"))
        .dropDuplicates(["event_id"])
    )

@dlt.table(
    comment="Customer feature table for ML",
    table_properties={"quality": "gold"},
)
def gold_customer_features():
    return (
        dlt.read("silver_events")
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("total_events"),
            F.sum("amount").alias("total_spend"),
            F.avg("amount").alias("avg_transaction"),
        )
    )
```

The `@dlt.expect` decorators define data quality rules. `expect_or_drop` drops rows that violate the constraint; `expect_or_fail` fails the pipeline; plain `expect` logs violations but does not drop rows.

### 20.5.4 MLflow Integration and Feature Store

Databricks provides a managed MLflow instance with deep integration into the platform. The **Databricks Feature Store** extends MLflow with feature-specific capabilities: point-in-time lookups, automatic feature lineage tracking, and seamless integration between offline (batch) and online (real-time) feature serving.

---

## 20.6 dbt (Data Build Tool)

dbt (data build tool) has become the standard for SQL-based data transformations in the modern data stack. It treats SQL queries as software: transformations are version-controlled, tested, documented, and deployed through CI/CD pipelines.

### 20.6.1 Core Concepts

**Models** are SQL SELECT statements that define transformations. Each model produces a table or view in the warehouse:

```sql
-- models/staging/stg_transactions.sql
{{ config(materialized='view') }}

SELECT
    transaction_id,
    customer_id,
    CAST(amount AS DECIMAL(10, 2)) AS amount,
    CAST(transaction_timestamp AS TIMESTAMP) AS transaction_timestamp,
    merchant_category,
    CASE
        WHEN payment_method IN ('credit_card', 'debit_card') THEN 'card'
        WHEN payment_method = 'bank_transfer' THEN 'bank'
        ELSE 'other'
    END AS payment_type
FROM {{ source('raw', 'transactions') }}
WHERE transaction_timestamp IS NOT NULL
  AND amount > 0
```

**Sources** declare the raw tables that dbt reads from. **Refs** declare dependencies between models: `{{ ref('stg_transactions') }}` creates a dependency on the `stg_transactions` model, and dbt uses these dependencies to determine execution order.

### 20.6.2 Feature Engineering with dbt

dbt is well-suited for feature engineering because it can express complex aggregations, window functions, and joins in SQL:

```sql
-- models/features/customer_features.sql
{{ config(
    materialized='incremental',
    unique_key='customer_id',
    on_schema_change='sync_all_columns'
) }}

WITH transaction_features AS (
    SELECT
        customer_id,
        COUNT(*) AS transaction_count_all_time,
        SUM(amount) AS total_spend,
        AVG(amount) AS avg_transaction_amount,
        STDDEV(amount) AS std_transaction_amount,
        MAX(transaction_timestamp) AS last_transaction_timestamp,
        COUNT(DISTINCT merchant_category) AS unique_merchant_categories,
        -- Recency features
        DATEDIFF('day', MAX(transaction_timestamp), CURRENT_TIMESTAMP()) AS days_since_last_transaction,
        -- Frequency features (last 30 days)
        COUNT(CASE WHEN transaction_timestamp >= DATEADD('day', -30, CURRENT_TIMESTAMP()) THEN 1 END)
            AS transactions_last_30d,
        SUM(CASE WHEN transaction_timestamp >= DATEADD('day', -30, CURRENT_TIMESTAMP()) THEN amount ELSE 0 END)
            AS spend_last_30d,
        -- Monetary features
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median_transaction_amount,
        MAX(amount) AS max_transaction_amount
    FROM {{ ref('stg_transactions') }}
    {% if is_incremental() %}
    WHERE transaction_timestamp > (SELECT MAX(last_transaction_timestamp) FROM {{ this }})
    {% endif %}
    GROUP BY customer_id
)

SELECT
    t.*,
    c.signup_date,
    DATEDIFF('month', c.signup_date, CURRENT_TIMESTAMP()) AS tenure_months,
    c.contract_type,
    c.payment_method
FROM transaction_features t
LEFT JOIN {{ ref('stg_customers') }} c
    ON t.customer_id = c.customer_id
```

### 20.6.3 Testing

dbt provides built-in tests and supports custom tests:

```yaml
# models/features/schema.yml
version: 2

models:
  - name: customer_features
    description: "Customer-level features for churn prediction"
    columns:
      - name: customer_id
        description: "Unique customer identifier"
        tests:
          - unique
          - not_null

      - name: total_spend
        description: "Total lifetime spend"
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0

      - name: avg_transaction_amount
        description: "Average transaction amount"
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              max_value: 100000

      - name: contract_type
        description: "Customer contract type"
        tests:
          - accepted_values:
              values: ['monthly', 'annual', 'two_year']

      - name: tenure_months
        description: "Months since customer signup"
        tests:
          - dbt_utils.accepted_range:
              min_value: 0
```

### 20.6.4 Incremental Models and Snapshots

**Incremental models** process only new or changed data, avoiding the cost of reprocessing the entire dataset on every run. The `is_incremental()` macro enables conditional logic that filters for new records.

**Snapshots** capture the state of a source table at each point in time, implementing Slowly Changing Dimension Type 2 (SCD2). This is essential for ML features that depend on historical states (e.g., what was the customer's contract type at the time of each transaction?).

```sql
-- snapshots/customer_snapshot.sql
{% snapshot customer_snapshot %}

{{ config(
    target_schema='snapshots',
    unique_key='customer_id',
    strategy='timestamp',
    updated_at='updated_at',
) }}

SELECT * FROM {{ source('raw', 'customers') }}

{% endsnapshot %}
```

---

## 20.7 Apache Iceberg

Apache Iceberg is an open table format for large analytic datasets, designed to address the limitations of Hive-style table management. Originally developed at Netflix, it has gained adoption as a vendor-neutral alternative to Delta Lake (which is closely associated with Databricks).

### 20.7.1 Key Features

**Hidden Partitioning.** Unlike Hive tables, where the user must understand the physical partition layout to write efficient queries, Iceberg handles partitioning transparently. The table definition specifies partition transforms (year, month, day, hour, bucket, truncate), and the query engine automatically applies partition pruning:

```sql
CREATE TABLE events (
    event_id BIGINT,
    event_timestamp TIMESTAMP,
    user_id STRING,
    event_type STRING,
    amount DECIMAL(10, 2)
) USING iceberg
PARTITIONED BY (days(event_timestamp), bucket(16, user_id));

-- Queries automatically benefit from partitioning without specifying partition columns
SELECT * FROM events WHERE event_timestamp > '2025-06-01' AND user_id = 'user_123';
```

**Schema Evolution.** Iceberg supports adding, dropping, renaming, and reordering columns without rewriting data. Column types can be widened (e.g., int to long) without breaking existing queries.

**Time Travel.** Like Delta Lake, Iceberg maintains a history of table snapshots:

```sql
-- Query a specific snapshot
SELECT * FROM events TIMESTAMP AS OF '2025-06-15 10:00:00';

-- Roll back to a previous version
CALL catalog.system.rollback_to_snapshot('db.events', 12345678901234);
```

**Branching and Tagging.** Iceberg supports named branches and tags, enabling write isolation (multiple writers can write to different branches and merge later) and reproducibility (tag a snapshot as `training-data-v3` and reference it by name).

### 20.7.2 Comparison to Delta Lake and Hudi

| Feature | Delta Lake | Iceberg | Hudi |
|---|---|---|---|
| **ACID Transactions** | Yes | Yes | Yes |
| **Time Travel** | Yes | Yes | Yes |
| **Schema Evolution** | Limited | Full | Limited |
| **Hidden Partitioning** | No | Yes | No |
| **Engine Support** | Spark-centric | Multi-engine | Spark-centric |
| **Branching** | No | Yes | No |
| **Governance** | Databricks | Open | Open |

Iceberg's primary advantage is its engine neutrality: it works equally well with Spark, Flink, Trino, Dremio, and Snowflake. Delta Lake's primary advantage is its tight integration with Databricks and the broader Spark ecosystem. The choice often depends on the organization's existing infrastructure and vendor relationships.

---

## 20.8 Apache Flink

Apache Flink is a distributed stream processing framework that provides true event-at-a-time processing with exactly-once guarantees. Where Spark Structured Streaming processes data in micro-batches (typically 100ms to seconds), Flink processes each event individually, achieving lower latency.

### 20.8.1 Core Concepts

**DataStream API.** Flink's primary API for stream processing:

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaOffsetsInitializer,
    KafkaSink,
    KafkaRecordSerializationSchema,
)
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common import WatermarkStrategy, Duration

env = StreamExecutionEnvironment.get_execution_environment()

# Configure Kafka source
kafka_source = (
    KafkaSource.builder()
    .set_bootstrap_servers("kafka-broker:9092")
    .set_topics("user-events")
    .set_group_id("flink-ml-features")
    .set_starting_offsets(KafkaOffsetsInitializer.latest())
    .set_value_only_deserializer(SimpleStringSchema())
    .build()
)

# Define watermark strategy for event-time processing
watermark_strategy = (
    WatermarkStrategy
    .for_bounded_out_of_orderness(Duration.of_seconds(10))
    .with_timestamp_assigner(lambda event, ts: extract_timestamp(event))
)

# Create the data stream
events = env.from_source(
    kafka_source,
    watermark_strategy,
    "Kafka Source",
)
```

### 20.8.2 Event Time, Watermarks, and Windows

**Event Time vs. Processing Time.** Event time is when the event actually occurred; processing time is when the system processes it. For ML features, event time is almost always the correct choice---you want features based on when events happened, not when they were processed.

**Watermarks** are Flink's mechanism for tracking event-time progress. A watermark with timestamp $t$ asserts that no events with timestamp $< t$ will arrive in the future. Watermarks enable Flink to close windows and emit results even when events arrive out of order.

**Windows** aggregate events over time:

```python
from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows
from pyflink.common import Time

# Tumbling window: non-overlapping 5-minute aggregations
windowed_features = (
    events
    .key_by(lambda e: e["user_id"])
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(
        FeatureAggregator(),  # Custom aggregation function
        output_type=Types.ROW([
            Types.STRING(),  # user_id
            Types.FLOAT(),   # avg_amount
            Types.INT(),     # event_count
            Types.FLOAT(),   # max_amount
        ]),
    )
)

# Sliding window: 1-hour aggregations, updated every 5 minutes
sliding_features = (
    events
    .key_by(lambda e: e["user_id"])
    .window(SlidingEventTimeWindows.of(Time.hours(1), Time.minutes(5)))
    .aggregate(FeatureAggregator())
)
```

### 20.8.3 Comparison to Spark Structured Streaming

| Aspect | Flink | Spark Structured Streaming |
|---|---|---|
| **Processing Model** | True streaming (event-at-a-time) | Micro-batch |
| **Latency** | Milliseconds | Seconds (micro-batch interval) |
| **State Management** | Built-in, RocksDB-backed | Built-in, HDFS-backed |
| **Event Time** | First-class support | Good support (since Spark 2.1) |
| **SQL Support** | Flink SQL (mature) | Spark SQL (mature) |
| **Batch Processing** | Supported (unified) | Native strength |
| **Ecosystem** | Growing | Very mature |
| **Ease of Use** | Moderate | Easier (Spark knowledge transfers) |

Flink is the better choice when low latency (sub-second) is a hard requirement, when complex event processing is needed, or when the event-time semantics must be very precise. Spark Structured Streaming is the better choice when the team already knows Spark, when batch and streaming workloads must share code, or when the latency requirements are in the seconds-to-minutes range.

---

## 20.9 Feature Stores

Feature stores solve three problems that plague ML systems without them: **feature reuse** (computing the same feature in multiple models), **training-serving skew** (using different feature computation logic in training and serving), and **point-in-time correctness** (ensuring that training data reflects only information that was available at the time of each training example).

### 20.9.1 Why Feature Stores Matter

Consider a fraud detection model that uses the feature "average transaction amount in the last 30 days." Without a feature store:

- The training pipeline computes this feature using a SQL query against the data warehouse.
- The serving pipeline computes it using a different SQL query (or Python code) against the production database.
- The two implementations diverge over time, causing training-serving skew.
- During training, the feature is computed using data from the future (data leakage) because the query does not enforce point-in-time correctness.

A feature store centralizes feature definitions and provides consistent access for both training and serving, eliminating these problems.

### 20.9.2 Feast

Feast (Feature Store) is the leading open-source feature store. Its architecture consists of three components:

- **Offline Store:** Stores historical feature values for training data generation. Typically backed by BigQuery, Snowflake, Redshift, or a file store.
- **Online Store:** Stores the latest feature values for low-latency serving. Typically backed by Redis, DynamoDB, or an in-memory store.
- **Feature Server:** An HTTP API that serves features for real-time inference.

```python
# feature_definitions.py
from feast import Entity, Feature, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String
from datetime import timedelta

# Define entities
customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    description="Unique customer identifier",
)

# Define data sources
customer_features_source = FileSource(
    path="s3://feature-store/customer_features.parquet",
    timestamp_field="feature_timestamp",
    created_timestamp_column="created_at",
)

# Define feature views
customer_transaction_features = FeatureView(
    name="customer_transaction_features",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="transaction_count_30d", dtype=Int64),
        Field(name="total_spend_30d", dtype=Float32),
        Field(name="avg_transaction_amount", dtype=Float32),
        Field(name="std_transaction_amount", dtype=Float32),
        Field(name="unique_merchant_categories", dtype=Int64),
        Field(name="days_since_last_transaction", dtype=Int64),
    ],
    source=customer_features_source,
    online=True,
)

customer_profile_features = FeatureView(
    name="customer_profile_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="tenure_months", dtype=Int64),
        Field(name="contract_type", dtype=String),
        Field(name="payment_method", dtype=String),
    ],
    source=customer_profile_source,
    online=True,
)
```

### 20.9.3 Materialization and Point-in-Time Joins

**Materialization** copies feature data from the offline store to the online store:

```bash
# Materialize features from offline to online store
feast materialize 2025-01-01T00:00:00 2025-12-31T23:59:59

# Incremental materialization (only new data)
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
```

**Point-in-time joins** are the most important feature of a feature store. When generating training data, Feast joins features to entity data as of the timestamp of each entity row, ensuring that no future information leaks into the training data:

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

# Entity DataFrame: who and when
entity_df = pd.DataFrame({
    "customer_id": ["C001", "C002", "C003", "C001"],
    "event_timestamp": pd.to_datetime([
        "2025-06-01", "2025-06-15", "2025-07-01", "2025-08-01"
    ]),
    "label": [0, 1, 0, 1],
})

# Get historical features with point-in-time correctness
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_transaction_features:transaction_count_30d",
        "customer_transaction_features:total_spend_30d",
        "customer_transaction_features:avg_transaction_amount",
        "customer_profile_features:tenure_months",
        "customer_profile_features:contract_type",
    ],
).to_df()

# For online serving (low-latency, latest features)
online_features = store.get_online_features(
    features=[
        "customer_transaction_features:transaction_count_30d",
        "customer_transaction_features:total_spend_30d",
        "customer_profile_features:tenure_months",
    ],
    entity_rows=[
        {"customer_id": "C001"},
        {"customer_id": "C002"},
    ],
).to_dict()
```

### 20.9.4 Alternatives: Tecton and Hopsworks

**Tecton** is a managed feature platform built by the original Feast creators. It adds real-time feature computation (streaming and on-demand features), a feature monitoring system, and enterprise governance. Tecton is the right choice for organizations that need streaming features and are willing to pay for a managed service.

**Hopsworks** is an open-source feature store with a focus on Python-first experience. It provides a feature engineering framework, a model registry, and support for both batch and real-time features. Its unique differentiator is strong support for time-series features and built-in data validation.

---

## 20.10 Data Quality

Data quality is the foundation on which ML systems are built. A model trained on incorrect, incomplete, or inconsistent data will produce incorrect, unreliable predictions---no matter how sophisticated the algorithm. The adage "garbage in, garbage out" is as true in deep learning as it was in classical statistics.

### 20.10.1 Great Expectations

Great Expectations is an open-source framework for data validation. It provides a declarative interface for defining **expectations** (assertions about data), generating **data docs** (human-readable documentation of data quality), and running **checkpoints** (automated validation suites).

```python
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration

# Initialize context
context = gx.get_context()

# Create a data source
datasource = context.sources.add_pandas("pandas_datasource")
data_asset = datasource.add_dataframe_asset(name="customer_features")

# Build a batch request
batch_request = data_asset.build_batch_request(dataframe=customer_df)

# Create an expectation suite
suite = context.add_expectation_suite("customer_features_suite")

# Add expectations
suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_table_row_count_to_be_between",
        kwargs={"min_value": 10000, "max_value": 10000000},
    )
)

suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "customer_id"},
    )
)

suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_unique",
        kwargs={"column": "customer_id"},
    )
)

suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "avg_transaction_amount",
            "min_value": 0,
            "max_value": 100000,
            "mostly": 0.99,  # Allow 1% outliers
        },
    )
)

suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_in_set",
        kwargs={
            "column": "contract_type",
            "value_set": ["monthly", "annual", "two_year"],
        },
    )
)

suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_pair_values_a_to_be_greater_than_b",
        kwargs={
            "column_A": "total_spend",
            "column_B": "avg_transaction_amount",
            "or_equal": True,
        },
    )
)

# Run validation
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="customer_features_suite",
)
results = validator.validate()

if not results.success:
    failed = [
        r.expectation_config.expectation_type
        for r in results.results
        if not r.success
    ]
    raise ValueError(f"Data quality check failed: {failed}")

# Build data docs (HTML reports)
context.build_data_docs()
```

### 20.10.2 Data Contracts

Data contracts formalize the expectations between data producers and consumers. They define the schema, semantics, quality requirements, and SLAs for a dataset. Without data contracts, upstream schema changes can silently break downstream ML models.

**Schema Registries** (Confluent Schema Registry, AWS Glue Schema Registry) enforce schema evolution rules:

- **Backward Compatibility:** New schemas can read data written by old schemas. New consumers can process old messages.
- **Forward Compatibility:** Old schemas can read data written by new schemas. Old consumers can process new messages.
- **Full Compatibility:** Both backward and forward compatible.

```python
# Example: Enforcing a data contract with Pydantic
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class CustomerFeatureContract(BaseModel):
    """Data contract for customer features.

    Version: 2.0
    Owner: data-engineering@company.com
    SLA: Updated daily by 6 AM UTC
    """

    customer_id: str = Field(..., description="Unique customer identifier")
    transaction_count_30d: int = Field(..., ge=0, description="Transactions in last 30 days")
    total_spend_30d: float = Field(..., ge=0, description="Total spend in last 30 days (USD)")
    avg_transaction_amount: float = Field(..., ge=0, le=100000)
    tenure_months: int = Field(..., ge=0, le=600)
    contract_type: str = Field(..., description="Customer contract type")
    feature_timestamp: datetime = Field(..., description="When features were computed")

    @validator("contract_type")
    def validate_contract_type(cls, v):
        allowed = {"monthly", "annual", "two_year"}
        if v not in allowed:
            raise ValueError(f"Invalid contract_type: {v}. Must be one of {allowed}")
        return v

    @validator("feature_timestamp")
    def validate_freshness(cls, v):
        from datetime import timezone
        age_hours = (datetime.now(timezone.utc) - v).total_seconds() / 3600
        if age_hours > 36:
            raise ValueError(f"Feature data is {age_hours:.1f} hours old (max 36)")
        return v

# Validate incoming data against the contract
def validate_batch(df):
    errors = []
    for _, row in df.iterrows():
        try:
            CustomerFeatureContract(**row.to_dict())
        except Exception as e:
            errors.append({"row": row["customer_id"], "error": str(e)})
    if errors:
        raise ValueError(f"Contract violations: {len(errors)} rows failed validation")
```

---

## 20.11 Snowflake

Snowflake is a cloud-native data warehouse that separates storage, compute, and cloud services into independent layers, enabling each to scale independently.

### 20.11.1 Architecture

**Storage Layer.** Data is stored in a compressed, columnar format on the cloud provider's object storage (S3, GCS, Azure Blob). Snowflake manages the storage transparently---users interact with tables, not files.

**Compute Layer (Virtual Warehouses).** Query execution happens in virtual warehouses---clusters of compute resources that can be started, stopped, and resized independently. Multiple warehouses can query the same data simultaneously without contention. This is Snowflake's key innovation: you can have a small warehouse for interactive queries, a large warehouse for ETL, and an extra-large warehouse for ML training, all sharing the same data.

**Cloud Services Layer.** Metadata management, query optimization, access control, and transaction management happen in this always-on layer.

### 20.11.2 Snowpark for ML

Snowpark enables running Python, Java, or Scala code directly in Snowflake's compute layer:

```python
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, avg, count, stddev
from snowflake.ml.modeling.ensemble import GradientBoostingClassifier
from snowflake.ml.modeling.preprocessing import StandardScaler, OneHotEncoder

session = Session.builder.configs(connection_params).create()

# Read data using Snowpark DataFrames
customers = session.table("ML_PLATFORM.FEATURES.CUSTOMER_FEATURES")

# Feature engineering in Snowflake's compute
features = (
    customers
    .filter(col("TENURE_MONTHS") > 0)
    .with_column("SPEND_PER_MONTH", col("TOTAL_SPEND") / col("TENURE_MONTHS"))
    .with_column("TICKET_RATE", col("SUPPORT_TICKETS") / col("TENURE_MONTHS"))
)

# Train a model using Snowflake ML
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler(input_cols=numerical_cols, output_cols=numerical_cols)),
        ("encoder", OneHotEncoder(input_cols=categorical_cols, output_cols=categorical_cols)),
        ("model", GradientBoostingClassifier(
            input_cols=feature_cols,
            label_cols=["CHURNED"],
            n_estimators=200,
            max_depth=5,
        )),
    ]
)

train_df, test_df = features.random_split([0.8, 0.2], seed=42)
pipeline.fit(train_df)
predictions = pipeline.predict(test_df)
```

### 20.11.3 Time Travel and Data Sharing

Snowflake's time travel allows querying data as it existed at any point within the retention period (up to 90 days on Enterprise edition):

```sql
-- Query data as of a specific timestamp
SELECT * FROM customer_features AT(TIMESTAMP => '2025-06-15 10:00:00'::timestamp);

-- Query data before a specific statement
SELECT * FROM customer_features BEFORE(STATEMENT => '8e5d1c-a342-4f5b');

-- Restore a dropped table
UNDROP TABLE customer_features;
```

**Data Sharing** allows Snowflake accounts to share live data with other accounts without copying. This is valuable for ML teams that need access to data owned by other organizations (e.g., a data provider sharing labeled datasets with a model development team).

---

## 20.12 Federated Queries with Trino

Trino (formerly PrestoSQL) is a distributed SQL query engine designed for federated queries---executing a single SQL query across multiple, heterogeneous data sources.

### 20.12.1 Connector Architecture

Trino connects to data sources through **connectors**. Each connector translates Trino's query plan into operations on the underlying data source. Available connectors include:

- **Hive/Iceberg/Delta Lake** connectors for data lake queries
- **MySQL, PostgreSQL, SQL Server** connectors for relational databases
- **Elasticsearch** connector for search indices
- **Kafka** connector for streaming data
- **MongoDB** connector for document stores
- **Google Sheets** connector (yes, really)

```sql
-- Query across multiple data sources in a single statement
SELECT
    c.customer_id,
    c.name,
    t.total_spend,
    p.last_prediction,
    p.prediction_timestamp
FROM postgresql.crm.customers c
JOIN delta.lake.customer_spending t
    ON c.customer_id = t.customer_id
JOIN elasticsearch.ml.predictions p
    ON c.customer_id = p.customer_id
WHERE t.total_spend > 10000
  AND p.prediction_timestamp > TIMESTAMP '2025-06-01'
ORDER BY t.total_spend DESC
LIMIT 100;
```

### 20.12.2 Query Optimization

Trino's optimizer pushes predicates and projections down to source connectors when possible, minimizing data transfer. For joins between a large and small table, Trino can broadcast the small table (similar to Spark's broadcast join). The query engine supports cost-based optimization when table statistics are available.

Trino is particularly valuable for ML teams that need to join training data from multiple systems---a customer table in PostgreSQL, transaction features in a Delta Lake table, and labels in a third system---without first ETL-ing everything into a single warehouse.

---

## 20.13 Apache Beam

Apache Beam provides a unified programming model for both batch and streaming data processing. The key insight is that the same pipeline definition can run on different execution engines (**runners**): Google Cloud Dataflow, Apache Flink, Apache Spark, or a local DirectRunner for development.

### 20.13.1 Core Abstractions

**PCollections** are distributed datasets (analogous to Spark RDDs or Flink DataStreams).

**PTransforms** are operations that take one or more PCollections as input and produce one or more PCollections as output.

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

class ComputeFeatures(beam.DoFn):
    """Compute ML features from raw events."""

    def process(self, element):
        import json
        event = json.loads(element)

        features = {
            "user_id": event["user_id"],
            "event_type": event["event_type"],
            "amount": float(event.get("amount", 0)),
            "hour_of_day": int(event["timestamp"].split("T")[1][:2]),
            "day_of_week": self._get_day_of_week(event["timestamp"]),
            "is_weekend": self._get_day_of_week(event["timestamp"]) >= 5,
        }
        yield features

    def _get_day_of_week(self, timestamp_str):
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp_str)
        return dt.weekday()


class AggregateFeatures(beam.CombineFn):
    """Aggregate features per user."""

    def create_accumulator(self):
        return {"count": 0, "total_amount": 0.0, "amounts": []}

    def add_input(self, accumulator, element):
        accumulator["count"] += 1
        accumulator["total_amount"] += element["amount"]
        accumulator["amounts"].append(element["amount"])
        return accumulator

    def merge_accumulators(self, accumulators):
        merged = {"count": 0, "total_amount": 0.0, "amounts": []}
        for acc in accumulators:
            merged["count"] += acc["count"]
            merged["total_amount"] += acc["total_amount"]
            merged["amounts"].extend(acc["amounts"])
        return merged

    def extract_output(self, accumulator):
        import statistics
        amounts = accumulator["amounts"]
        return {
            "event_count": accumulator["count"],
            "total_amount": accumulator["total_amount"],
            "avg_amount": statistics.mean(amounts) if amounts else 0,
            "std_amount": statistics.stdev(amounts) if len(amounts) > 1 else 0,
        }


# Pipeline definition — works for both batch and streaming
options = PipelineOptions(
    runner="DataflowRunner",
    project="my-gcp-project",
    region="us-central1",
    temp_location="gs://my-bucket/temp/",
    streaming=True,  # Set to False for batch
)

with beam.Pipeline(options=options) as pipeline:
    (
        pipeline
        | "Read Events" >> beam.io.ReadFromPubSub(topic="projects/my-project/topics/events")
        | "Parse and Compute Features" >> beam.ParDo(ComputeFeatures())
        | "Key by User" >> beam.Map(lambda x: (x["user_id"], x))
        | "Window" >> beam.WindowInto(beam.window.FixedWindows(300))  # 5-minute windows
        | "Aggregate" >> beam.CombinePerKey(AggregateFeatures())
        | "Format Output" >> beam.Map(lambda kv: json.dumps({"user_id": kv[0], **kv[1]}))
        | "Write to BigQuery" >> beam.io.WriteToBigQuery(
            table="my_project:features.user_features",
            schema="user_id:STRING,event_count:INTEGER,total_amount:FLOAT,avg_amount:FLOAT,std_amount:FLOAT",
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        )
    )
```

### 20.13.2 Runners

The same Beam pipeline can run on:

- **DirectRunner:** Local execution for development and testing.
- **DataflowRunner:** Google Cloud Dataflow (managed, auto-scaling).
- **FlinkRunner:** Apache Flink cluster.
- **SparkRunner:** Apache Spark cluster.

This portability is Beam's primary value proposition: you can develop locally, test on Flink, and deploy on Dataflow without changing the pipeline code. In practice, however, the level of feature parity across runners varies, and performance characteristics differ significantly. Most teams that use Beam in production deploy on Dataflow or Flink.

---

## Exercises

1. **Spark Feature Pipeline.** Write a PySpark job that reads transaction data, computes customer-level features (RFM: Recency, Frequency, Monetary), and writes the result as a Delta Lake table. Include proper partitioning and Z-ordering.

2. **Kafka Streaming Features.** Build a Kafka producer that simulates user events, and a consumer that computes rolling 5-minute feature aggregations. Use the confluent-kafka Python client.

3. **dbt Feature Models.** Create a dbt project with staging models, a feature engineering model (incremental), and tests for all columns. Run it against a local DuckDB or PostgreSQL database.

4. **Feature Store.** Set up a Feast feature store with at least two feature views. Generate training data using point-in-time joins and serve features online using the Feast feature server.

5. **Data Quality.** Using Great Expectations, create an expectation suite for a dataset of your choice. Define at least 10 expectations, run validation, and generate data docs.

6. **Medallion Architecture.** Implement a complete bronze/silver/gold pipeline using Delta Lake. Include data quality checks at the silver layer and feature aggregations at the gold layer.

7. **Stream Processing Comparison.** Implement the same feature computation pipeline in both Flink (using PyFlink) and Spark Structured Streaming. Compare latency, throughput, and development experience.

8. **Federated Query.** Set up Trino with connectors to at least two different data sources (e.g., PostgreSQL and a Parquet file store). Write a query that joins data across both sources.

---

## References

Apache Beam Documentation. (2024). *Apache Beam Programming Guide*. https://beam.apache.org/documentation/programming-guide/

Apache Flink Documentation. (2024). *Apache Flink*. https://flink.apache.org/docs/

Apache Iceberg Documentation. (2024). *Apache Iceberg*. https://iceberg.apache.org/docs/

Armbrust, M., Das, T., Sun, L., Yavuz, B., Zhu, S., Murthy, M., Torres, J., van Hovell, H., Ionescu, A., Luszczak, A., Switakowski, M., Szafranski, M., Li, X., Ueshin, T., Mokhtar, M., Boncz, P., Ghodsi, A., Xin, R., & Zaharia, M. (2020). Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores. *Proceedings of the VLDB Endowment*, 13(12), 3411--3424.

Armbrust, M., Ghodsi, A., Xin, R., & Zaharia, M. (2021). Lakehouse: A New Generation of Open Platforms that Unify Data Warehousing and Advanced Analytics. In *Proceedings of the 11th Conference on Innovative Data Systems Research (CIDR '21)*.

dbt Documentation. (2024). *dbt: Data Build Tool*. https://docs.getdbt.com/

Feast Documentation. (2024). *Feast: Feature Store for Machine Learning*. https://docs.feast.dev/

Great Expectations Documentation. (2024). *Great Expectations*. https://docs.greatexpectations.io/

Kreps, J., Narkhede, N., & Rao, J. (2011). Kafka: A Distributed Messaging System for Log Processing. In *Proceedings of the NetDB Workshop*.

Snowflake Documentation. (2024). *Snowflake Documentation*. https://docs.snowflake.com/

Trino Documentation. (2024). *Trino: Distributed SQL Query Engine*. https://trino.io/docs/current/

Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., Meng, X., Rosen, J., Venkataraman, S., Franklin, M. J., Ghodsi, A., Gonzalez, J., Shenker, S., & Stoica, I. (2016). Apache Spark: A Unified Engine for Big Data Processing. *Communications of the ACM*, 59(11), 56--65.
