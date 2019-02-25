# Text semantic search

This example shows how to build an end-to-end real-time text semantic search:
1. Extract text embeddings of Wikipedia titles from BigQuery, 
using the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2) module,
via Cloud Dataflow. 
2. Build an approximate similarity matching index using Spotify’s [Annoy](https://github.com/spotify/annoy)
library, via Cloud ML Engine.
3. Serve the index for real-time search queries in a [Flask](http://flask.pocoo.org/)
web application, via Google AppEngine.

To run the end-to-end example, you need to execute the following script files,
after setting the configuration parameters:
1. embeddings_extraction/run.sh
2. index_builder/submit.sh
3. semantic_search/deploy.sh

## Requirements

You need to have your [GCP Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
You can use [Cloud Shell](https://cloud.google.com/shell/docs/quickstart) 
or [gcloud CLI](https://cloud.google.com/sdk/) to run all the commands in this
guideline.

## Setup a project

Follow the [instruction](https://cloud.google.com/resource-manager/docs/creating-managing-projects) and create a GCP project. 
Once created, enable the Dataflow API, BigQuery API in this [page](https://console.developers.google.com/apis/enabled).
You can also find more details about enabling the [billing](https://cloud.google.com/billing/docs/how-to/modify-project?#enable-billing).

We recommend to use CloudShell from the GCP console to run the below commands.
CloudShell starts with an environment already logged in to your account and set
to the currently selected project. The following commands are required only in a
workstation shell environment, they are not needed in the CloudShell. 

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project [your-project-id]
gcloud config set compute/zone us-central1-a
```

## Setup python environment and sample code

Follow commands below to install required python packages and download a
Dataflow pipeline code.

```bash
git clone [this-repo]
cd [this-repo]/text-semantic-search

# Make sure you have python 2.7 environement
pip install -r requirements.txt
```

## 1. Run the Dataflow pipeline for embedding extraction

First, Set the following configurations for your Dataflow job in
embeddings_extraction/run.sh script file:

```bash
PROJECT=[your-project-ID]
BUCKET=[your-bucket-name]
REGION=[your-preferred-region]

# BigQuery parameters
LIMIT=1000000

# Datastore parameters
KIND="wikipedia"

# Directory for output data files
OUTPUT_PREFIX="${BUCKET}/${KIND}/embeddings/embed"

# Working directories for Dataflow
DF_JOB_DIR="${BUCKET}/${KIND}/dataflow"
STAGING_LOCATION="${DF_JOB_DIR}/staging"
TEMP_LOCATION="${DF_JOB_DIR}/temp"

# Working directories for tf.transform
TRANSFORM_ROOT_DIR="${DF_JOB_DIR}/transform"
TRANSFORM_TEMP_DIR="${TRANSFORM_ROOT_DIR}/temp"
TRANSFORM_EXPORT_DIR="${TRANSFORM_ROOT_DIR}/export"

# Working directories for Debug log
DEBUG_OUTPUT_PREFIX="${DF_JOB_DIR}/debug/log"

# Running Config for Dataflow
RUNNER=DataflowRunner
JOB_NAME=job-"${KIND}-embeddings-extraction-"$(date +%Y%m%d%H%M%S)
MACHINE_TYPE=n1-highmem-2
```

Then, you can run the pipeline by executing the **run.sh** script file:

```bash
bash embeddings_extraction/run.sh
```

## 2. Submit the Cloud ML Engine job to build the index

First, Set the following configurations for your Cloud ML Engine job in
index_builder/submit.sh script file:

```bash
# Configurable parameters
PROJECT=[your-project-ID]     
BUCKET=[your-bucket-name]     
REGION=[your-preferred-region]

# Datastore parameters
KIND="wikipedia"

# Cloud ML Engine parameters
TIER="CUSTOM"

# Annoy parameters
NUM_TREES=10

# File locations parameters
EMBED_FILES=gs://"${BUCKET}/${KIND}/embeddings/embed-*"
INDEX_FILE=gs://"${BUCKET}/${KIND}/index/embeds.index"

# Cloud ML Engine parameters
PACKAGE_PATH=builder
JOB_DIR=gs://"${BUCKET}/${KIND}/index/jobs"
CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME="${KIND}_build_annoy_index_${CURRENT_DATE}"
```

Then, you can submit the job by executing the **submit.sh** script file:

```bash
bash index_builder/submit.sh
```

## 3. Deploy an AppEngine for semantic search app

First, set the following configurations for your search service in the 
semantic_search/utis/search.py and the semantic_search/deploy.sh files

```code
GCS_BUCKET = [your-bucket-name]
KIND = 'wikipedia'
```

Second, set your GCP project ID in the **deploy.sh** script file: 

```bash
PROJECT=[your-project-ID]
```

Then, you can deploy the app to AppEngine by executing the **deploy.sh**
script file:

```bash
bash semantic_search/deploy.sh
```