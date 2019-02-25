#!/bin/bash -eu
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configurable parameters
PROJECT=""
BUCKET=""
REGION=""

# BigQuery parameters
LIMIT=1000000

# Datastore parameters
KIND="wikipedia"

[[ -z "${PROJECT}" ]] && echo "PROJECT not set" && exit 1
[[ -z "${BUCKET}" ]] && echo "BUCKET not set" && exit 1
[[ -z "${REGION}" ]] && echo "REGION not set" && exit 1
[[ -z "${LIMIT}" ]] && echo "LIMIT not set" && exit 1
[[ -z "${KIND}" ]] && echo "KIND not set" && exit 1

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

# Cleaning working and oputput directories before running the Dataflow job
echo "Cleaning working and output directories..."

gsutil -m rm -r "${DF_JOB_DIR}"
gsutil -m rm -r "${OUTPUT_PREFIX}"

echo "Running the Dataflow job..."

# Command to run the Dataflow job
python run.py \
  --output_dir="${OUTPUT_PREFIX}" \
  --transform_temp_dir="${TRANSFORM_TEMP_DIR}" \
  --transform_export_dir="${TRANSFORM_EXPORT_DIR}" \
  --project="${PROJECT}" \
  --runner="${RUNNER}" \
  --region="${REGION}" \
  --kind="${KIND}" \
  --limit="${LIMIT}" \
  --staging_location="${STAGING_LOCATION}" \
  --temp_location="${TEMP_LOCATION}" \
  --setup_file=$(pwd)/setup.py \
  --job_name="${JOB_NAME}" \
  --worker_machine_type="${MACHINE_TYPE}" \
#  --enable_debug \
#  --debug_output_prefix="${DEBUG_OUTPUT_PREFIX}"

echo "Dataflow job submitted successfully!"