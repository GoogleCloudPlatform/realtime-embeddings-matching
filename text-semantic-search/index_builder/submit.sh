#!/usr/bin/env bash
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

# Datastore parameters
KIND="wikipedia"

# Cloud ML Engine parameters
TIER="CUSTOM"

# Annoy parameters
NUM_TREES=100

[[ -z "${PROJECT}" ]] && echo "PROJECT not set" && exit 1
[[ -z "${BUCKET}" ]] && echo "BUCKET not set" && exit 1
[[ -z "${REGION}" ]] && echo "REGION not set" && exit 1
[[ -z "${KIND}" ]] && echo "KIND not set" && exit 1
[[ -z "${TIER}" ]] && echo "TIER not set" && exit 1
[[ -z "${NUM_TREES}" ]] && echo "NUM_TREES not set" && exit 1

# File locations parameters
EMBED_FILES=gs://"${BUCKET}/${KIND}/embeddings/embed-*"
INDEX_FILE=gs://"${BUCKET}/${KIND}/index/embeds.index"

# Cloud ML Engine parameters
PACKAGE_PATH=builder
JOB_DIR=gs://"${BUCKET}/${KIND}/index/jobs"
CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME="${KIND}_build_annoy_index_${CURRENT_DATE}"

echo "Submitting a Cloud ML Engine job..."

# Command to submit the Cloud ML Engine job
gcloud ml-engine jobs submit training "${JOB_NAME}" \
        --job-dir="${JOB_DIR}" \
        --runtime-version=1.12 \
        --region="${REGION}" \
        --scale-tier="${TIER}" \
        --module-name=builder.task \
        --package-path="${PACKAGE_PATH}"  \
        --config=config.yaml \
        -- \
        --embedding-files="${EMBED_FILES}" \
        --index-file="${INDEX_FILE}" \
        --num-trees="${NUM_TREES}"

echo -e "Cloud ML Engine job submitted successfully!"