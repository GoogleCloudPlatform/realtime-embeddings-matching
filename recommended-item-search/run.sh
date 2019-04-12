#!/bin/bash
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
MODEL_DIR="gs://yaboo-sandbox-recommendation/softmax"
DATA_DIR="./data"

# Download MovieLens dataset and save it in tfrecord format.
python3 data_preparation.py \
  --export_dir=${DATA_DIR} \
  --filename='ml-20m.zip' \
  --rating_threshold='4.0'

# Train Recommendation model
python3 softmax_main.py \
  --model_dir=${MODEL_DIR} \
  --train_max_steps=100000000 \
  --train_batch_size=200 \
  --eval_batch_size=1000 \
  --eval_steps=10 \
  --train_filename="${DATA_DIR}/train*.tfrecord" \
  --eval_filename="${DATA_DIR}/eval*.tfrecord" \
  --log_step_count_steps=1000 \
  --save_checkpoints_steps=100000 \
  --metadata_path="${DATA_DIR}/metadata.pickle" \
  --hidden_dims=35 \
  --activation='None'
