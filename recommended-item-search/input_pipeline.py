#!/usr/bin/python
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

import tensorflow as tf
# tf.enable_eager_execution()


def parse_fn(serialized_example):
  """Parse a serialized example."""
  
  # user_id is not currently used.
  context_features = {
    'user_id': tf.FixedLenFeature([], dtype=tf.int64)
  }
  sequence_features = {
    'movie_ids': tf.FixedLenSequenceFeature([], dtype=tf.int64)
  }
  parsed_feature, parsed_sequence_feature = tf.parse_single_sequence_example(
    serialized=serialized_example,
    context_features=context_features,
    sequence_features=sequence_features
  )
  movie_ids = parsed_sequence_feature['movie_ids']
  return movie_ids

def generate_input_fn(file_pattern, batch_size, mode=tf.estimator.ModeKeys.EVAL):
  """Generate input function for Estimator. 
  
  Args:
    file_pattern: pattern of input file names. 
    batch_size: batch size used in input function.
  Returns:
    input function which returns sequences of movie_ids.
  """
  def _input_fn():
    #ToDo(yaboo): num_cpu should be parameterized.
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=8)
    
    #ToDo(yaboo): buffer_size should be parameterized.
    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2 * batch_size)
    
    # Note that movie_id sequences are padded with -1.
    dataset = dataset.padded_batch(
      batch_size=batch_size, padded_shapes=(tf.TensorShape([None])),
      padding_values=(tf.constant(-1, dtype=tf.int64)))
    return dataset
  return _input_fn