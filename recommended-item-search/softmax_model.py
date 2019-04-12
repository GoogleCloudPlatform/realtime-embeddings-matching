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

import os
import pickle
import tensorflow as tf
# tf.enable_eager_execution()


def get_feature_columns(metadata_path, embeddings_dim):
  def _get_num_bucket():
    with tf.io.gfile.GFile(metadata_path, 'rb') as f:
      metadata = pickle.load(f)
    return metadata['N']
    
  categorical_col = tf.feature_column.categorical_column_with_identity(
      key='movie_ids', num_buckets=_get_num_bucket())
  feature_columns = [
    # movie_ids
    tf.feature_column.embedding_column(
      categorical_column=categorical_col, dimension=embeddings_dim,
      combiner='mean')
  ]
  return feature_columns

def get_activation_fn(activation):
  if activation == 'relu':
    return tf.nn.relu
  else:
    return None

def build_network(
    inputs, hidden_dims, activation_fn, scope='user_embeddings'):
  """Build a forward network graph."""
  with tf.variable_scope(scope):
    input_dim = inputs.shape[1].value
    for i, output_dim in enumerate(hidden_dims):
      layer = tf.layers.Dense(output_dim, activation=activation_fn)
      outputs = layer(inputs)
      input_dim = output_dim
      inputs = outputs
    return outputs

def generate_labels(features):
  def _select_random(x):
    """Selectes a random elements from each row of x."""
    def to_float(x):
      return tf.cast(x, tf.float32)
    def to_int(x):
      return tf.cast(x, tf.int64)

    batch_size = tf.shape(x)[0]
    rn = tf.range(batch_size)
    nnz = to_float(tf.count_nonzero(x >= 0, axis=1))
    rnd = tf.random_uniform([batch_size])
    ids = tf.stack([to_int(rn), to_int(nnz * rnd)], axis=1)
    return to_int(tf.gather_nd(x, ids))
  return _select_random(features['movie_ids'])

def softmax_loss(user_embeddings, movie_embeddings, labels):
  """Calculate loss with sampled movie id."""
  user_embedding_size = user_embeddings.shape[1].value
  movie_embedding_size = movie_embeddings.shape[1].value
  if user_embedding_size != movie_embedding_size:
    raise ValueError(
        "The user embedding dimension %d should match the movie embedding "
        "dimension % d" % (user_embedding_size, movie_embedding_size))

  logits = tf.matmul(user_embeddings, movie_embeddings, transpose_b=True)
  loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
  return loss

def serving_input_fn():
  receiver_tensor = {'input': tf.placeholder(shape=[None, None], dtype=tf.int64)}
  features = {'movie_ids': receiver_tensor['input']}
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

def model_fn(features, labels, mode, params):
  """A recommendation model for movielens dataset.
  
  Args:
    features: sequences of movie_ids (batch_size x sequence_size)
    labels: None
  """
  # fill a difference between training and prediction input.
  if not isinstance(features, dict):
    features = {'movie_ids': features}

  # create user_embeddings
  feature_columns = get_feature_columns(
      metadata_path=params.metadata_path,
      embeddings_dim=params.hidden_dims[-1])
  user_input = tf.feature_column.input_layer(
      features=features, feature_columns=feature_columns)
  user_embeddings = build_network(
      inputs=user_input, hidden_dims=params.hidden_dims,
      activation_fn=get_activation_fn(params.activation_name))
  
  # extract movie_embeddings
  with tf.variable_scope('input_layer', reuse=True):
    movie_embeddings = tf.get_variable('movie_ids_embedding/embedding_weights')

  # generate labels from features['movie_ids']
  labels = generate_labels(features)
  loss = softmax_loss(user_embeddings, movie_embeddings, labels)
  
  estimator_spec = None
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'user_embeddings': user_embeddings
    }
    export_outputs = {
        'predictions': tf.estimator.export.PredictOutput(predictions)
    }
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions, export_outputs=export_outputs)
    
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(
        learning_rate=params.learning_rate, global_step=global_step,
        decay_steps=params.lr_decay_steps, decay_rate=params.lr_decay_rate)
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
    predictions = tf.matmul(
        user_embeddings, movie_embeddings, transpose_b=True)
    eval_metric_ops = {
        'precision_at_10': tf.metrics.precision_at_k(
            labels=labels, predictions=predictions, k=10)
    }
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
  return estimator_spec