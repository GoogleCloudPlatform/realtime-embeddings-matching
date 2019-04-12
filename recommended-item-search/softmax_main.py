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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf
import input_pipeline
import softmax_model
# pylint: enable=g-bad-import-order

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='metadata_path', default='metadata.pickle',
    help='Set a path to metadata created by preprocess_movielens.py')
flags.DEFINE_list(
    name='hidden_dims', default=['64','32'],
    help='The sizes of hidden layers for MLP. e.g. --layers=32,16,8,4')
flags.DEFINE_enum(
    name='activation', default='relu',
    enum_values=['relu', 'None'], case_sensitive=False,
    help='Specify an activation function used in hidden layers.')
flags.DEFINE_string(
    name='model_dir', default='./model',
    help='Set a model directory where model and checkpoint files are stored.')
flags.DEFINE_string(
    name='export_dir', default='Servo',
    help='Set a sub directory where savedmodels are saved.')
flags.DEFINE_boolean(
    name='resume_training', default=False,
    help='Resume training from a latest checkpoint.')
flags.DEFINE_string(
    name='train_filename', default='train*.tfrecord',
    help='Set a file pattern of training inputs.')
flags.DEFINE_integer(
    name='train_batch_size', default=200,
    help='Set a batch size for training process.')
flags.DEFINE_integer(
    name='train_max_steps', default=1000000,
    help='Set a max training step per execution.')
flags.DEFINE_string(
    name='eval_filename', default='eval*.tfrecord',
    help='Set a file pattern of evaluation inputs.')
flags.DEFINE_integer(
    name='eval_batch_size', default=10000,
    help='Set a batch size for evaluation process.')
flags.DEFINE_integer(
    name='eval_steps', default=10,
    help='Set the number of steps per evaluation.')
flags.DEFINE_integer(
    name='eval_throttle_secs', default=10,
    help='Set throttle secs for each evaluation.')
flags.DEFINE_float(
    name='learning_rate', default=0.01,
    help='Set a learning rate for optimizer.')
flags.DEFINE_integer(
    name='lr_decay_steps', default=100000,
    help='Set a learning rate decay steps.')
flags.DEFINE_float(
    name='lr_decay_rate', default=0.96,
    help='Set a learning rate decay rate.')
flags.DEFINE_integer(
    name='save_checkpoints_steps', default=10000,
    help='Set frequency of saving checkpoints.')
flags.DEFINE_integer(
    name='keep_checkpoint_max', default=3,
    help='Set maximum number of saved checkpoints.')
flags.DEFINE_integer(
    name='log_step_count_steps', default=1000,
    help='Set frequency of loss logging.')
flags.DEFINE_integer(
    name='tf_random_seed', default=20190501,
    help='Set random seed for TensorFlow.')

tf.logging.set_verbosity(tf.logging.INFO)


def get_run_config():
  """Get running parameters for Estimator."""
  return tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      tf_random_seed=FLAGS.tf_random_seed,
      log_step_count_steps=FLAGS.log_step_count_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      train_distribute=None,
      session_config=tf.ConfigProto(allow_soft_placement=True)
  )

def get_hyperparams():
  """Get hyper params which are used in model function."""
  return tf.contrib.training.HParams(
      metadata_path=FLAGS.metadata_path,
      hidden_dims=[int(dim) for dim in FLAGS.hidden_dims],
      activation_name=FLAGS.activation,
      learning_rate=FLAGS.learning_rate,
      lr_decay_steps=FLAGS.lr_decay_steps,
      lr_decay_rate=FLAGS.lr_decay_rate,
  )

def get_train_spec():
  """Get train spec for Estimator."""
  profile_hook = tf.train.ProfilerHook(
      save_steps=FLAGS.save_checkpoints_steps, output_dir=FLAGS.model_dir,
      show_memory=True)
  train_input_fn = input_pipeline.generate_input_fn(
      file_pattern=FLAGS.train_filename, batch_size=FLAGS.train_batch_size,
      mode=tf.estimator.ModeKeys.TRAIN)
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=FLAGS.train_max_steps,
      hooks=[profile_hook])
  return train_spec
  
def get_eval_spec():
  """Get eval spec for Estimator."""
  exporter = tf.estimator.LatestExporter(
      name=FLAGS.export_dir, exports_to_keep=FLAGS.keep_checkpoint_max,
      serving_input_receiver_fn=softmax_model.serving_input_fn)
  eval_input_fn = input_pipeline.generate_input_fn(
      file_pattern=FLAGS.eval_filename, batch_size=FLAGS.eval_batch_size,
      mode=tf.estimator.ModeKeys.EVAL)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn, steps=FLAGS.eval_steps,
      throttle_secs=FLAGS.eval_throttle_secs, exporters=exporter)
  return eval_spec

def remove_artifacts():
  """Remove previous artifacts if needed."""
  if not FLAGS.resume_training:
    if tf.gfile.Exists(FLAGS.model_dir):
      tf.logging.info('Removing {} ...'.format(FLAGS.model_dir))
      tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.summary.FileWriterCache.clear()
  
def main(_):
  remove_artifacts()
  estimator = tf.estimator.Estimator(
      model_fn=softmax_model.model_fn,
      params=get_hyperparams(),
      config=get_run_config())
  tf.estimator.train_and_evaluate(
      estimator=estimator,
      train_spec=get_train_spec(),
      eval_spec=get_eval_spec())
    
if __name__ == '__main__':
  absl_app.run(main)
