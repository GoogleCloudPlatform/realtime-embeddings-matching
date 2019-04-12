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

from absl import app as absl_app
from absl import flags
import numpy as np
import os
import pandas
import pickle
import tempfile
import tensorflow as tf
import urllib
import zipfile


FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='base_url',
    default='http://files.grouplens.org/datasets/movielens/',
    help='Specify a base url of movielens dataset.')
flags.DEFINE_enum(
    name='filename', default='ml-latest-small.zip',
    enum_values=['ml-latest.zip', 'ml-latest-small.zip', 'ml-20m.zip'],
    help='Specify movielens dataset to download from MovieLens site.')
flags.DEFINE_string(
    name='export_dir', default='.',
    help='Specify directory where preprocessed data is saved.')
flags.DEFINE_float(
    name='rating_threshold', default=4.0,
    help='Ignore movies which rating is lower than the threshold.')

tf.logging.set_verbosity(tf.logging.INFO)

def load_movielens_data():
  # Download MovieLens dataset
  urllib.request.urlretrieve(
      url=os.path.join(FLAGS.base_url, FLAGS.filename),
      filename=FLAGS.filename)
  
  # Extract MovieLens dataset from zipfile
  zipfile.ZipFile(FLAGS.filename, 'r').extractall()

  # Load MovieLens dataset
  tmp_dir = FLAGS.filename.split('.')[0]
  movies = pandas.read_csv(
      filepath_or_buffer=os.path.join(tmp_dir, 'movies.csv'),
      names=['movie_id', 'title', 'genres'], header=0)
  ratings = pandas.read_csv(
      filepath_or_buffer=os.path.join(tmp_dir, 'ratings.csv'),
      names=['user_id', 'movie_id', 'rating', 'unix_timestamp'],
      header=0)
  
  # Remove unnecessary files.
  tf.gfile.DeleteRecursively(tmp_dir)
  tf.gfile.Remove(FLAGS.filename)
  return movies, ratings

def split_dataframe(dataframe, holdout_fraction=0.1):
  """Splits a DataFrame into training and test sets.
  Args:
    dataframe: a dataframe.
    holdout_fraction: fraction of dataframe rows to use in the test set.
  Returns:
    train: dataframe for training
    test: dataframe for testing
  """
  test = dataframe.sample(frac=holdout_fraction, replace=False)
  train = dataframe[~dataframe.index.isin(test.index)]
  return train, test

def make_sequence_example(user_id, movie_ids):
  """Returns a SequenceExample for the given user_id and movie_ids.
  Args:
    user_id: An user_id.
    movie_ids: A list of strings.
  Returns:
    A tf.train.SequenceExample containing user_id and movie_ids.
  """
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  feature = {'user_id': _int64_feature(user_id)}
  movie_id_features = [
      tf.train.Feature(int64_list=tf.train.Int64List(value=[id_]))
      for id_ in movie_ids]
  feature_list = {'movie_ids': tf.train.FeatureList(feature=movie_id_features)}
  features = tf.train.Features(feature=feature)
  feature_lists = tf.train.FeatureLists(feature_list=feature_list)
  return tf.train.SequenceExample(context=features, feature_lists=feature_lists)

def split_range(num_records, num_buckets):
  interval = num_records // num_buckets
  head_ids = [interval*i for i in range(num_buckets)]
  tail_ids = head_ids[1:] + [num_records]
  return list(zip(head_ids, tail_ids))

def make_tfrecord_files(dataframe, file_type, num_files):
  """Writes training and test data in TFRecord format."""
  user_id = dataframe['user_id'].values
  movie_ids = dataframe['movie_id2'].values
  
  N = dataframe.shape[0]
  for file_id, head_tail in enumerate(split_range(N, num_files)):
    export_file = "{}-{:05d}.tfrecord".format(file_type, file_id)
    export_path = os.path.join(FLAGS.export_dir, export_file)
    with tf.python_io.TFRecordWriter(export_path) as record_writer:
      for i in range(head_tail[0], head_tail[1]):
        sequence_example = make_sequence_example(user_id[i], movie_ids[i])
        record_writer.write(sequence_example.SerializeToString())

def main(_):
  tf.logging.info('Download {} ...'.format(FLAGS.filename))
  movies, ratings = load_movielens_data()
  movieid_to_index = dict(zip(movies['movie_id'].values, np.arange(len(movies))))
  movies['movie_id2'] = movies['movie_id'].apply(lambda x: movieid_to_index[x])
  ratings['movie_id2'] = ratings['movie_id'].apply(lambda x: movieid_to_index[x])
  
  tf.logging.info('Converting dataset ...')
  ratings = ratings[ratings['rating'] > FLAGS.rating_threshold]
  rawdata = (
      ratings[['user_id', 'movie_id2']]
      .groupby('user_id', as_index=False).aggregate(lambda x: list(x)))
  
  if tf.gfile.Exists(FLAGS.export_dir):
    tf.logging.info('Remove {} ...'.format(FLAGS.export_dir))
    tf.gfile.DeleteRecursively(FLAGS.export_dir)
  tf.gfile.MakeDirs(FLAGS.export_dir)
  
  tf.logging.info('Exporting TFRecord to {}'.format(FLAGS.export_dir))
  train_inputs, eval_inputs = split_dataframe(rawdata)
  make_tfrecord_files(dataframe=train_inputs, file_type='train', num_files=8)
  make_tfrecord_files(dataframe=eval_inputs, file_type='eval', num_files=1)
  
  tf.logging.info('Exporting metadata to {}'.format(FLAGS.export_dir))
  with tempfile.TemporaryDirectory() as tmp_dir:
    filename = 'metadata.pickle'
    metadata = {'N': len(movies), 'movies': movies, 'rawdata': rawdata}
    old_path = os.path.join(tmp_dir, filename)
    new_path = os.path.join(FLAGS.export_dir, filename)
    with open(old_path, 'wb') as f:
      pickle.dump(metadata, f)
    tf.gfile.Copy(old_path, new_path, overwrite=True)
    
  tf.logging.info('Exporting an index file for TensorBoard projector')
  with tempfile.TemporaryDirectory() as tmp_dir:
    filename = 'projector_index.tsv'
    old_path = os.path.join(tmp_dir, filename)
    new_path = os.path.join(FLAGS.export_dir, filename)
    movies[['movie_id2', 'title']].to_csv(
        old_path, header=True, index=False, sep='\t')
    tf.gfile.Copy(old_path, new_path, overwrite=True)

if __name__ == '__main__':
  absl_app.run(main)