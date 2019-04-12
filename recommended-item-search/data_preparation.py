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

import argparse
import logging

import numpy as np
import os
import pandas
import pickle
import tensorflow as tf
import urllib
import zipfile

def download(base_url, file_name):
  """Download movielens dataset archives.

  Args:
    base_url: A base url of movielens dataset.
    file_name: A file name of downloaded zipped dataset.
  """
  path = os.path.join(base_url, file_name)
  urllib.request.urlretrieve(path, file_name)

def extract(file_name):
  """Extract movielens dataset from downloaded archives
  
  Args:
    file_name: A file name of downloaded zipped dataset.
  """
  zipfile.ZipFile(file_name, 'r').extractall()

def load_movielens_data(base_url, file_name):
  download(base_url, file_name)
  extract(file_name)

  movies_def = {
    'path': os.path.join(file_name.split('.')[0], 'movies.csv'),
    'cols': ['movie_id', 'title', 'genres']
  }
  movies = pandas.read_csv(
      movies_def['path'], names=movies_def['cols'], header=0)

  ratings_def = {
    'path': os.path.join(file_name.split('.')[0], 'ratings.csv'),
    'cols': ['user_id', 'movie_id', 'rating', 'unix_timestamp']
  }
  ratings = pandas.read_csv(
      ratings_def['path'], names=ratings_def['cols'], header=0)

  return movies, ratings

def split_dataframe(df, holdout_fraction=0.1):
  """Splits a DataFrame into training and test sets.
  Args:
    df: a dataframe.
    holdout_fraction: fraction of dataframe rows to use in the test set.
  Returns:
    train: dataframe for training
    test: dataframe for testing
  """
  test = df.sample(frac=holdout_fraction, replace=False)
  train = df[~df.index.isin(test.index)]
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

def make_tfrecord_files(df, output_file='train.tfrecord'):
  """Writes training and test data in TFRecord format.""" 
  user_id = df['user_id'].values
  movie_ids = df['movie_id2'].values
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for i in range(df.shape[0]):
      sequence_example = make_sequence_example(user_id[i], movie_ids[i])
      record_writer.write(sequence_example.SerializeToString())

def preprocess(args):
  tf.logging.info('downloading movielens dataset...')
  movies, ratings = load_movielens_data(args.base_url, args.file_name)

  tf.logging.info('converting movielens dataset ...')
  movieid_to_index = dict(zip(movies['movie_id'].values, np.arange(len(movies))))
  movies['movie_id2'] = movies['movie_id'].apply(lambda x: movieid_to_index[x])
  ratings['movie_id2'] = ratings['movie_id'].apply(lambda x: movieid_to_index[x])
  # TODO(yaboo): decide this line is still needed.
  ratings = ratings[ratings['rating'] > 4.0]
  rawdata = (
    ratings[['user_id', 'movie_id2']]
    .groupby('user_id', as_index=False)
    .aggregate(lambda x: list(x)))
    
  tf.logging.info('writing tfrecord files ...')
  train_rawdata, valid_rawdata = split_dataframe(rawdata)
  make_tfrecord_files(train_rawdata, output_file='train.tfrecord')
  make_tfrecord_files(valid_rawdata, output_file='eval.tfrecord')

  tf.logging.info('writing metadata ...')
  metadata = {'N': len(movies), 'movies': movies, 'rawdata': rawdata}
  with open('metadata.pickle', 'wb') as f:
    pickle.dump(metadata, f)

  tf.logging.info('writing index data for TF projector ...')
  movies[['movie_id2', 'title']].to_csv(
      'projector_index.tsv', header=True, index=False, sep='\t')

  tf.logging.info('preprocess finished.')

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_url',
                      default='http://files.grouplens.org/datasets/movielens/',
                      help='A base url of movielens dataset.')
  parser.add_argument('--file_name',
                      default='ml-20m.zip',
                      help='A file name of movielens dataset.')

  return parser.parse_args()

if __name__ == '__main__':
  preprocess(get_args())