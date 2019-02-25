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
import numpy as np
import logging
import pickle
import os
from annoy import AnnoyIndex

VECTOR_LENGTH = 512
METRIC = 'angular'


def build_index(embedding_files_pattern, index_filename,
                num_trees=100):

  annoy_index = AnnoyIndex(VECTOR_LENGTH, metric=METRIC)
  mapping = {}

  embed_files = tf.gfile.Glob(embedding_files_pattern)[:250]
  logging.info('{} embedding files are found.'.format(len(embed_files)))

  item_counter = 0
  for f, embed_file in enumerate(embed_files):
    logging.info('Loading embeddings in file {} of {}...'.format(
      f, len(embed_files)))
    record_iterator = tf.python_io.tf_record_iterator(
      path=embed_file)

    for string_record in record_iterator:
      example = tf.train.Example()
      example.ParseFromString(string_record)
      string_identifier = example.features.feature['id'].bytes_list.value[0]
      mapping[item_counter] = string_identifier
      embedding = np.array(
        example.features.feature['embedding'].float_list.value)
      annoy_index.add_item(item_counter, embedding)
      item_counter += 1

    logging.info('Loaded {} items to the index'.format(item_counter))

  logging.info('Start building the index with {} trees...'.format(num_trees))
  annoy_index.build(n_trees=num_trees)
  logging.info('Index is successfully built.')
  logging.info('Saving index to disk...')
  annoy_index.save(index_filename)
  logging.info('Index is saved to disk.')
  logging.info("Index file size: {} GB".format(
    round(os.path.getsize(index_filename) / float(1024 ** 3), 2)))
  annoy_index.unload()
  logging.info('Saving mapping to disk...')
  with open(index_filename + '.mapping', 'wb') as handle:
    pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
  logging.info('Mapping is saved to disk.')
  logging.info("Mapping file size: {} MB".format(
    round(os.path.getsize(index_filename + '.mapping') / float(1024 ** 2), 2)))

