#!/usr/bin/python
#
# Copyright 2018 Google LLC
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
import tensorflow_hub as hub
import logging

MODULE_URL = 'https://tfhub.dev/google/universal-sentence-encoder/2'


class EmbedUtil:

  def __init__(self):

    logging.info("Initialising embedding utility...")
    embed_module = hub.Module(MODULE_URL)
    placeholder = tf.placeholder(dtype=tf.string)
    embed = embed_module(placeholder)
    session = tf.Session()
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    logging.info('tf.Hub module is loaded.')

    def _embeddings_fn(sentences):
      computed_embeddings = session.run(
        embed, feed_dict={placeholder: sentences})
      return computed_embeddings

    self.embedding_fn = _embeddings_fn
    logging.info("Embedding utility initialised.")

  def extract_embeddings(self, query):
    return self.embedding_fn([query])[0]




