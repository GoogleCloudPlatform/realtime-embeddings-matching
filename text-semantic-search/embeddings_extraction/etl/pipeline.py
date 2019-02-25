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

import apache_beam as beam
from apache_beam.io.gcp.datastore.v1.datastoreio import WriteToDatastore
import tensorflow as tf
from google.cloud.proto.datastore.v1 import entity_pb2
from googledatastore import helper as datastore_helper
import tensorflow_transform.coders as tft_coders

from tensorflow_transform.beam import impl


encoder = None


def get_source_query(limit=1000000):
  query = """
    SELECT
      GENERATE_UUID() as id,
      text
    FROM
    (
        SELECT
          DISTINCT LOWER(title) text
        FROM
          `bigquery-samples.wikipedia_benchmark.Wiki100B`
        WHERE
          ARRAY_LENGTH(split(title,' ')) >= 5
        AND
          language = 'en'
        AND
          LENGTH(title) < 500
     )
    LIMIT {0}
  """.format(limit)
  return query


def embed_text(text):
  import tensorflow_hub as hub
  global encoder
  if encoder is None:
    encoder = hub.Module(
      'https://tfhub.dev/google/universal-sentence-encoder/2')
  embedding = encoder(text)
  return embedding


def parse_articles(csv_line):
    return csv_line.split(',')[1], None


def get_metadata():
  from tensorflow_transform.tf_metadata import dataset_schema
  from tensorflow_transform.tf_metadata import dataset_metadata

  metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
    'id': dataset_schema.ColumnSchema(
      tf.string, [], dataset_schema.FixedColumnRepresentation()),
    'text': dataset_schema.ColumnSchema(
      tf.string, [], dataset_schema.FixedColumnRepresentation())
  }))
  return metadata


def preprocess_fn(input_features):
  import tensorflow_transform as tft
  embedding = tft.apply_function(embed_text, input_features['text'])
  output_features = {
    'id': input_features['id'],
    'embedding': embedding
  }
  return output_features


def create_entity(input_features, kind):
  entity = entity_pb2.Entity()
  datastore_helper.add_key_path(
    entity.key, kind, input_features['id'])
  datastore_helper.add_properties(
    entity, {
      'text': unicode(input_features['text'])
    })
  return entity


def run(pipeline_options, known_args):

  pipeline = beam.Pipeline(options=pipeline_options)
  gcp_project = pipeline_options.get_all_options()['project']

  with impl.Context(known_args.transform_temp_dir):
    articles = (
        pipeline
        | 'Read articles from BigQuery' >> beam.io.Read(beam.io.BigQuerySource(
      project=gcp_project, query=get_source_query(known_args.limit),
      use_standard_sql=True))
    )

    articles_dataset = (articles, get_metadata())
    embeddings_dataset, _ = (
        articles_dataset
        | 'Extract embeddings' >> impl.AnalyzeAndTransformDataset(preprocess_fn)
    )

    embeddings, transformed_metadata = embeddings_dataset

    embeddings | 'Write embeddings to TFRecords' >> beam.io.tfrecordio.WriteToTFRecord(
      file_path_prefix='{0}'.format(known_args.output_dir),
      file_name_suffix='.tfrecords',
      coder=tft_coders.example_proto_coder.ExampleProtoCoder(
        transformed_metadata.schema),
      num_shards=int(known_args.limit/25000)
    )

    (
        articles
        | 'Convert to entity' >> beam.Map(
              lambda input_features: create_entity(
                input_features, known_args.kind))
        | 'Write to Datastore' >> WriteToDatastore(project=gcp_project)
    )

    if known_args.enable_debug:
      embeddings | 'Debug Output' >> beam.io.textio.WriteToText(
        file_path_prefix=known_args.debug_output_prefix,
        file_name_suffix='.txt')

  job = pipeline.run()

  if pipeline_options.get_all_options()['runner'] == 'DirectRunner':
    job.wait_until_finish()