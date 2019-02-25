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
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from etl import pipeline


def get_args(argv):

  parser = argparse.ArgumentParser()

  parser.add_argument('--output_dir',
                      help='A directory location of output embeddings')

  parser.add_argument('--enable_debug',
                      action='store_true',
                      help='Enable debug options.')

  parser.add_argument('--debug_output_prefix',
                      help='Specify prefix of debug output.')

  parser.add_argument('--transform_temp_dir',
                      default='tft_temp',
                      help='A temp directory used by tf.transform.')

  parser.add_argument('--transform_export_dir',
                      default='tft_out',
                      help='A directory where tft function is saved')

  parser.add_argument('--kind',
                      help='The Cloud Datastore kind to store the items in.')

  parser.add_argument('--limit',
                      type=int,
                      default=1000000,
                      help='Maximum number of records to retrieve from BigQuery.')

  known_args, pipeline_args = parser.parse_known_args(argv)
  return known_args, pipeline_args


def main(argv=None):
  known_args, pipeline_args = get_args(argv)
  pipeline_options = PipelineOptions(pipeline_args)
  setup_options = pipeline_options.view_as(SetupOptions)
  setup_options.save_main_session = True
  pipeline.run(pipeline_options, known_args)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.ERROR)
  main()
