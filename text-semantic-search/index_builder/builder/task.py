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

import logging
import argparse
from datetime import datetime
import index
from httplib2 import Http
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from oauth2client.client import GoogleCredentials


LOCAL_INDEX_FILE = 'embeds.index'
CHUNKSIZE = 64 * 1024 * 1024


def _upload_to_gcs(gcs_services, local_file_name, bucket_name, gcs_location):

  logging.info('Uploading file {} to {}...'.format(
    local_file_name, "gs://{}/{}".format(bucket_name, gcs_location)))

  media = MediaFileUpload(
    local_file_name, mimetype='application/octet-stream', chunksize=CHUNKSIZE, resumable=True)
  request = gcs_services.objects().insert(
    bucket=bucket_name, name=gcs_location, media_body=media)
  response = None
  while response is None:
    progress, response = request.next_chunk()

  logging.info('File {} uploaded to {}.'.format(
    local_file_name, "gs://{}/{}".format(bucket_name, gcs_location)))


def upload_artefacts(gcs_index_file):

  http = Http()
  credentials = GoogleCredentials.get_application_default()
  credentials.authorize(http)
  gcs_services = build('storage', 'v1', http=http)

  split_list = gcs_index_file[5:].split('/', 1)
  bucket_name = split_list[0]
  blob_path = split_list[1] if len(split_list) == 2 else None
  _upload_to_gcs(gcs_services, LOCAL_INDEX_FILE, bucket_name, blob_path)
  _upload_to_gcs(gcs_services, LOCAL_INDEX_FILE+'.mapping', bucket_name, blob_path+'.mapping')


def get_args():

  args_parser = argparse.ArgumentParser()

  args_parser.add_argument(
    '--embedding-files',
    help='GCS or local paths to embedding files',
    required=True
  )

  args_parser.add_argument(
    '--index-file',
    help='GCS or local paths to output index file',
    required=True
  )

  args_parser.add_argument(
    '--num-trees',
    help='Number of trees to build in the index',
    default=1000,
    type=int
  )

  args_parser.add_argument(
    '--job-dir',
    help='GCS or local paths to job package'
  )

  return args_parser.parse_args()


def main():

  args = get_args()

  time_start = datetime.utcnow()
  logging.info("Index building started...")
  index.build_index(args.embedding_files, LOCAL_INDEX_FILE, args.num_trees)
  time_end = datetime.utcnow()
  logging.info("Index building  finished.")
  time_elapsed = time_end - time_start
  logging.info("Index building  elapsed time: {} seconds".format(time_elapsed.total_seconds()))

  time_start = datetime.utcnow()
  logging.info("Uploading index artefacts started...")
  upload_artefacts(args.index_file)
  time_end = datetime.utcnow()
  logging.info("Uploading index artefacts finished.")
  time_elapsed = time_end - time_start
  logging.info("Uploading index artefacts elapsed time: {} seconds".format(time_elapsed.total_seconds()))


if __name__ == '__main__':
    main()