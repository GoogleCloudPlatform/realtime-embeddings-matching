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

import embedding
import matching
import lookup
import os
import logging
from httplib2 import Http
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from oauth2client.client import GoogleCredentials


KIND = 'wikipedia'
GCS_BUCKET = ''
GCS_INDEX_LOCATION = '{}/index/5M-embeds.index100'.format(KIND)
INDEX_FILE = 'embeds.index'
CHUNKSIZE = 16 * 1024 * 1024


def _download_from_gcs(gcs_services, bucket_name, gcs_location, local_file_name):

  print('Downloading file {} to {}...'.format(
    "gs://{}/{}".format(bucket_name, gcs_location), local_file_name))

  with open(local_file_name, 'wb') as file_writer:
    request = gcs_services.objects().get_media(bucket=bucket_name, object=gcs_location)
    media = MediaIoBaseDownload(file_writer, request, chunksize=CHUNKSIZE)
    download_complete = False
    while not download_complete:
        progress, download_complete = media.next_chunk()

  print('File {} downloaded to {}.'.format(
    "gs://{}/{}".format(bucket_name, gcs_location), local_file_name))

  print("File size: {} GB".format(
    round(os.path.getsize(local_file_name) / float(1024 ** 3), 2)))


def download_artefacts(index_file, bucket_name, gcs_index_location):
  http = Http()
  credentials = GoogleCredentials.get_application_default()
  credentials.authorize(http)
  gcs_services = build('storage', 'v1', http=http)
  _download_from_gcs(gcs_services, bucket_name, gcs_index_location, index_file)
  _download_from_gcs(gcs_services, bucket_name, gcs_index_location + '.mapping', index_file + '.mapping')


class SearchUtil:

  def __init__(self):

    print("Initialising search utility...")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    index_file = os.path.join(dir_path, INDEX_FILE)

    print("Downloading index artefacts...")
    download_artefacts(index_file, GCS_BUCKET, GCS_INDEX_LOCATION)
    print("Index artefacts downloaded.")

    print("Initialising matching util...")
    self.match_util = matching.MatchingUtil(index_file)
    print("Matching util initialised.")

    print("Initialising embedding util...")
    self.embed_util = embedding.EmbedUtil()
    print("Embedding util initialised.")

    print("Initialising datastore util...")
    self.datastore_util = lookup.DatastoreUtil(KIND)
    print("Datastore util is initialised.")

    print("Search utility is up and running.")

  def search(self, query, num_matches=10):
    query_embedding = self.embed_util.extract_embeddings(query)
    item_ids = self.match_util.find_similar_items(query_embedding, num_matches)
    items = self.datastore_util.get_items(item_ids)
    return items



