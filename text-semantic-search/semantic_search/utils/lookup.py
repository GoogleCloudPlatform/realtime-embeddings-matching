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
from google.cloud import datastore


class DatastoreUtil:

  def __init__(self, kind):
    logging.info("Initialising datastore lookup utility...")
    self.kind = kind
    self.client = datastore.Client()
    logging.info("Datastore lookup utility initialised.")

  def get_items(self, keys):

    keys = [self.client.key(self.kind, key)
            for key in keys]

    items = self.client.get_multi(keys)
    return items




