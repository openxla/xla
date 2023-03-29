# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import collections

from absl.testing import absltest

from xla.python import xla_client

pytree = xla_client._xla.pytree


def make_fake_pickling():
  values = []
  values_by_key = {}

  def fake_loads(key):
    return values_by_key[key]

  def fake_dumps(value):
    key = bytes(str(len(values)), "utf-8")
    values.append(value)
    values_by_key[key] = value
    return key

  return values, fake_loads, fake_dumps


ExampleType = collections.namedtuple("ExampleType", "field0 field1")


class ExampleType2:

  def __init__(self, field0, field1):
    self.field0 = field0
    self.field1 = field1

  def to_iterable(self):
    return [self.field0, self.field1], (None,)


def from_iterable(state, values):
  del state
  return ExampleType2(field0=values[0], field1=values[1])


pytree.register_node(ExampleType2, ExampleType2.to_iterable, from_iterable)


class PyTreeTest(absltest.TestCase):

  def testSerializeDeserializeNoPickle(self):
    o = object()
    t = pytree.flatten(({"a": o, "b": o}, [o, (o, o), None]))[1]
    self.assertEqual(pytree.deserialize(t.serialize()), t)

  def testSerializeWithFallback(self):
    values, fake_loads, fake_dumps = make_fake_pickling()
    o = object()
    t = pytree.flatten({"a": ExampleType(field0=o, field1=o)})[1]
    self.assertEqual(
        pytree.deserialize(
            t.serialize(pickling_fn=fake_dumps), unpickling_fn=fake_loads
        ),
        t,
    )
    self.assertLen(values, 1)
    self.assertEqual(values[0], [ExampleType])

  def testRegisteredType(self):
    values, fake_loads, fake_dumps = make_fake_pickling()
    o = object()
    t = pytree.flatten({"a": ExampleType2(field0=o, field1=o)})[1]
    self.assertEqual(
        pytree.deserialize(
            t.serialize(pickling_fn=fake_dumps), unpickling_fn=fake_loads
        ),
        t,
    )
    self.assertLen(values, 1)
    self.assertEqual(values[0], [ExampleType2, (None,)])


if __name__ == "__main__":
  absltest.main()
