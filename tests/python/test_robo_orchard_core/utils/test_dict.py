# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import pytest

from robo_orchard_core.utils.dict import flatten_dict


class TestFlattenDict:
    def test_default_mode_recursively_flattens_sequences(self):
        """The default mode preserves indexed sequence-key behavior."""

        assert flatten_dict(
            {
                "a": 1,
                "b": {
                    "c": 2,
                    "d": [3, 4],
                },
            }
        ) == {
            "a": 1,
            "b/c": 2,
            "b/d/index_0": 3,
            "b/d/index_1": 4,
        }

    def test_json_mode_serializes_complete_nested_sequence_subtrees(self):
        """JSON mode keeps recursive sequence content in one leaf value."""

        assert flatten_dict(
            {
                "model": {
                    "blocks": [
                        {
                            "type": "attention",
                            "layers": [1, 2],
                        },
                        {
                            "type": "mlp",
                            "options": {
                                "hidden_sizes": (1024, 4096),
                            },
                        },
                    ],
                    "empty": [],
                },
            },
            sep=".",
            sequence_mode="json",
        ) == {
            "model.blocks": (
                '[{"layers":[1,2],"type":"attention"},'
                '{"options":{"hidden_sizes":[1024,4096]},"type":"mlp"}]'
            ),
            "model.empty": "[]",
        }

    def test_json_mode_reports_the_sequence_key_on_serialization_error(self):
        """JSON serialization errors identify the affected flattened key."""

        with pytest.raises(TypeError, match="Sequence at key 'config.items'"):
            flatten_dict(
                {"config": {"items": [{1, 2}]}},
                sep=".",
                sequence_mode="json",
            )

    def test_invalid_sequence_mode_raises_value_error(self):
        """Unsupported sequence modes fail at the public utility boundary."""

        with pytest.raises(ValueError, match="sequence_mode must be"):
            flatten_dict({}, sequence_mode="preserve")  # type: ignore[arg-type]

    def test_invalid_top_level_input_raises_type_error(self):
        """Non-container top-level values remain unsupported."""

        with pytest.raises(TypeError, match="dictionary, list, or tuple"):
            flatten_dict(1)  # type: ignore[arg-type]
