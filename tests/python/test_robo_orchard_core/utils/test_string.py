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

from robo_orchard_core.utils.string import (
    resolve_matching_names,
    resolve_matching_names_values,
)


class TestResolveMatchingNames:
    def test_default_order_follows_target_strings(self):
        indices, names = resolve_matching_names(
            ["a|c", "b"],
            ["a", "b", "c", "d", "e"],
        )

        assert indices == [0, 1, 2]
        assert names == ["a", "b", "c"]

    def test_preserve_order_follows_query_keys(self):
        indices, names = resolve_matching_names(
            ["a|c", "b"],
            ["a", "b", "c", "d", "e"],
            preserve_order=True,
        )

        assert indices == [0, 2, 1]
        assert names == ["a", "c", "b"]


class TestResolveMatchingNamesValues:
    def test_default_order_follows_target_strings(self):
        indices, names, values = resolve_matching_names_values(
            {"a|d|e": 1, "b|c": 2},
            ["a", "b", "c", "d", "e"],
        )

        assert indices == [0, 1, 2, 3, 4]
        assert names == ["a", "b", "c", "d", "e"]
        assert values == [1, 2, 2, 1, 1]

    def test_preserve_order_follows_query_keys(self):
        indices, names, values = resolve_matching_names_values(
            {"a|d|e": 1, "b|c": 2},
            ["a", "b", "c", "d", "e"],
            preserve_order=True,
        )

        assert indices == [0, 3, 4, 1, 2]
        assert names == ["a", "d", "e", "b", "c"]
        assert values == [1, 1, 1, 2, 2]
