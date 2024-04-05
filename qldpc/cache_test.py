"""Unit tests for cache.py

   Copyright 2023 The qLDPC Authors and Infleqtion Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import unittest.mock

import qldpc.cache


def test_use_disk_cache() -> None:
    """Cache function outputs."""

    @qldpc.cache.use_disk_cache("test")
    def get_five(arg: str) -> int:
        return 5

    # use cache to save/retrieve results
    cache: dict[tuple[str], int] = {}
    with unittest.mock.patch("diskcache.Cache", return_value=cache):
        get_five("test")  # save results to cache
        assert cache == {("test",): 5}  # check cache
        assert cache[("test",)] == get_five("test")  # retrieve results
