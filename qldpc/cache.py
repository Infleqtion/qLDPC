"""Helper function(s) for caching results

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

import functools
import os
from collections.abc import Callable, Hashable
from typing import Any

import diskcache
import platformdirs


def get_disk_cache(cache_name: str, *, cache_dir: str | None = None) -> diskcache.Cache:
    """Retrieve a dictionary-like cache object."""
    cache_dir = cache_dir or platformdirs.user_cache_dir()
    cache_path = os.path.join(cache_dir, cache_name)
    return diskcache.Cache(cache_path)


def use_disk_cache(
    cache_name: str, *, cache_dir: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to cache results to disk."""

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:

        @functools.wraps(function)
        def function_with_cache(*args: Hashable, **kwargs: Hashable) -> Any:

            # retrieve results from cache, if available
            cache = get_disk_cache(cache_name, cache_dir=cache_dir)
            key = args + tuple(kwargs.items())
            if key in cache:
                return cache[key]

            # compute results and save to cache
            result = function(*args, **kwargs)
            cache[key] = result
            return result

        return function_with_cache

    return decorator
