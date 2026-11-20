"""Vendored myhhub/stock modules (Apache-2.0) + supplement adapter.

Only two upstream modules are vendored: ``cyq.py`` (chip distribution)
and ``pattern_recognitions.py`` (K-line pattern recognition), plus
``patterns.py`` (the 61-pattern table extracted from upstream
``tablestructure.py``). ``adapter.py`` feeds them from AkShare push2his
daily bars and returns never-raising result dicts. See ``adapter.py``
for the full contract and the upstream commit.
"""
