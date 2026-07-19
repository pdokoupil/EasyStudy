"""EasyStudy: a framework for deploying customizable interactive user studies.

This top-level `easystudy` package is a thin CLI + loader around the actual application
code, which lives in the `server/` directory of the source repository and is vendored
here (unmodified) as `easystudy._vendor_server` — see `easystudy._bootstrap` for why.
"""
__version__ = "2.0.0.dev0"
