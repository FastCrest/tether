"""Accel GR00T N1.7 action head (VLSA + 4-step AlternateVLDiT, per-token FP8).

:class:`~tether.accel.groot_n17.frontend.AccelGrootActionGenerator` replaces a
native ``Gr00tN1d7ActionHead.get_action``; the native backbone, processors and
state encoder keep running. Tether has no native N1.7 serving loop yet, so this
engine is reachable only from code that already holds an upstream N1.7 policy.
"""

#: Native N1.7 action geometry (shared by the engine and the generator without importing Triton).
HORIZON, ACT_DIM = 40, 132
