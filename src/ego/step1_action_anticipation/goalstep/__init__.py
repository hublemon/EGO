"""Ego4D GoalStep -> Step-1 action-anticipation pipeline.

Everything needed to go from raw GoalStep annotations to a trainable Z=1 index
lives here: the EK100-style verb/noun/action parser, the taxonomy/registry
builders, the index builder, the trainer, and the committed ``taxonomy/`` and
``index/`` artifacts those steps produce (kept in-tree because training cannot
start without them).
"""
