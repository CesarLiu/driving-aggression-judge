"""The parts of the CAT fork's ``responsibility`` package the screens use: the scene arrays
(``scene``), the lane graph (``lanes``), the right-of-way rules (``right_of_way``, with ``stl``)
and the lane frame of the RSS check (``rss``).

Copied from the CAT fork github.com/CesarLiu/catspond, branch counterfactual-responsibility,
commit 1237d71 (2026-10-18), plus the uncommitted ``order_unknown_tie`` option of right_of_way.py,
so that this repository needs neither CAT nor PyTorch. CAT's MIT licence is in LICENSE-CAT.
Changes to the copies:

* lanes.py: ``LaneGraph.near`` computes the point-to-lane distance with NumPy (float32, as before)
  instead of ``torch.cdist``.
* rss.py: ``DT = 0.1`` is defined here instead of imported from ``responsibility.interaction``.

The functions that attribute a collision in a rollout (``right_of_way.rollout_right_of_way``,
``rss``'s rollout helper) import ``responsibility.blame``, which is not copied; the screens do not
use them.
"""
