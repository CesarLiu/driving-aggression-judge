"""The parts of the CAT fork's ``responsibility`` package the skill uses.

* The screens use the scene arrays (``scene``), the lane graph (``lanes``), the right-of-way
  rules (``right_of_way``, with ``stl``) and the lane frame of the RSS check (``rss``).
* The duty x cost x attribution judge (scripts/judge.py) also uses:
  * ``forced_yield``: the braking a crossing entry demands of the holder;
  * ``attribution``: who caused a crossing, a join or a following episode;
  * ``forced_merge``: the braking a cut-in or tailgating demands;
  * ``aggression_judge``: the verdict against thresholds calibrated on logged driving.

Copied from the CAT fork github.com/CesarLiu/catspond, branch counterfactual-responsibility.
scene.py, stl.py, lanes.py and rss.py come from commit 1237d71 (2026-10-18); right_of_way.py and
the four judge modules from commit f74ab27 (2026-10-29), which adds the ``order_unknown_tie`` and
``traffic_side`` options. With these copies the repository needs neither CAT nor PyTorch. CAT's
MIT licence is in LICENSE-CAT. Changes to the copies:

* lanes.py: ``LaneGraph.near`` computes the point-to-lane distance with NumPy (float32, as before)
  instead of ``torch.cdist``.
* rss.py: ``DT = 0.1`` is defined here instead of imported from ``responsibility.interaction``.

Some modules have been left behind:
* ``responsibility.blame`` is not copied. Only the functions that attribute a collision in a
  rollout import it (``right_of_way.rollout_right_of_way`` and ``rss``'s rollout helper), and
  nothing here uses them.
* The judge modules' docstrings name scripts that stay in the CAT fork: make_go_first.py,
  make_cut_in.py, evaluate_aggression_sets.py and attribute_pairs.py. Those scripts build the
  ground-truth sets the judge was validated on; see references/judge.md.
"""
