from collections import Counter

import pytest

from scripts.split_refined_libero import partition_episodes
from wam_h3.refined.data import OFFICIAL_SUITES


def test_seeded_task_stratification_and_duplicate_isolation():
    episodes = [dict(index=suite_index * 50 + i, path=f"{suite}/task.hdf5", suite=suite,
                     episode_sha256=f"{suite}:{i}")
                for suite_index, suite in enumerate(OFFICIAL_SUITES) for i in range(50)]
    episodes.append({**episodes[0], "index": 250, "path": "libero_90/duplicate.hdf5"})
    train, validation, aliases = partition_episodes(episodes, seed=0)
    assert len(train) == 225 and len(validation) == 25
    assert Counter(e["suite"] for e in validation) == dict.fromkeys(OFFICIAL_SUITES, 5)
    assert not {e["episode_sha256"] for e in train} & {e["episode_sha256"] for e in validation}
    assert aliases[episodes[0]["episode_sha256"]] == [0, 250]
    assert partition_episodes(episodes, seed=0) == (train, validation, aliases)
    assert partition_episodes(episodes, seed=1)[1] != validation


def test_split_rejects_unsplittable_task():
    with pytest.raises(ValueError, match="fewer than two"):
        partition_episodes([dict(index=0, path="libero_spatial/task.hdf5", episode_sha256="only")])
