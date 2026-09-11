import json

from wam_h3.eval.results import summarize


def write(d, suite, gpu, task, succ, fail, desc="do it"):
    p = d / suite
    p.mkdir(parents=True, exist_ok=True)
    (p / f"gpu{gpu}_task{task}_results.json").write_text(json.dumps(dict(
        benchmark="libero", task_suite=suite, task_id=task, task_description=desc, successes=len(succ),
        total_episodes=len(succ) + len(fail), gpu_id=gpu, success_episodes=succ, failure_episodes=fail, duration=12.5)))


def test_summarize_writes_episode_task_suite_and_overall(tmp_path):
    write(tmp_path, "libero_spatial", 0, 0, [0, 1, 2], [3])
    write(tmp_path, "libero_spatial", 1, 1, [0], [1, 2, 3])
    write(tmp_path, "libero_goal", 0, 0, [0, 1], [], desc="goal task")
    s = summarize(tmp_path)
    eps = [json.loads(l) for l in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(eps) == 10 and sum(e["success"] for e in eps) == 6
    assert {"suite", "task_id", "task_description", "episode", "success"} <= set(eps[0])
    suites = json.loads((tmp_path / "suites.json").read_text())
    assert suites["libero_spatial"]["success_rate"] == 0.5 and suites["libero_spatial"]["episodes"] == 8
    assert suites["libero_spatial"]["tasks"]["1"]["success_rate"] == 0.25
    assert suites["libero_goal"]["success_rate"] == 1.0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary == s
    assert abs(summary["success_rate_mean_over_suites"] - 0.75) < 1e-9
    assert summary["episodes"] == 10 and summary["successes"] == 6 and set(summary["suites"]) == {"libero_spatial", "libero_goal"}
