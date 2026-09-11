import json
from pathlib import Path


def summarize(output_dir):
    out = Path(output_dir)
    suites, episodes = {}, []
    for f in sorted(out.glob("*/gpu*_task*_results.json")):
        r = json.loads(f.read_text())
        suite, task = r["task_suite"], int(r["task_id"])
        n_ok, n = len(r["success_episodes"]), int(r["total_episodes"])
        s = suites.setdefault(suite, {"tasks": {}, "episodes": 0, "successes": 0})
        s["tasks"][str(task)] = dict(task_description=r.get("task_description"), episodes=n, successes=n_ok,
                                success_rate=n_ok / max(n, 1), duration=r.get("duration"))
        s["episodes"] += n
        s["successes"] += n_ok
        for i in range(n):
            episodes.append(dict(suite=suite, task_id=task, task_description=r.get("task_description"),
                                 episode=i, success=i in r["success_episodes"]))
    for s in suites.values():
        s["success_rate"] = s["successes"] / max(s["episodes"], 1)
        s["tasks"] = dict(sorted(s["tasks"].items(), key=lambda kv: int(kv[0])))
    summary = dict(
        suites={k: v["success_rate"] for k, v in suites.items()},
        success_rate_mean_over_suites=sum(v["success_rate"] for v in suites.values()) / max(len(suites), 1),
        episodes=sum(v["episodes"] for v in suites.values()),
        successes=sum(v["successes"] for v in suites.values()),
    )
    (out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (out / "suites.json").write_text(json.dumps(suites, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
