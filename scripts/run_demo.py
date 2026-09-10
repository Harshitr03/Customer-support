"""One command to reproduce headline results end-to-end.

Default mode is fully offline and makes zero API calls: every LLM/embedding
call needed for the headline numbers is served from the committed replay
cache at data/llm_cache/ (see support_agent.llm_client's replay fallback).
Pass --live to make real Gemini API calls instead (needs GEMINI_API_KEY in
.env); pass --export-cache alongside --live to copy whatever that run
touched into data/llm_cache/ for the next offline run.

Controller ruling (task-13-addendum.md ruling 1): both src/ and the repo
root must be on sys.path BEFORE importing support_agent/eval, so `from eval
import ...` resolves when this file is run directly (`python
scripts/run_demo.py`), not just via `pytest` (whose `pythonpath` ini option
already covers both directories for the test suite).
"""
import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

# Ruling 3: offline by default. Set before importing llm_client (transitively,
# via the imports below) so the default takes effect even if some future
# import path ends up making a call at import time; main() flips this off
# for --live before any stage runs.
os.environ.setdefault("SUPPORT_AGENT_OFFLINE", "1")

from support_agent import config, data_prep, pipeline, retrieve  # noqa: E402
from support_agent.llm_client import (  # noqa: E402
    OfflineModeError, QuotaExhaustedError, export_touched_cache,
)
from eval import build_golden_set, human_agreement, run_eval  # noqa: E402

FIXED_MESSAGE = "@SpotifyCares the app keeps crashing every time I press play on my iPhone"

N_STAGES = 6
STAGE_LABELS = {
    1: "Building thread pool (first run parses the raw CSV)",
    2: "Building retrieval index (no-op: committed data/kb/ index present)",
    3: "Building golden set (no-op: committed data/golden/golden_eval.csv present)",
    4: "Running eval harness",
    5: "Judge/human agreement",
    6: "Live demo: one message end-to-end",
}


def _banner(n: int) -> None:
    print(f"\n[{n}/{N_STAGES}] {STAGE_LABELS[n]}...")


def _stage5_human_agreement():
    """Ruling 4.5: only run human_agreement.main() if the human scores file
    exists; otherwise print one line explaining why it was skipped, rather
    than human_agreement's own multi-line "how to produce it" instructions."""
    path = config.GOLDEN_DIR / "human_scores.csv"
    if not path.exists():
        print(
            "Human scores not added yet (data/golden/human_scores.csv missing) "
            "-- skipping; see README's human-agreement workflow."
        )
        return None
    return human_agreement.main()


def _print_live_example(out: dict) -> None:
    evidence = out.get("evidence") or []
    top_score = evidence[0]["score"] if evidence else float("nan")
    print("\n--- Live demo: one message end-to-end ---")
    print("message:", FIXED_MESSAGE)
    print("intent:", out["intent"], "| confidence:", f"{out['confidence']:.2f}",
          "| escalate:", out["escalate"])
    print("reason:", out["reason"])
    print("top evidence score:", f"{top_score:.4f}")
    print("reply:", out["reply"])


def run_stages() -> int:
    """Run all 6 stages in order. Returns a process exit code: 0 on
    success, 1 if offline mode hit a call with no cached/replayed response
    (a helpful message identifying the stage has already been printed)."""
    stage = 0
    try:
        stage = 1
        t0 = time.monotonic()
        _banner(stage)
        data_prep.build_pool()
        print(f"      done in {time.monotonic() - t0:.1f}s")

        stage = 2
        t0 = time.monotonic()
        _banner(stage)
        retrieve.build_index()
        print(f"      done in {time.monotonic() - t0:.1f}s")

        stage = 3
        t0 = time.monotonic()
        _banner(stage)
        build_golden_set.build()
        print(f"      done in {time.monotonic() - t0:.1f}s")

        stage = 4
        t0 = time.monotonic()
        _banner(stage)
        run_eval.main()
        print(f"      done in {time.monotonic() - t0:.1f}s")

        stage = 5
        t0 = time.monotonic()
        _banner(stage)
        _stage5_human_agreement()
        print(f"      done in {time.monotonic() - t0:.1f}s")

        stage = 6
        t0 = time.monotonic()
        _banner(stage)
        out = pipeline.handle(FIXED_MESSAGE)
        _print_live_example(out)
        print(f"      done in {time.monotonic() - t0:.1f}s")
    except OfflineModeError as exc:
        print(
            f"\nStopped at stage {stage}/{N_STAGES} ({STAGE_LABELS[stage]}): {exc}\n"
            "This run is offline (SUPPORT_AGENT_OFFLINE=1) and the committed replay\n"
            "cache (data/llm_cache/) doesn't have a response for this call yet.\n"
            "To proceed:\n"
            "  - rerun with --live (needs GEMINI_API_KEY in .env) to make real API calls, or\n"
            "  - if you're the maintainer, run with --live --export-cache to refresh the\n"
            "    committed replay cache so graders can reproduce this offline."
        )
        return 1
    except QuotaExhaustedError as exc:
        print(f"\nStopped at stage {stage}/{N_STAGES} ({STAGE_LABELS[stage]}): {exc}")
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true",
                    help="Make real Gemini API calls instead of replaying the "
                         "committed cache (needs GEMINI_API_KEY in .env).")
    p.add_argument("--export-cache", action="store_true",
                    help="After a --live run, copy every LLM/embedding call this "
                         "process touched into data/llm_cache/ (skips files "
                         "already there). Has no effect without --live.")
    p.add_argument("--no-replay", action="store_true",
                    help="With --live, ignore the committed replay cache "
                         "(data/llm_cache/) so any call not already in the local "
                         "cache/ hits the network. Delete data/cache/ too for a "
                         "full recompute. Only valid together with --live.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.no_replay and not args.live:
        print(
            "\n--no-replay is only valid together with --live (offline mode never "
            "reads the replay cache in the first place -- it's the default)."
        )
        return 1

    if args.live:
        os.environ.pop("SUPPORT_AGENT_OFFLINE", None)
        # Fix round 1: pre-flight check. config (imported at module load, above)
        # already called load_dotenv() by this point, so os.environ reflects
        # .env. Without this check, --live with no key ran stage 1 (up to ~1
        # min CSV parse) and then crashed with a raw traceback deep inside
        # run_eval.main() the first time it needed a client -- never print the
        # key's value, just whether it's present.
        if not os.environ.get("GEMINI_API_KEY"):
            print(
                "\n--live requires GEMINI_API_KEY, which is not set. Set it in .env "
                "(see .env.example) or in the environment, then rerun with --live; "
                "or omit --live to reproduce the headline results offline from the "
                "committed replay cache (data/llm_cache/)."
            )
            return 1
        if args.no_replay:
            os.environ["SUPPORT_AGENT_NO_REPLAY"] = "1"
        else:
            os.environ.pop("SUPPORT_AGENT_NO_REPLAY", None)
    else:
        os.environ["SUPPORT_AGENT_OFFLINE"] = "1"
        os.environ.pop("SUPPORT_AGENT_NO_REPLAY", None)

    code = run_stages()

    if args.export_cache:
        if not args.live:
            print("\n--export-cache has no effect without --live (nothing new was fetched).")
        else:
            n, nbytes = export_touched_cache()
            print(f"\n--export-cache: copied {n} file(s), {nbytes} bytes into {config.REPLAY_CACHE_DIR}")

    return code


if __name__ == "__main__":
    sys.exit(main())
