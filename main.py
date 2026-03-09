"""PokerArena — LLMs play heads-up No-Limit Texas Hold'em."""

import os
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from player import LLMPlayer
from game import HandState, fmt, make_deck

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
STARTING_STACK = 100  # BB units
MAX_WORKERS = 5


def run_hand(hand_id: str, hand_num: int, dealer_idx: int,
             player_cfgs: list, deck=None) -> dict:
    """Run a single hand with fresh players."""
    players = []
    for cfg in player_cfgs:
        p = LLMPlayer(name=cfg["name"], model=cfg["model"], api_key=API_KEY)
        p.stack = STARTING_STACK
        players.append(p)

    result = HandState(players, dealer_idx, hand_num, deck=deck).run()
    result["hand_id"] = hand_id
    result["tokens"] = {
        p.name: (p.total_tokens_in, p.total_tokens_out) for p in players
    }
    result["decisions"] = [
        {
            "actor": d.player_name,
            "street": d.street,
            "action": d.action,
            "amount": d.amount,
            "tokens_in": d.tokens_in,
            "tokens_out": d.tokens_out,
            "reasoning_tokens": d.reasoning_tokens,
            "latency_ms": d.latency_ms,
            "reasoning": d.reasoning,
            "error": d.error,
            "valid_first_try": d.valid_first_try,
            "repair_attempted": d.repair_attempted,
            "forced_action": d.forced_action,
        }
        for p in players
        for d in p.decisions
    ]
    return result


def run_pair(pair_id: str, pair_num: int, player_cfgs: list) -> dict:
    """Run a duplicate pair: same deck, swapped seats so each player gets the other's cards."""
    deck = make_deck()

    # Hand A: original seat order, player_cfgs[0] is dealer/SB
    hand_a = run_hand(
        hand_id=f"{pair_id}_a",
        hand_num=pair_num * 2 - 1,
        dealer_idx=0,
        player_cfgs=player_cfgs,
        deck=list(deck),
    )
    # Hand B: reversed seat order — players swap cards, same dealer_idx=0
    hand_b = run_hand(
        hand_id=f"{pair_id}_b",
        hand_num=pair_num * 2,
        dealer_idx=0,
        player_cfgs=[player_cfgs[1], player_cfgs[0]],
        deck=list(deck),
    )

    player_a = player_cfgs[0]["name"]
    net_a = (hand_a["stacks"][player_a] - STARTING_STACK) + \
            (hand_b["stacks"][player_a] - STARTING_STACK)

    return {
        "pair_id": pair_id,
        "pair_num": pair_num,
        "net_bb": {player_a: net_a, player_cfgs[1]["name"]: -net_a},
        "hands": [hand_a, hand_b],
    }


def run_matchup(player_cfgs: list, num_pairs: int, quiet: bool = False):
    """Run a full matchup between two players using duplicate pairs."""
    player_a = player_cfgs[0]["name"]
    player_b = player_cfgs[1]["name"]

    print(f"\n{'='*60}")
    print(f"  {player_a}  vs  {player_b}")
    print(f"  {num_pairs} pairs ({num_pairs * 2} hands), {MAX_WORKERS} parallel workers")
    print(f"{'='*60}")

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(run_pair, str(uuid.uuid4()), p, player_cfgs)
            for p in range(1, num_pairs + 1)
        ]
        pair_results = [f.result() for f in futures]

    pair_results.sort(key=lambda r: r["pair_num"])
    elapsed = time.time() - t_start

    # Collect all hands and decisions
    all_hands = []
    all_decisions = []
    for pr in pair_results:
        for hand in pr["hands"]:
            all_hands.append(hand)
            for dec in hand["decisions"]:
                dec["hand_id"] = hand["hand_id"]
                all_decisions.append(dec)

    # Write three JSONL files + text log
    os.makedirs("logs", exist_ok=True)
    tag = f"{player_a}_vs_{player_b}"

    pairs_file = f"logs/{tag}_pairs.jsonl"
    hands_file = f"logs/{tag}_hands.jsonl"
    decisions_file = f"logs/{tag}_decisions.jsonl"
    log_file = f"logs/{tag}.log"

    with open(pairs_file, "a") as f:
        for pr in pair_results:
            f.write(json.dumps({
                "pair_id": pr["pair_id"],
                "pair_num": pr["pair_num"],
                "net_bb": pr["net_bb"],
            }) + "\n")

    with open(hands_file, "a") as f:
        for h in all_hands:
            f.write(json.dumps({
                "hand_id": h["hand_id"],
                "hand_num": h["hand_num"],
                "dealer_idx": h["dealer_idx"],
                "winner": h["winner"],
                "pot": h["pot"],
                "stacks": h["stacks"],
                "hole_cards": h["hole_cards"],
                "board": h["board"],
                "action_history": h["action_history"],
            }) + "\n")

    with open(decisions_file, "a") as f:
        for dec in all_decisions:
            f.write(json.dumps(dec) + "\n")

    with open(log_file, "a") as f:
        for h in all_hands:
            f.write(h["log"] + "\n")

    print(f"  Pairs:     {pairs_file}")
    print(f"  Hands:     {hands_file}")
    print(f"  Decisions: {decisions_file}")
    print(f"  Log:       {log_file}")

    # Print hand logs to stdout unless quiet
    if not quiet:
        for h in all_hands:
            print(h["log"])

    # Summary
    total_hands = len(all_hands)
    wins = {}
    total_tokens = {}
    for h in all_hands:
        wins[h["winner"]] = wins.get(h["winner"], 0) + 1
        for name, (tin, tout) in h["tokens"].items():
            if name not in total_tokens:
                total_tokens[name] = [0, 0]
            total_tokens[name][0] += tin
            total_tokens[name][1] += tout

    net_a = sum(pr["net_bb"][player_a] for pr in pair_results)
    net_b = sum(pr["net_bb"][player_b] for pr in pair_results)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {player_a} vs {player_b}")
    print(f"{'='*60}")
    for name, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} hand(s) won")
    sign_a = "+" if net_a >= 0 else ""
    sign_b = "+" if net_b >= 0 else ""
    print(f"\n  {player_a}: {sign_a}{fmt(net_a)} BB")
    print(f"  {player_b}: {sign_b}{fmt(net_b)} BB")
    print(f"\n  Pairs: {len(pair_results)}  |  Total hands: {total_hands}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total_hands:.1f}s per hand)")
    print()
    for name, (tin, tout) in total_tokens.items():
        print(f"  {name}: {tin} tokens in, {tout} tokens out")

    # Integrity check: warn if a model has high forced_action rate
    forced_counts = {}
    total_counts = {}
    for dec in all_decisions:
        name = dec["actor"]
        total_counts[name] = total_counts.get(name, 0) + 1
        if dec.get("forced_action"):
            forced_counts[name] = forced_counts.get(name, 0) + 1

    for name in total_counts:
        forced = forced_counts.get(name, 0)
        total = total_counts[name]
        if forced > 0:
            pct = forced / total * 100
            marker = "!!!" if pct > 50 else "!"
            print(f"\n  {marker} WARNING: {name} had {forced}/{total} "
                  f"forced actions ({pct:.0f}%)")
            if pct > 50:
                print(f"     Results for this matchup are unreliable. "
                      f"Check max_tokens / model compatibility.")


# ── Model configs ──────────────────────────────────────────

GEMINI_FLASH = {"name": "Gemini-Flash", "model": "google/gemini-2.0-flash-001"}
GEMINI_25_LITE = {"name": "Gemini-2.5-Flash-Lite", "model": "google/gemini-2.5-flash-lite"}
QWEN_35_FLASH = {"name": "Qwen3.5-Flash", "model": "qwen/qwen3.5-flash-02-23"}
GPT5_NANO = {"name": "GPT5-Nano", "model": "openai/gpt-5-nano"}
QWEN3_235B = {"name": "Qwen3-235B", "model": "qwen/qwen3-235b-a22b-2507"}
DEEPSEEK_V31 = {"name": "DeepSeek-V3.1", "model": "deepseek/deepseek-chat-v3.1"}
GROK_41_FAST = {"name": "Grok-4.1-Fast", "model": "x-ai/grok-4.1-fast"}

MATCHUPS = [
    ([QWEN_35_FLASH, GEMINI_FLASH], 100),       # 100 pairs = 200 hands
    ([QWEN_35_FLASH, GEMINI_25_LITE], 100),      # 100 pairs = 200 hands
]


def make_pairwise(models: list[dict], num_pairs: int) -> list:
    """Generate all pairwise matchups from a list of models."""
    from itertools import combinations
    return [([a, b], num_pairs) for a, b in combinations(models, 2)]


def main():
    quiet = "--quiet" in sys.argv or "-q" in sys.argv

    # Cheap reasoning models — pairwise, 10 pairs each
    cheap_reasoning = [
        GPT5_NANO, QWEN3_235B, DEEPSEEK_V31, GROK_41_FAST,
        GEMINI_25_LITE, QWEN_35_FLASH,
    ]
    matchups = make_pairwise(cheap_reasoning, num_pairs=10)

    print("=" * 60)
    print("  PokerArena: LLMs Play Heads-Up No-Limit Hold'em")
    print(f"  {len(matchups)} matchups, {len(matchups) * 10} pairs, {len(matchups) * 20} hands")
    print("=" * 60)

    for player_cfgs, num_pairs in matchups:
        run_matchup(player_cfgs, num_pairs, quiet=quiet)

    print(f"\n{'='*60}")
    print("  ALL MATCHUPS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
