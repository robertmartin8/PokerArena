# PokerArena

A framework for evaluating LLM decision-making by having models play heads-up No-Limit Texas Hold'em against each other. Each model receives game state as a text prompt and must return a structured action. Every decision is instrumented: token usage, latency, reasoning traces (when available), and parsed actions are captured per-hand for analysis.

## Motivation

Poker is a useful benchmark for LLM reasoning because it requires:

- **Incomplete information** — you see your cards but not your opponent's.
- **Sequential decision-making** — each street builds on prior actions.
- **Risk/reward tradeoffs** — bet sizing, pot odds, implied odds.
- **Opponent modeling** — interpreting bets as signals of hand strength.

Unlike chess or Go, poker has a stochastic element (the deal) and hidden state (opponent's hole cards), making it closer to real-world decision problems. By running thousands of independent hands, we can measure whether a model's strategy is profitable in aggregate, separate from variance.

## Architecture

```
main.py          Matchup runner — dispatches hands in parallel, collects results
game.py          Game engine — deck, blinds, betting rounds, showdown
player.py        LLM agent — API call, response parsing, retry logic, decision capture
evaluator.py     Hand evaluator — pure Python, no dependencies
models.py        OpenRouter model browser — fetch, filter, export to CSV
```

### Game engine (`game.py`)

Each hand is fully self-contained: fresh deck, fresh 100 BB stacks, no history between hands. This makes hands independent and parallelizable.

- **Blind structure**: SB = 0.5 BB, BB = 1 BB (all values in BB units).
- **Heads-up position rules**: Dealer posts SB and acts first preflop, second post-flop.
- **Streets**: Preflop, Flop (3 cards + burn), Turn (1 + burn), River (1 + burn).
- **Action history**: The full hand action history is included in every prompt, so models can see prior streets' actions (who raised preflop, etc.).
- **Min-raise tracking**: Proper last-raise-size tracking per street. Min-raise-to = current bet + last raise increment. Short all-ins don't reopen action for the previous raiser.
- **Action validation**: The engine corrects illegal LLM moves — checking when facing a bet becomes a call, raising below minimum is bumped to min-raise, etc. All corrections are tracked with `{"from", "to", "reason"}` and stored in the action history.
- **Action cap**: 10 actions per street. If the cap is hit with unmatched bets, the owing player is folded (rather than silently ending the street).
- **Output buffering**: All game events are appended to an internal log list (not stdout), making the engine thread-safe for parallel execution.

#### Action history fields

Each action in `action_history` contains:

| Field | Description |
|-------|-------------|
| `amount_put_in_bb` | Incremental chips put in on this action |
| `bet_to_bb` | Total bet level after the action (actor's current bet) |
| `pot_after_bb` | Pot size after the action |
| `stack_after_bb` | Actor's remaining stack |
| `corrections` | List of engine corrections applied (if any) |

### LLM agent (`player.py`)

Each model receives a system prompt constraining output format and a user prompt with the current game state:

```
Street: Flop
Your hole cards: Ah Kh
Board: Qh Jh 2c
Pot: 6
Your stack: 97
Opponent stack: 97
Current bet: 0
To call: 0
You can check or raise.
Action history:
  GPT-OSS-120B posts SB
  Qwen3-235B posts BB
  [Preflop] GPT-OSS-120B: raise to 3
  [Preflop] Qwen3-235B: call
```

The model must respond with `ACTION: fold|check|call|raise AMOUNT`. Raise amounts support decimals (e.g., `raise 2.5`). A regex extracts the action; parse failures trigger one retry with a corrective prompt in the same `ACTION:` format.

Key parameters: `max_tokens=2048` (reasoning models need headroom for thinking tokens), `temperature=0.7`.

#### Failure handling

- **API failure** (after 5 retries): defaults to `check` if nothing to call, else `fold`. Marked as `forced_action`.
- **Parse failure** (after 1 retry): same smart default. Marked as `forced_action`.
- **Consecutive failure warning**: After 3 consecutive forced actions, prints a loud warning that the model may be incompatible (e.g., reasoning models that exhaust their token budget on thinking and return empty content).
- **Post-matchup integrity check**: Reports forced action rate per model. Above 50%, results are flagged as unreliable.

#### Decision capture

Every API call produces a `Decision` object containing:

| Field | Description |
|-------|-------------|
| `street` | Which betting round (Preflop, Flop, Turn, River) |
| `game_state` | The full text prompt sent to the model |
| `raw_response` | The model's raw output text |
| `action` | Parsed action after validation (fold, check, call, raise) |
| `amount` | Raise amount if applicable (float) |
| `tokens_in` | Prompt tokens for this call |
| `tokens_out` | Completion tokens (includes reasoning tokens for thinking models) |
| `reasoning_tokens` | Chain-of-thought tokens, reported separately by thinking models |
| `latency_ms` | Wall-clock time for the API call |
| `reasoning` | Full reasoning trace text, if the model provides one |
| `error` | Error message if the call failed after all retries |
| `forced_action` | Whether this action was a fallback due to API/parse failure |

### Hand evaluator (`evaluator.py`)

Pure Python, no external libraries. Evaluates the best 5-card hand from 5-7 cards by brute-forcing all C(7,5) = 21 combinations. Returns a comparable tuple `(category, *kickers)` where category is an integer 0-9 (High Card through Royal Flush). Native Python tuple comparison handles all tiebreaking automatically.

### Model browser (`models.py`)

Fetches all available models from the OpenRouter API and supports filtering by cost, reasoning capability, and modality.

```bash
python models.py --reasoning --max-prompt 1.0    # cheap reasoning models (< $1/M input)
python models.py --reasoning --sort context       # reasoning models sorted by context window
python models.py --csv models.csv                 # export to CSV
```

Programmatic usage:

```python
from models import fetch_models, filter_models
models = fetch_models()
cheap_thinkers = filter_models(models, max_prompt_cost=1e-6, reasoning_only=True)
```

### Parallelism (`main.py`)

Hands are independent (fresh stacks each hand), so they run in parallel via `ThreadPoolExecutor`. Each hand creates its own player instances and OpenAI client. Game output is buffered to a log list (not stdout) to avoid interleaving. Results are sorted by hand number before output.

Matchups use **duplicate pairs**: two hands with the same deck but swapped seats, so each model gets the other's cards. This controls for card luck and isolates decision quality.

## Data outputs

All output files use append mode, so results accumulate across runs.

### `logs/{A}_vs_{B}_hands.jsonl` — one JSON object per hand

Primary analysis artifact. Contains action history with full bet tracking, hole cards, and board.

### `logs/{A}_vs_{B}_decisions.jsonl` — one JSON object per decision

Every LLM API call with prompt, response, reasoning trace, token counts, latency, and forced action flag.

### `logs/{A}_vs_{B}_pairs.jsonl` — one JSON object per duplicate pair

Net BB for each player across the two hands in the pair.

### `logs/{A}_vs_{B}.log` — human-readable hand histories

Full text log of every hand: blinds posted, actions taken, board cards dealt, showdown results.

## Models tested

Cheap reasoning-capable models via OpenRouter (all < $0.20/M input tokens):

| Model | ID | $/M in | $/M out | Notes |
|-------|----|--------|---------|-------|
| GPT-OSS-120B | `openai/gpt-oss-120b` | $0.04 | $0.19 | Works well, ~11s/hand |
| Qwen3-235B | `qwen/qwen3-235b-a22b-2507` | $0.07 | $0.10 | Cheapest output, very concise responses |
| Qwen3.5-Flash | `qwen/qwen3.5-flash-02-23` | $0.10 | $0.40 | Detailed reasoning traces |
| Gemini-2.5-Flash-Lite | `google/gemini-2.5-flash-lite` | $0.10 | $0.40 | Fast, aggressive play style |
| DeepSeek-V3.1 | `deepseek/deepseek-chat-v3.1` | $0.15 | $0.75 | |
| Grok-4.1-Fast | `x-ai/grok-4.1-fast` | $0.20 | $0.50 | 2M context window |

**Not recommended**: `openai/gpt-5-nano` — returns empty strings (all tokens consumed by reasoning), 330s/hand, 100% forced action rate.

## Running

```bash
pip install openai python-dotenv
echo "OPENROUTER_API_KEY=sk-or-..." > .env
python main.py           # full output
python main.py --quiet   # summary only
```

Edit `MATCHUPS` in `main.py` or define model dicts and call `run_matchup()` directly to configure matchups.

## Known limitations

- **No opponent modeling across hands.** Hands are fully independent. The model can't learn that an opponent bluffs frequently or always folds to 3-bets.
- **Action validation hides errors.** Illegal moves are corrected (now tracked in `corrections`), which inflates the apparent competence of weaker models.
- **Single system prompt.** All models receive the same prompt. Performance is partly a function of prompt engineering.
- **Variance.** Poker has high variance even between well-played strategies. Distinguishing a 0.5 BB/hand edge from zero requires tens of thousands of hands.
