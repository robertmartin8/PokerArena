"""LLM Player: calls OpenRouter to decide poker actions."""

import re
import time
import random
from openai import OpenAI

SYSTEM_PROMPT = """You are a poker player in a heads-up No-Limit Texas Hold'em game.

You will be told your hole cards, the community cards (if any), the pot size, your stack, your opponent's stack, and the full action history for the current hand.

You MUST respond with EXACTLY one line in this format:
ACTION: fold
ACTION: check
ACTION: call
ACTION: raise AMOUNT

Where AMOUNT is a number of chips to raise TO (total bet size, not raise by).

Rules:
- You can only check if there is no bet to you.
- If there is a bet, you must fold, call, or raise.
- Minimum raise is 2x the current bet (or all-in).
- Think briefly about your hand strength and position, then give your action.

Respond with ONLY the ACTION line. No explanation."""

MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds


class Decision:
    """Captures everything about a single LLM decision."""

    __slots__ = (
        "player_name", "street", "game_state", "raw_response",
        "reasoning", "action", "amount",
        "tokens_in", "tokens_out", "reasoning_tokens",
        "latency_ms", "error",
        "valid_first_try", "repair_attempted",
        "forced_action",
    )

    def __init__(self):
        self.player_name = ""
        self.street = ""
        self.game_state = ""
        self.raw_response = ""
        self.reasoning = None       # chain-of-thought if model provides it
        self.action = ""
        self.amount = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.reasoning_tokens = 0   # thinking tokens (separate from output)
        self.latency_ms = 0
        self.error = None
        self.valid_first_try = True
        self.repair_attempted = False
        self.forced_action = False


class LLMPlayer:
    def __init__(self, name: str, model: str, api_key: str):
        self.name = name
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.stack = 0
        self.hole_cards: list[str] = []
        self.current_bet = 0  # how much this player has put in this street
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.decisions: list[Decision] = []  # all decisions this hand
        self._consecutive_forced = 0  # track consecutive forced actions

    def _call_api(self, messages: list[dict], dec: Decision) -> str:
        """Make an API call with retries. Returns raw response text."""
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.7,
                )
                elapsed = int((time.time() - t0) * 1000)
                dec.latency_ms += elapsed

                usage = response.usage
                if usage:
                    dec.tokens_in += usage.prompt_tokens
                    dec.tokens_out += usage.completion_tokens
                    self.total_tokens_in += usage.prompt_tokens
                    self.total_tokens_out += usage.completion_tokens
                    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                        rt = getattr(usage.completion_tokens_details, "reasoning_tokens", 0)
                        dec.reasoning_tokens += rt or 0

                msg = response.choices[0].message
                reasoning = getattr(msg, "reasoning", None)
                if reasoning:
                    dec.reasoning = reasoning

                return msg.content.strip() if msg.content else ""

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    print(f"  [{self.name}] API error (attempt {attempt+1}/{MAX_RETRIES}), retrying in {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                dec.error = str(e)
                print(f"  [{self.name}] API error after {MAX_RETRIES} retries: {e}")
                return None

    def _track_forced(self, forced: bool):
        """Track consecutive forced actions and warn if model is broken."""
        if forced:
            self._consecutive_forced += 1
            if self._consecutive_forced == 3:
                print(f"\n  *** WARNING: {self.name} ({self.model}) has failed "
                      f"3 consecutive decisions — model may be incompatible ***\n")
        else:
            self._consecutive_forced = 0

    @staticmethod
    def _smart_default(game_state: str) -> str:
        """Return 'check' if nothing to call, else 'fold'."""
        if "To call: 0" in game_state:
            return "check"
        return "fold"

    def decide(self, game_state: str, street: str = "") -> tuple[str, float]:
        """Ask the LLM for an action. Returns (action, amount)."""
        dec = Decision()
        dec.player_name = self.name
        dec.street = street
        dec.game_state = game_state

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": game_state},
        ]

        raw = self._call_api(messages, dec)

        # API failure — smart default
        if raw is None:
            fallback = self._smart_default(game_state)
            dec.raw_response = fallback
            dec.action = fallback
            dec.forced_action = True
            self.decisions.append(dec)
            self._track_forced(forced=True)
            return (fallback, 0)

        dec.raw_response = raw
        action, amount = self._parse(raw)

        # If parse failed, give the model one chance with feedback
        if action is None:
            dec.valid_first_try = False
            print(f"  [{self.name}] Could not parse: '{raw}' — retrying with feedback")
            dec.repair_attempted = True
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                "I couldn't understand your response. "
                "Reply with ONLY one line in this format:\n"
                "ACTION: fold\n"
                "ACTION: check\n"
                "ACTION: call\n"
                "ACTION: raise AMOUNT"
            })
            retry_raw = self._call_api(messages, dec)
            if retry_raw is not None:
                dec.raw_response = raw + " | RETRY: " + retry_raw
                action, amount = self._parse(retry_raw)

        # Still couldn't parse — smart default
        if action is None:
            fallback = self._smart_default(game_state)
            print(f"  [{self.name}] Parse failed after retry — defaulting to {fallback}")
            dec.action = fallback
            dec.amount = 0
            dec.forced_action = True
            self.decisions.append(dec)
            self._track_forced(forced=True)
            return (fallback, 0)

        dec.action = action
        dec.amount = amount
        self.decisions.append(dec)
        self._track_forced(forced=False)
        return (action, amount)

    @staticmethod
    def _parse(raw: str) -> tuple[str | None, float]:
        """Parse LLM response into (action, amount). Returns (None, 0) on failure."""
        # Try strict format first: ACTION: raise 4.5
        match = re.search(r"ACTION\s*:?\s*(fold|check|call|raise)\s*([0-9]+(?:\.[0-9]+)?)?", raw, re.IGNORECASE)
        if match:
            return (match.group(1).lower(), float(match.group(2)) if match.group(2) else 0)

        # Try bare action word (last occurrence — models put the answer at the end)
        matches = list(re.finditer(r"\b(fold|check|call|raise)\s*(?:to\s*)?([0-9]+(?:\.[0-9]+)?)?", raw, re.IGNORECASE))
        if matches:
            m = matches[-1]
            return (m.group(1).lower(), float(m.group(2)) if m.group(2) else 0)

        return (None, 0)
