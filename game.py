"""Game engine: deck, dealing, betting rounds, showdown for heads-up NLHE."""

import random
from evaluator import evaluate, hand_name


RANKS = "23456789TJQKA"
SUITS = "hdcs"
SMALL_BLIND = 0.5
BIG_BLIND = 1


def make_deck() -> list[str]:
    deck = [r + s for r in RANKS for s in SUITS]
    random.shuffle(deck)
    return deck


def format_cards(cards: list[str]) -> str:
    return " ".join(cards) if cards else "(none)"


def fmt(chips) -> str:
    """Format chip amounts: show as int when whole, float otherwise."""
    chips = round(chips, 2)
    return str(int(chips)) if chips == int(chips) else str(chips)


class HandState:
    """Runs a single hand of heads-up NLHE."""

    def __init__(self, players: list, dealer_idx: int, hand_num: int, deck=None):
        self.players = players  # [player0, player1]
        self.dealer_idx = dealer_idx  # dealer is also SB in heads-up
        self.bb_idx = 1 - dealer_idx
        self.hand_num = hand_num
        self.deck = list(deck) if deck is not None else make_deck()
        self.board: list[str] = []
        self.pot = 0
        self.hand_over = False
        self.winner = None
        self.log: list[str] = []
        self.action_history: list[dict] = []

    def _log(self, msg: str):
        self.log.append(msg)

    def run(self) -> dict:
        """Run the entire hand. Returns result dict."""
        dealer = self.players[self.dealer_idx]
        bb = self.players[self.bb_idx]

        self._log(f"\n{'='*60}")
        self._log(f"Hand #{self.hand_num}  |  Dealer: {dealer.name}  |  "
                  f"{self.players[0].name}: {fmt(self.players[0].stack)}  "
                  f"{self.players[1].name}: {fmt(self.players[1].stack)}")
        self._log(f"{'='*60}")

        # Deal hole cards
        for p in self.players:
            p.hole_cards = [self.deck.pop(), self.deck.pop()]
            p.current_bet = 0

        # Post blinds — dealer=SB, other=BB in heads-up
        self._post_blind(dealer, SMALL_BLIND, "SB")
        self._post_blind(bb, BIG_BLIND, "BB")

        # --- Preflop ---
        if not self._betting_round("Preflop", first_actor_idx=self.dealer_idx):
            return self._finish()

        # --- Flop ---
        self.deck.pop()  # burn
        self.board.extend([self.deck.pop() for _ in range(3)])
        self._log(f"\n--- Flop: {format_cards(self.board)} ---")
        if not self._betting_round("Flop", first_actor_idx=self.bb_idx):
            return self._finish()

        # --- Turn ---
        self.deck.pop()  # burn
        self.board.append(self.deck.pop())
        self._log(f"\n--- Turn: {format_cards(self.board)} ---")
        if not self._betting_round("Turn", first_actor_idx=self.bb_idx):
            return self._finish()

        # --- River ---
        self.deck.pop()  # burn
        self.board.append(self.deck.pop())
        self._log(f"\n--- River: {format_cards(self.board)} ---")
        if not self._betting_round("River", first_actor_idx=self.bb_idx):
            return self._finish()

        # --- Showdown ---
        return self._showdown()

    def _post_blind(self, player, amount, label: str):
        actual = min(amount, player.stack)
        player.stack -= actual
        player.current_bet = actual
        self.pot += actual
        self._log(f"  {player.name} posts {label}: {fmt(actual)}")
        self.action_history.append({
            "street": "Preflop",
            "actor": player.name,
            "action": label.lower(),
            "amount_put_in_bb": actual,
            "bet_to_bb": actual,
            "pot_after_bb": self.pot,
            "stack_after_bb": player.stack,
        })

    def _betting_round(self, street: str, first_actor_idx: int) -> bool:
        """Run a betting round. Returns True if hand continues, False if someone folded."""
        for p in self.players:
            p.current_bet = 0

        if street == "Preflop":
            self.players[self.dealer_idx].current_bet = SMALL_BLIND
            self.players[self.bb_idx].current_bet = BIG_BLIND

        current_bet = max(p.current_bet for p in self.players)
        actors = [first_actor_idx, 1 - first_actor_idx]
        last_raiser = None
        last_raise_size = BIG_BLIND
        action_count = 0
        max_actions = 10

        i = 0
        while i < max_actions:
            actor_idx = actors[i % 2]
            actor = self.players[actor_idx]
            opponent = self.players[1 - actor_idx]

            if actor.stack == 0:
                i += 1
                if i >= 2 and action_count >= 2:
                    break
                continue

            if action_count >= 2:
                if last_raiser == actor_idx:
                    break
                if last_raiser is None and actor.current_bet == current_bet:
                    break

            state = self._build_state(actor, opponent, street, current_bet)
            raw = actor.decide(state, street=street)
            action, amount = actor.parse_action(raw)

            action, amount, corrections = self._validate_action(
                actor, action, amount, current_bet, last_raise_size
            )

            old_current_bet_actor = actor.current_bet
            self._execute_action(actor, action, amount, current_bet, street, corrections)
            action_count += 1

            if action == "fold":
                self.hand_over = True
                self.winner = opponent
                return False

            if action == "raise":
                raise_delta = actor.current_bet - old_current_bet_actor
                if raise_delta >= last_raise_size:
                    # Full raise — reopens action
                    last_raise_size = raise_delta
                    last_raiser = actor_idx
                else:
                    # Short all-in — does NOT reopen for previous raiser
                    current_bet = actor.current_bet
                current_bet = actor.current_bet

            i += 1

        # Safety: if cap hit with unmatched bets, fold the owing player
        if i >= max_actions:
            p0 = self.players[actors[0]]
            p1 = self.players[actors[1]]
            if p0.current_bet != p1.current_bet:
                if p0.current_bet < p1.current_bet:
                    owing = p0
                    winner = p1
                else:
                    owing = p1
                    winner = p0
                self._log(f"  {owing.name}: FOLD (action cap reached with unmatched bets)")
                self.action_history.append({
                    "street": street,
                    "actor": owing.name,
                    "action": "fold",
                    "amount_put_in_bb": 0,
                    "bet_to_bb": owing.current_bet,
                    "pot_after_bb": self.pot,
                    "stack_after_bb": owing.stack,
                    "corrections": [{"from": "cap_reached", "to": "fold",
                                     "reason": "action cap hit with unmatched bets"}],
                })
                self.hand_over = True
                self.winner = winner
                return False

        return True

    def _build_state(self, actor, opponent, street: str, current_bet) -> str:
        to_call = current_bet - actor.current_bet
        lines = [
            f"Street: {street}",
            f"Your hole cards: {format_cards(actor.hole_cards)}",
        ]
        if self.board:
            lines.append(f"Board: {format_cards(self.board)}")
        lines.extend([
            f"Pot: {fmt(self.pot)}",
            f"Your stack: {fmt(actor.stack)}",
            f"Opponent stack: {fmt(opponent.stack)}",
            f"Current bet: {fmt(current_bet)}",
            f"To call: {fmt(to_call)}",
        ])
        if to_call == 0:
            lines.append("You can check or raise.")
        else:
            lines.append("You must fold, call, or raise.")

        # Action history
        if self.action_history:
            lines.append("Action history:")
            for a in self.action_history:
                if a["action"] in ("sb", "bb"):
                    lines.append(f"  {a['actor']} posts {a['action'].upper()}")
                else:
                    detail = f" to {fmt(a['bet_to_bb'])}" if a["action"] == "raise" else ""
                    lines.append(f"  [{a['street']}] {a['actor']}: {a['action']}{detail}")

        return "\n".join(lines)

    def _validate_action(self, actor, action: str, amount, current_bet,
                         last_raise_size) -> tuple:
        """Validate and fix an LLM action. Returns (action, amount, corrections)."""
        to_call = current_bet - actor.current_bet
        corrections = []

        if action == "check":
            if to_call > 0:
                self._log(f"  [{actor.name}] Can't check (owes {fmt(to_call)}), converting to call")
                corrections.append({"from": "check", "to": "call",
                                    "reason": f"cannot check with {fmt(to_call)} to call"})
                action = "call"

        if action == "call":
            if to_call == 0:
                self._log(f"  [{actor.name}] Nothing to call, converting to check")
                corrections.append({"from": "call", "to": "check",
                                    "reason": "nothing to call"})
                action = "check"
                amount = 0
            else:
                amount = min(to_call, actor.stack)

        if action == "raise":
            min_raise_to = current_bet + last_raise_size
            if amount < min_raise_to:
                if actor.stack + actor.current_bet >= min_raise_to:
                    corrections.append({"from": f"raise {fmt(amount)}",
                                        "to": f"raise {fmt(min_raise_to)}",
                                        "reason": f"below min raise of {fmt(min_raise_to)}"})
                    amount = min_raise_to
                else:
                    # Short all-in is OK
                    amount = actor.stack + actor.current_bet
            if amount >= actor.stack + actor.current_bet:
                amount = actor.stack + actor.current_bet
            if actor.stack <= to_call:
                self._log(f"  [{actor.name}] Not enough to raise, going all-in as call")
                corrections.append({"from": "raise", "to": "call",
                                    "reason": "not enough chips to raise"})
                action = "call"
                amount = actor.stack

        return (action, amount, corrections)

    def _execute_action(self, actor, action: str, amount, current_bet, street: str,
                        corrections: list):
        """Execute a validated action."""
        amount_put_in = 0
        if action == "fold":
            self._log(f"  {actor.name}: FOLD")
        elif action == "check":
            self._log(f"  {actor.name}: CHECK")
        elif action == "call":
            call_amount = min(current_bet - actor.current_bet, actor.stack)
            actor.stack -= call_amount
            actor.current_bet += call_amount
            self.pot += call_amount
            amount_put_in = call_amount
            self._log(f"  {actor.name}: CALL {fmt(call_amount)} (pot: {fmt(self.pot)})")
        elif action == "raise":
            raise_to = amount
            cost = raise_to - actor.current_bet
            cost = min(cost, actor.stack)
            actor.stack -= cost
            actor.current_bet += cost
            self.pot += cost
            amount_put_in = cost
            self._log(f"  {actor.name}: RAISE to {fmt(actor.current_bet)} (pot: {fmt(self.pot)})")

        entry = {
            "street": street,
            "actor": actor.name,
            "action": action,
            "amount_put_in_bb": amount_put_in,
            "bet_to_bb": actor.current_bet,
            "pot_after_bb": self.pot,
            "stack_after_bb": actor.stack,
        }
        if corrections:
            entry["corrections"] = corrections
        self.action_history.append(entry)

    def _showdown(self) -> dict:
        """Evaluate hands and determine winner."""
        scores = []
        for p in self.players:
            all_cards = p.hole_cards + self.board
            score = evaluate(all_cards)
            scores.append(score)

        if scores[0] > scores[1]:
            self.winner = self.players[0]
        elif scores[1] > scores[0]:
            self.winner = self.players[1]
        else:
            self.winner = None

        if self.winner:
            self.winner.stack += self.pot
        else:
            half = self.pot / 2
            self.players[0].stack += half
            self.players[1].stack += self.pot - half

        return self._finish()

    def _finish(self) -> dict:
        """Log hand summary and return result."""
        if self.winner and not self.hand_over:
            pass
        elif self.winner and self.hand_over:
            self.winner.stack += self.pot

        self._log(f"\n--- Hand Summary ---")
        if self.board:
            self._log(f"  Board: {format_cards(self.board)}")
        for p in self.players:
            self._log(f"  {p.name}: {format_cards(p.hole_cards)}")
        if self.winner:
            reason = "fold" if self.hand_over else hand_name(evaluate(self.winner.hole_cards + self.board))
            self._log(f"  >> {self.winner.name} wins pot of {fmt(self.pot)} ({reason})")
        else:
            self._log(f"  >> Split pot! ({fmt(self.pot)})")

        return {
            "hand_num": self.hand_num,
            "dealer_idx": self.dealer_idx,
            "winner": self.winner.name if self.winner else "Split",
            "pot": self.pot,
            "stacks": {p.name: p.stack for p in self.players},
            "log": "\n".join(self.log),
            "action_history": self.action_history,
            "hole_cards": {p.name: p.hole_cards for p in self.players},
            "board": list(self.board),
        }
