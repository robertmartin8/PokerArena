"""Test suite for PokerArena."""

import pytest
from evaluator import (
    evaluate, hand_name, _evaluate_5, card_rank, card_suit,
    HIGH_CARD, ONE_PAIR, TWO_PAIR, THREE_OF_A_KIND, STRAIGHT,
    FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH, ROYAL_FLUSH,
)
from game import HandState, make_deck, format_cards, fmt


# ── Mock player for game engine tests ────────────────────────

class MockPlayer:
    """Scriptable player that returns pre-set actions from decide()."""

    def __init__(self, name, actions=None):
        self.name = name
        self.stack = 100
        self.hole_cards = []
        self.current_bet = 0
        self.decisions = []
        self._actions = list(actions) if actions else []
        self._action_idx = 0

    def decide(self, game_state, street=""):
        if self._action_idx < len(self._actions):
            action = self._actions[self._action_idx]
            self._action_idx += 1
            return action
        return ("check", 0)


def run_hand_with_actions(p0_actions, p1_actions, deck=None, p0_stack=100, p1_stack=100):
    """Helper: run a hand with scripted actions and a fixed deck."""
    p0 = MockPlayer("Alice", p0_actions)
    p1 = MockPlayer("Bob", p1_actions)
    p0.stack = p0_stack
    p1.stack = p1_stack
    if deck is None:
        deck = make_deck()
    result = HandState([p0, p1], dealer_idx=0, hand_num=1, deck=list(deck)).run()
    return result, p0, p1


# ══════════════════════════════════════════════════════════════
#  evaluator.py tests
# ══════════════════════════════════════════════════════════════

class TestEvaluatorBasics:
    def test_card_rank(self):
        assert card_rank("2h") == 0
        assert card_rank("Ah") == 12
        assert card_rank("Ts") == 8

    def test_card_suit(self):
        assert card_suit("Ah") == "h"
        assert card_suit("Tc") == "c"


class TestHandCategories:
    def test_high_card(self):
        cards = ["2h", "5d", "8c", "Ts", "Ah"]
        score = _evaluate_5(cards)
        assert score[0] == HIGH_CARD

    def test_one_pair(self):
        cards = ["Ah", "Ad", "5c", "8s", "Td"]
        score = _evaluate_5(cards)
        assert score[0] == ONE_PAIR
        assert score[1] == 12  # aces

    def test_two_pair(self):
        cards = ["Ah", "Ad", "Kc", "Ks", "5d"]
        score = _evaluate_5(cards)
        assert score[0] == TWO_PAIR

    def test_three_of_a_kind(self):
        cards = ["7h", "7d", "7c", "Ks", "2d"]
        score = _evaluate_5(cards)
        assert score[0] == THREE_OF_A_KIND

    def test_straight(self):
        cards = ["5h", "6d", "7c", "8s", "9d"]
        score = _evaluate_5(cards)
        assert score[0] == STRAIGHT
        assert score[1] == 7  # 9-high (rank_value of 9 = 7)

    def test_ace_low_straight(self):
        cards = ["Ah", "2d", "3c", "4s", "5d"]
        score = _evaluate_5(cards)
        assert score[0] == STRAIGHT
        assert score[1] == 3  # 5-high

    def test_ace_high_straight(self):
        cards = ["Th", "Jd", "Qc", "Ks", "Ad"]
        score = _evaluate_5(cards)
        assert score[0] == STRAIGHT
        assert score[1] == 12  # ace-high

    def test_flush(self):
        cards = ["2h", "5h", "8h", "Th", "Ah"]
        score = _evaluate_5(cards)
        assert score[0] == FLUSH

    def test_full_house(self):
        cards = ["Ah", "Ad", "Ac", "Ks", "Kd"]
        score = _evaluate_5(cards)
        assert score[0] == FULL_HOUSE

    def test_four_of_a_kind(self):
        cards = ["Qh", "Qd", "Qc", "Qs", "2d"]
        score = _evaluate_5(cards)
        assert score[0] == FOUR_OF_A_KIND

    def test_straight_flush(self):
        cards = ["5h", "6h", "7h", "8h", "9h"]
        score = _evaluate_5(cards)
        assert score[0] == STRAIGHT_FLUSH

    def test_royal_flush(self):
        cards = ["Th", "Jh", "Qh", "Kh", "Ah"]
        score = _evaluate_5(cards)
        assert score[0] == ROYAL_FLUSH


class TestHandRanking:
    def test_flush_beats_straight(self):
        straight = _evaluate_5(["5h", "6d", "7c", "8s", "9d"])
        flush = _evaluate_5(["2h", "5h", "8h", "Th", "Ah"])
        assert flush > straight

    def test_pair_beats_high_card(self):
        high = _evaluate_5(["2h", "5d", "8c", "Ts", "Ah"])
        pair = _evaluate_5(["2h", "2d", "3c", "4s", "5d"])
        assert pair > high

    def test_higher_pair_wins(self):
        low_pair = _evaluate_5(["2h", "2d", "8c", "Ts", "Ah"])
        high_pair = _evaluate_5(["Kh", "Kd", "3c", "4s", "5d"])
        assert high_pair > low_pair

    def test_same_pair_kicker_decides(self):
        weak_kicker = _evaluate_5(["Ah", "Ad", "3c", "4s", "5d"])
        strong_kicker = _evaluate_5(["Ah", "Ad", "Kc", "Qs", "Jd"])
        assert strong_kicker > weak_kicker

    def test_identical_hands_are_equal(self):
        hand1 = _evaluate_5(["Ah", "Kd", "Qc", "Js", "9d"])
        hand2 = _evaluate_5(["As", "Kc", "Qh", "Jd", "9h"])
        assert hand1 == hand2


class TestEvaluateBestOf7:
    def test_best_of_7(self):
        # 7 cards where the best 5 form a flush
        cards = ["2h", "5h", "8h", "Th", "Ah", "Kd", "3c"]
        score = evaluate(cards)
        assert score[0] == FLUSH

    def test_best_of_6(self):
        cards = ["Ah", "Ad", "Ac", "Kh", "Kd", "2s"]
        score = evaluate(cards)
        assert score[0] == FULL_HOUSE

    def test_5_cards_directly(self):
        cards = ["Ah", "Kh", "Qh", "Jh", "Th"]
        assert evaluate(cards)[0] == ROYAL_FLUSH

    def test_error_on_too_few(self):
        with pytest.raises(ValueError):
            evaluate(["Ah", "Kh"])


class TestHandName:
    def test_all_names(self):
        assert hand_name((HIGH_CARD, 12, 10, 8, 5, 3)) == "High Card"
        assert hand_name((ROYAL_FLUSH, 12)) == "Royal Flush"
        assert hand_name((ONE_PAIR, 5, 12, 10, 8)) == "One Pair"
        assert hand_name((FLUSH, 12, 10, 8, 5, 3)) == "Flush"


# ══════════════════════════════════════════════════════════════
#  game.py utility tests
# ══════════════════════════════════════════════════════════════

class TestGameUtils:
    def test_make_deck_has_52_unique_cards(self):
        deck = make_deck()
        assert len(deck) == 52
        assert len(set(deck)) == 52

    def test_format_cards(self):
        assert format_cards(["Ah", "Kd"]) == "Ah Kd"
        assert format_cards([]) == "(none)"
        assert format_cards(["2c"]) == "2c"

    def test_fmt_integer(self):
        assert fmt(5) == "5"
        assert fmt(100) == "100"
        assert fmt(0) == "0"

    def test_fmt_float(self):
        assert fmt(0.5) == "0.5"
        assert fmt(2.5) == "2.5"

    def test_fmt_rounds(self):
        assert fmt(1.999) == "2"
        assert fmt(3.0) == "3"
        assert fmt(2.456) == "2.46"


# ══════════════════════════════════════════════════════════════
#  game.py engine tests
# ══════════════════════════════════════════════════════════════

class TestBlinds:
    def test_blinds_posted_correctly(self):
        """Dealer (p0) posts SB=0.5, BB (p1) posts BB=1."""
        result, p0, p1 = run_hand_with_actions(
            [("fold", 0)], []  # SB folds preflop
        )
        # SB (Alice) posted 0.5 then folded → Bob wins 1.5 pot
        assert result["winner"] == "Bob"
        assert result["pot"] == 1.5
        assert result["stacks"]["Alice"] == 99.5
        assert result["stacks"]["Bob"] == 100.5

    def test_partial_blind_small_stack(self):
        """Player with less than SB amount posts partial blind and is all-in."""
        result, p0, p1 = run_hand_with_actions(
            [], [],  # Alice is all-in, engine skips her turns
            p0_stack=0.3  # Alice can only post 0.3 of 0.5 SB
        )
        # Alice posted 0.3 (all she had), pot = 1.3 (0.3 + 1.0 BB)
        assert result["pot"] == 1.3
        # Stacks conserved: Alice started at 0.3, Bob at 100
        assert result["stacks"]["Alice"] + result["stacks"]["Bob"] == 100.3
        # First action history entry shows partial blind
        sb_entry = result["action_history"][0]
        assert sb_entry["action"] == "sb"
        assert sb_entry["amount_put_in_bb"] == 0.3
        assert sb_entry["stack_after_bb"] == 0


class TestPreflopAction:
    def test_preflop_call(self):
        """SB calls BB, both check to showdown."""
        result, _, _ = run_hand_with_actions(
            [("call", 0)],  # Alice (SB) calls the BB
            [],  # Bob checks through
        )
        # Pot = 2 (SB completes to 1 + BB already posted 1)
        assert result["pot"] == 2
        assert result["stacks"]["Alice"] + result["stacks"]["Bob"] == 200

    def test_preflop_raise_call(self):
        """SB raises, BB calls."""
        result, _, _ = run_hand_with_actions(
            [("raise", 4)],  # Alice raises to 4
            [("call", 0)],   # Bob calls
        )
        assert result["pot"] >= 8  # At least 4+4 preflop
        assert result["stacks"]["Alice"] + result["stacks"]["Bob"] == 200

    def test_preflop_fold_by_bb(self):
        """SB raises, BB folds."""
        result, _, _ = run_hand_with_actions(
            [("raise", 4)],  # Alice raises to 4
            [("fold", 0)],   # Bob folds
        )
        assert result["winner"] == "Alice"
        # Alice raised to 4 (put in 3.5 more), Bob folded (lost BB=1)
        assert result["stacks"]["Alice"] == 101
        assert result["stacks"]["Bob"] == 99


class TestActionValidation:
    def test_check_when_facing_bet_becomes_call(self):
        """If model checks when facing a bet, it's converted to call."""
        hs = HandState(
            [MockPlayer("A"), MockPlayer("B")],
            dealer_idx=0, hand_num=1, deck=make_deck()
        )
        p = hs.players[0]
        p.stack = 100
        p.current_bet = 0
        action, amount, corrections = hs._validate_action(p, "check", 0, 2, 1)
        assert action == "call"
        assert len(corrections) == 1
        assert corrections[0]["from"] == "check"
        assert corrections[0]["to"] == "call"

    def test_call_when_nothing_to_call_becomes_check(self):
        hs = HandState(
            [MockPlayer("A"), MockPlayer("B")],
            dealer_idx=0, hand_num=1, deck=make_deck()
        )
        p = hs.players[0]
        p.stack = 100
        p.current_bet = 2
        action, amount, corrections = hs._validate_action(p, "call", 0, 2, 1)
        assert action == "check"
        assert len(corrections) == 1

    def test_raise_below_min_bumped(self):
        """Raise below minimum gets bumped to min-raise."""
        hs = HandState(
            [MockPlayer("A"), MockPlayer("B")],
            dealer_idx=0, hand_num=1, deck=make_deck()
        )
        p = hs.players[0]
        p.stack = 100
        p.current_bet = 1  # BB already posted
        # Current bet = 2 (opponent raised to 2), last raise size = 1
        # Min raise to = 2 + 1 = 3
        action, amount, corrections = hs._validate_action(p, "raise", 2.5, 2, 1)
        assert action == "raise"
        assert amount == 3  # bumped to min-raise
        assert len(corrections) == 1

    def test_raise_with_insufficient_stack_becomes_call(self):
        """Player tries to raise but has less than the call amount."""
        hs = HandState(
            [MockPlayer("A"), MockPlayer("B")],
            dealer_idx=0, hand_num=1, deck=make_deck()
        )
        p = hs.players[0]
        p.stack = 1  # only 1 BB left
        p.current_bet = 0
        # Facing a bet of 5, needs 5 to call but only has 1
        action, amount, corrections = hs._validate_action(p, "raise", 10, 5, 2)
        assert action == "call"
        assert amount == 1


class TestFullHand:
    def test_fold_stacks_conserved(self):
        """When someone folds, stacks still sum to 200."""
        result, _, _ = run_hand_with_actions(
            [("raise", 10)],
            [("fold", 0)],
        )
        total = sum(result["stacks"].values())
        assert total == 200

    def test_showdown_stacks_conserved(self):
        """After showdown, stacks still sum to 200."""
        result, _, _ = run_hand_with_actions([], [])  # all checks
        total = sum(result["stacks"].values())
        assert total == 200

    def test_showdown_better_hand_wins(self):
        """Player with better hole cards wins at showdown."""
        # Build a deck where Alice gets AA, Bob gets 22, board is K-Q-J-T-9
        deck = [
            "Ah", "As",  # Alice's hole cards (popped first)
            "2h", "2s",  # Bob's hole cards
            "3c",        # burn before flop
            "Kd", "Qd", "Jd",  # flop
            "4c",        # burn before turn
            "Td",        # turn
            "5c",        # burn before river
            "9c",        # river
        ] + ["6c"] * 40  # padding
        deck.reverse()  # deck.pop() takes from end

        result, _, _ = run_hand_with_actions([], [], deck=deck)
        assert result["winner"] == "Alice"

    def test_split_pot(self):
        """Identical hands split the pot."""
        # Both players get the same effective hand via the board
        # Board: A-K-Q-J-T (broadway straight), hole cards irrelevant
        deck = [
            "2h", "3h",  # Alice's hole cards (irrelevant)
            "2d", "3d",  # Bob's hole cards (irrelevant)
            "4c",        # burn
            "Ad", "Kd", "Qd",  # flop
            "5c",        # burn
            "Jd",        # turn
            "6c",        # burn
            "Td",        # river → board is A-K-Q-J-T straight
        ] + ["7c"] * 40
        deck.reverse()

        result, _, _ = run_hand_with_actions([], [], deck=deck)
        assert result["winner"] == "Split"
        assert result["stacks"]["Alice"] == 100
        assert result["stacks"]["Bob"] == 100

    def test_result_dict_shape(self):
        """Result contains all expected keys."""
        result, _, _ = run_hand_with_actions([("fold", 0)], [])
        assert "hand_num" in result
        assert "dealer_idx" in result
        assert "winner" in result
        assert "pot" in result
        assert "stacks" in result
        assert "log" in result
        assert "action_history" in result
        assert "hole_cards" in result
        assert "board" in result


class TestActionHistory:
    def test_blinds_in_history(self):
        """Blind posts appear in action history."""
        result, _, _ = run_hand_with_actions([("fold", 0)], [])
        history = result["action_history"]
        assert history[0]["action"] == "sb"
        assert history[0]["actor"] == "Alice"
        assert history[1]["action"] == "bb"
        assert history[1]["actor"] == "Bob"

    def test_action_entry_fields(self):
        """Each action entry has required fields."""
        result, _, _ = run_hand_with_actions([("call", 0)], [])
        for entry in result["action_history"]:
            assert "street" in entry
            assert "actor" in entry
            assert "action" in entry
            assert "amount_put_in_bb" in entry
            assert "bet_to_bb" in entry
            assert "pot_after_bb" in entry
            assert "stack_after_bb" in entry

    def test_corrections_tracked(self):
        """Engine corrections appear in action history."""
        # Bob (BB) tries to check when facing a raise → should become call
        result, _, _ = run_hand_with_actions(
            [("raise", 4)],   # Alice raises
            [("check", 0)],   # Bob tries to check → corrected to call
        )
        # Find the corrected action
        corrected = [a for a in result["action_history"] if "corrections" in a]
        assert len(corrected) > 0


class TestBuildState:
    def test_state_contains_key_fields(self):
        """Built state string includes all required info."""
        hs = HandState(
            [MockPlayer("Alice"), MockPlayer("Bob")],
            dealer_idx=0, hand_num=1, deck=make_deck()
        )
        p0, p1 = hs.players
        p0.stack = 95
        p0.hole_cards = ["Ah", "Kh"]
        p0.current_bet = 0
        p1.stack = 90

        state = hs._build_state(p0, p1, "Flop", 0)
        assert "Street: Flop" in state
        assert "Your hole cards: Ah Kh" in state
        assert "Pot:" in state
        assert "Your stack: 95" in state
        assert "Opponent stack: 90" in state
        assert "To call: 0" in state
        assert "You can check or raise." in state

    def test_state_includes_action_history(self):
        """State includes cached action history lines."""
        hs = HandState(
            [MockPlayer("Alice"), MockPlayer("Bob")],
            dealer_idx=0, hand_num=1, deck=make_deck()
        )
        hs._history_lines.append("  Alice posts SB")
        hs._history_lines.append("  Bob posts BB")

        p0, p1 = hs.players
        p0.stack = 100
        p0.hole_cards = ["Ah", "Kh"]
        p0.current_bet = 0
        p1.stack = 100

        state = hs._build_state(p0, p1, "Preflop", 1)
        assert "Action history:" in state
        assert "Alice posts SB" in state
        assert "Bob posts BB" in state


# ══════════════════════════════════════════════════════════════
#  player.py tests
# ══════════════════════════════════════════════════════════════

from player import LLMPlayer


class TestParse:
    def test_strict_fold(self):
        assert LLMPlayer._parse("ACTION: fold") == ("fold", 0)

    def test_strict_check(self):
        assert LLMPlayer._parse("ACTION: check") == ("check", 0)

    def test_strict_call(self):
        assert LLMPlayer._parse("ACTION: call") == ("call", 0)

    def test_strict_raise(self):
        assert LLMPlayer._parse("ACTION: raise 4") == ("raise", 4.0)

    def test_strict_raise_decimal(self):
        assert LLMPlayer._parse("ACTION: raise 4.5") == ("raise", 4.5)

    def test_case_insensitive(self):
        assert LLMPlayer._parse("ACTION: CALL") == ("call", 0)
        assert LLMPlayer._parse("Action: Fold") == ("fold", 0)

    def test_bare_action(self):
        assert LLMPlayer._parse("I think I'll check here")[0] == "check"

    def test_bare_raise_with_to(self):
        action, amount = LLMPlayer._parse("I'll raise to 10")
        assert action == "raise"
        assert amount == 10

    def test_last_occurrence_wins(self):
        """If model says 'I could fold but I'll call', the last action wins."""
        action, _ = LLMPlayer._parse("I could fold but I'll call")
        assert action == "call"

    def test_garbage_returns_none(self):
        assert LLMPlayer._parse("I don't know what to do!") == (None, 0)
        assert LLMPlayer._parse("") == (None, 0)

    def test_no_colon_in_action(self):
        """ACTION without colon still parses."""
        assert LLMPlayer._parse("ACTION fold") == ("fold", 0)


class TestSmartDefault:
    def test_check_when_nothing_to_call(self):
        state = "Pot: 2\nTo call: 0\nYou can check or raise."
        assert LLMPlayer._smart_default(state) == "check"

    def test_fold_when_facing_bet(self):
        state = "Pot: 10\nTo call: 5\nYou must fold, call, or raise."
        assert LLMPlayer._smart_default(state) == "fold"


# ══════════════════════════════════════════════════════════════
#  main.py tests
# ══════════════════════════════════════════════════════════════

from main import make_pairwise


class TestMakePairwise:
    def test_correct_count(self):
        models = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        matchups = make_pairwise(models, 5)
        assert len(matchups) == 3  # C(3,2) = 3

    def test_pairs_are_unique(self):
        models = [{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}]
        matchups = make_pairwise(models, 10)
        assert len(matchups) == 6  # C(4,2) = 6
        pairs = set()
        for cfgs, _ in matchups:
            pair = frozenset(c["name"] for c in cfgs)
            assert pair not in pairs
            pairs.add(pair)

    def test_num_pairs_preserved(self):
        models = [{"name": "A"}, {"name": "B"}]
        matchups = make_pairwise(models, 42)
        assert matchups[0][1] == 42

    def test_two_models(self):
        models = [{"name": "A"}, {"name": "B"}]
        matchups = make_pairwise(models, 5)
        assert len(matchups) == 1

    def test_single_model(self):
        models = [{"name": "A"}]
        matchups = make_pairwise(models, 5)
        assert len(matchups) == 0
