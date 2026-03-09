"""Pure Python hand evaluator for Texas Hold'em (no external deps)."""

from collections import Counter
from collections.abc import Sequence
from itertools import combinations

RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANK_ORDER)}


def card_rank(card: str) -> int:
    return RANK_VALUE[card[0]]


def card_suit(card: str) -> str:
    return card[1]


# Hand category constants (higher = better)
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8
ROYAL_FLUSH = 9

HAND_NAMES = {
    HIGH_CARD: "High Card",
    ONE_PAIR: "One Pair",
    TWO_PAIR: "Two Pair",
    THREE_OF_A_KIND: "Three of a Kind",
    STRAIGHT: "Straight",
    FLUSH: "Flush",
    FULL_HOUSE: "Full House",
    FOUR_OF_A_KIND: "Four of a Kind",
    STRAIGHT_FLUSH: "Straight Flush",
    ROYAL_FLUSH: "Royal Flush",
}


def _evaluate_5(cards: Sequence[str]) -> tuple:
    """Evaluate exactly 5 cards. Returns a comparable tuple (category, *kickers)."""
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]

    is_flush = len(set(suits)) == 1

    # Check straight
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0

    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        # Ace-low straight (A-2-3-4-5)
        elif unique_ranks == [12, 3, 2, 1, 0]:
            is_straight = True
            straight_high = 3  # 5-high straight

    if is_straight and is_flush:
        if straight_high == 12:  # Ace-high straight flush
            return (ROYAL_FLUSH, straight_high)
        return (STRAIGHT_FLUSH, straight_high)

    # Count rank frequencies
    rank_counts = Counter(ranks)
    freq = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    counts = [f[1] for f in freq]
    ordered_ranks = [f[0] for f in freq]

    if counts == [4, 1]:
        return (FOUR_OF_A_KIND, ordered_ranks[0], ordered_ranks[1])

    if counts == [3, 2]:
        return (FULL_HOUSE, ordered_ranks[0], ordered_ranks[1])

    if is_flush:
        return (FLUSH, *ranks)

    if is_straight:
        return (STRAIGHT, straight_high)

    if counts == [3, 1, 1]:
        return (THREE_OF_A_KIND, ordered_ranks[0], ordered_ranks[1], ordered_ranks[2])

    if counts == [2, 2, 1]:
        pair_ranks = sorted([ordered_ranks[0], ordered_ranks[1]], reverse=True)
        return (TWO_PAIR, pair_ranks[0], pair_ranks[1], ordered_ranks[2])

    if counts == [2, 1, 1, 1]:
        kickers = sorted([ordered_ranks[1], ordered_ranks[2], ordered_ranks[3]], reverse=True)
        return (ONE_PAIR, ordered_ranks[0], *kickers)

    return (HIGH_CARD, *ranks)


def evaluate(cards: list[str]) -> tuple:
    """Evaluate the best 5-card hand from 5-7 cards. Returns (category, *kickers)."""
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")
    if len(cards) == 5:
        return _evaluate_5(cards)

    best = None
    for combo in combinations(cards, 5):
        score = _evaluate_5(combo)
        if best is None or score > best:
            best = score
    return best


def hand_name(score: tuple) -> str:
    """Return human-readable name for a hand score tuple."""
    return HAND_NAMES[score[0]]


def rank_to_str(rank_val: int) -> str:
    """Convert a rank value back to its character."""
    return RANK_ORDER[rank_val]
