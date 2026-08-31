"""
Bit-manipulation helpers for binary point arithmetic.

A binary point is represented as a Python int (bitmask):
  bit i of z  ==  value of x_i at point z
"""

from __future__ import annotations
from typing import Iterator

Point = int  # bitmask: bit i = value of x_i


def flip(z: Point, i: int) -> Point:
    """Return z with bit i flipped."""
    return z ^ (1 << i)


def flip2(z: Point, i: int, j: int) -> Point:
    """Return z with bits i and j flipped."""
    return z ^ (1 << i) ^ (1 << j)


def bit(i: int) -> int:
    """Return a bitmask with only bit i set."""
    return 1 << i


def popcount(z: int) -> int:
    """Return the number of set bits (Hamming weight)."""
    return bin(z).count("1")


def bits(mask: int) -> Iterator[int]:
    """Yield indices of set bits in mask, LSB first."""
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask &= mask - 1
