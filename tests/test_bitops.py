import pytest
from structural_fp.bitops import flip, flip2, bit, popcount, bits


class TestFlip:
    def test_flip_bit0(self):
        assert flip(0b101, 0) == 0b100  # 5 -> 4

    def test_flip_bit1(self):
        assert flip(0b101, 1) == 0b111  # 5 -> 7

    def test_flip_bit2(self):
        assert flip(0b000, 2) == 0b100  # 0 -> 4

    def test_flip_involution(self):
        # flipping twice returns original
        for z in range(8):
            for i in range(3):
                assert flip(flip(z, i), i) == z


class TestFlip2:
    def test_basic(self):
        assert flip2(0b101, 0, 1) == 0b110  # 5 -> 6
        assert flip2(0b000, 0, 2) == 0b101  # 0 -> 5

    def test_commutative(self):
        assert flip2(0b101, 0, 1) == flip2(0b101, 1, 0)

    def test_involution(self):
        for z in range(8):
            assert flip2(flip2(z, 0, 2), 0, 2) == z


class TestBit:
    def test_values(self):
        assert bit(0) == 1
        assert bit(1) == 2
        assert bit(2) == 4
        assert bit(3) == 8


class TestPopcount:
    def test_zero(self):
        assert popcount(0) == 0

    def test_powers_of_two(self):
        assert popcount(1) == 1
        assert popcount(2) == 1
        assert popcount(4) == 1

    def test_mixed(self):
        assert popcount(0b101) == 2
        assert popcount(0b111) == 3
        assert popcount(0b1111) == 4


class TestBitsIter:
    def test_zero(self):
        assert list(bits(0)) == []

    def test_single(self):
        assert list(bits(0b001)) == [0]
        assert list(bits(0b010)) == [1]
        assert list(bits(0b100)) == [2]

    def test_multiple(self):
        assert list(bits(0b101)) == [0, 2]
        assert list(bits(0b111)) == [0, 1, 2]

    def test_lsb_first(self):
        # 0b1010 = bits 1 and 3 set, LSB first
        assert list(bits(0b1010)) == [1, 3]

    def test_as_set(self):
        assert set(bits(0b101)) == {0, 2}
