"""Tests for msgflux.nn.modules.container module."""

import pytest
from collections import OrderedDict
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.container import Sequential, ModuleList, ModuleDict


class SimpleModule(Module):
    """Simple module for testing containers."""

    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def forward(self, x):
        return f"{x}-{self.value}"


class TestSequential:
    """Test suite for Sequential container."""

    def test_sequential_basic(self):
        """Test basic Sequential functionality."""
        seq = Sequential(
            SimpleModule("A"),
            SimpleModule("B"),
            SimpleModule("C")
        )

        result = seq("start")
        assert result == "start-A-B-C"

    def test_sequential_with_ordered_dict(self):
        """Test Sequential with OrderedDict."""
        seq = Sequential(OrderedDict([
            ("first", SimpleModule("X")),
            ("second", SimpleModule("Y")),
        ]))

        result = seq("begin")
        assert result == "begin-X-Y"

    def test_sequential_len(self):
        """Test Sequential length."""
        seq = Sequential(
            SimpleModule("A"),
            SimpleModule("B")
        )
        assert len(seq) == 2

    def test_sequential_getitem(self):
        """Test Sequential indexing."""
        seq = Sequential(
            SimpleModule("A"),
            SimpleModule("B"),
            SimpleModule("C")
        )

        assert seq[0].value == "A"
        assert seq[1].value == "B"
        assert seq[-1].value == "C"

    def test_sequential_slice(self):
        """Test Sequential slicing."""
        seq = Sequential(
            SimpleModule("A"),
            SimpleModule("B"),
            SimpleModule("C"),
            SimpleModule("D")
        )

        sliced = seq[1:3]
        assert isinstance(sliced, Sequential)
        assert len(sliced) == 2

    def test_sequential_setitem(self):
        """Test Sequential item assignment."""
        seq = Sequential(
            SimpleModule("A"),
            SimpleModule("B")
        )

        seq[1] = SimpleModule("X")
        assert seq[1].value == "X"

    def test_sequential_delitem(self):
        """Test Sequential item deletion."""
        seq = Sequential(
            SimpleModule("A"),
            SimpleModule("B"),
            SimpleModule("C")
        )

        del seq[1]
        assert len(seq) == 2
        assert seq[0].value == "A"
        assert seq[1].value == "C"

    def test_sequential_append(self):
        """Test Sequential append."""
        seq = Sequential(SimpleModule("A"))
        seq.append(SimpleModule("B"))

        assert len(seq) == 2
        result = seq("start")
        assert result == "start-A-B"

    def test_sequential_insert(self):
        """Test Sequential insert."""
        seq = Sequential(
            SimpleModule("A"),
            SimpleModule("C")
        )

        seq.insert(1, SimpleModule("B"))
        assert len(seq) == 3
        assert seq[1].value == "B"

    def test_sequential_extend(self):
        """Test Sequential extend."""
        seq1 = Sequential(SimpleModule("A"))
        seq2 = Sequential(SimpleModule("B"), SimpleModule("C"))

        seq1.extend(seq2)
        assert len(seq1) == 3

    def test_sequential_add(self):
        """Test Sequential addition."""
        seq1 = Sequential(SimpleModule("A"))
        seq2 = Sequential(SimpleModule("B"))

        seq3 = seq1 + seq2
        assert len(seq3) == 2
        assert isinstance(seq3, Sequential)

    def test_sequential_iadd(self):
        """Test Sequential in-place addition."""
        seq1 = Sequential(SimpleModule("A"))
        seq2 = Sequential(SimpleModule("B"))

        seq1 += seq2
        assert len(seq1) == 2

    def test_sequential_pop(self):
        """Test Sequential pop."""
        seq = Sequential(
            SimpleModule("A"),
            SimpleModule("B"),
            SimpleModule("C")
        )

        popped = seq.pop(1)
        assert popped.value == "B"
        assert len(seq) == 2

    def test_sequential_iter(self):
        """Test Sequential iteration."""
        seq = Sequential(
            SimpleModule("A"),
            SimpleModule("B")
        )

        values = [module.value for module in seq]
        assert values == ["A", "B"]


class TestModuleList:
    """Test suite for ModuleList container."""

    def test_modulelist_basic(self):
        """Test basic ModuleList functionality."""
        mlist = ModuleList([
            SimpleModule("A"),
            SimpleModule("B"),
            SimpleModule("C")
        ])

        assert len(mlist) == 3
        assert mlist[0].value == "A"

    def test_modulelist_empty(self):
        """Test empty ModuleList."""
        mlist = ModuleList()
        assert len(mlist) == 0

    def test_modulelist_getitem(self):
        """Test ModuleList indexing."""
        mlist = ModuleList([
            SimpleModule("A"),
            SimpleModule("B"),
            SimpleModule("C")
        ])

        assert mlist[0].value == "A"
        assert mlist[-1].value == "C"

    def test_modulelist_setitem(self):
        """Test ModuleList item assignment."""
        mlist = ModuleList([
            SimpleModule("A"),
            SimpleModule("B")
        ])

        mlist[1] = SimpleModule("X")
        assert mlist[1].value == "X"

    def test_modulelist_delitem(self):
        """Test ModuleList item deletion."""
        mlist = ModuleList([
            SimpleModule("A"),
            SimpleModule("B"),
            SimpleModule("C")
        ])

        del mlist[1]
        assert len(mlist) == 2
        assert mlist[1].value == "C"

    def test_modulelist_append(self):
        """Test ModuleList append."""
        mlist = ModuleList()
        mlist.append(SimpleModule("A"))
        mlist.append(SimpleModule("B"))

        assert len(mlist) == 2

    def test_modulelist_extend(self):
        """Test ModuleList extend."""
        mlist = ModuleList([SimpleModule("A")])
        mlist.extend([SimpleModule("B"), SimpleModule("C")])

        assert len(mlist) == 3

    def test_modulelist_iadd(self):
        """Test ModuleList in-place addition."""
        mlist = ModuleList([SimpleModule("A")])
        mlist += [SimpleModule("B"), SimpleModule("C")]

        assert len(mlist) == 3

    def test_modulelist_add(self):
        """Test ModuleList addition."""
        mlist1 = ModuleList([SimpleModule("A")])
        mlist2 = ModuleList([SimpleModule("B")])

        mlist3 = mlist1 + mlist2
        assert len(mlist3) == 2
        assert isinstance(mlist3, ModuleList)

    def test_modulelist_insert(self):
        """Test ModuleList insert."""
        mlist = ModuleList([
            SimpleModule("A"),
            SimpleModule("C")
        ])

        mlist.insert(1, SimpleModule("B"))
        assert len(mlist) == 3
        assert mlist[1].value == "B"

    def test_modulelist_pop(self):
        """Test ModuleList pop."""
        mlist = ModuleList([
            SimpleModule("A"),
            SimpleModule("B"),
            SimpleModule("C")
        ])

        popped = mlist.pop(1)
        assert popped.value == "B"
        assert len(mlist) == 2

    def test_modulelist_iter(self):
        """Test ModuleList iteration."""
        mlist = ModuleList([
            SimpleModule("A"),
            SimpleModule("B")
        ])

        values = [module.value for module in mlist]
        assert values == ["A", "B"]

    def test_modulelist_slice(self):
        """Test ModuleList slicing."""
        mlist = ModuleList([
            SimpleModule("A"),
            SimpleModule("B"),
            SimpleModule("C"),
            SimpleModule("D")
        ])

        sliced = mlist[1:3]
        assert isinstance(sliced, ModuleList)
        assert len(sliced) == 2


class TestModuleDict:
    """Test suite for ModuleDict container."""

    def test_moduledict_basic(self):
        """Test basic ModuleDict functionality."""
        mdict = ModuleDict({
            "first": SimpleModule("A"),
            "second": SimpleModule("B")
        })

        assert len(mdict) == 2
        assert mdict["first"].value == "A"

    def test_moduledict_empty(self):
        """Test empty ModuleDict."""
        mdict = ModuleDict()
        assert len(mdict) == 0

    def test_moduledict_getitem(self):
        """Test ModuleDict indexing."""
        mdict = ModuleDict({
            "first": SimpleModule("A"),
            "second": SimpleModule("B")
        })

        assert mdict["first"].value == "A"
        assert mdict["second"].value == "B"

    def test_moduledict_setitem(self):
        """Test ModuleDict item assignment."""
        mdict = ModuleDict()
        mdict["key1"] = SimpleModule("A")
        mdict["key2"] = SimpleModule("B")

        assert len(mdict) == 2
        assert mdict["key1"].value == "A"

    def test_moduledict_delitem(self):
        """Test ModuleDict item deletion."""
        mdict = ModuleDict({
            "first": SimpleModule("A"),
            "second": SimpleModule("B")
        })

        del mdict["first"]
        assert len(mdict) == 1
        assert "first" not in mdict

    def test_moduledict_contains(self):
        """Test ModuleDict contains."""
        mdict = ModuleDict({
            "first": SimpleModule("A")
        })

        assert "first" in mdict
        assert "second" not in mdict

    def test_moduledict_keys(self):
        """Test ModuleDict keys."""
        mdict = ModuleDict({
            "first": SimpleModule("A"),
            "second": SimpleModule("B")
        })

        keys = list(mdict.keys())
        assert "first" in keys
        assert "second" in keys

    def test_moduledict_values(self):
        """Test ModuleDict values."""
        mdict = ModuleDict({
            "first": SimpleModule("A"),
            "second": SimpleModule("B")
        })

        values = list(mdict.values())
        assert len(values) == 2

    def test_moduledict_items(self):
        """Test ModuleDict items."""
        mdict = ModuleDict({
            "first": SimpleModule("A"),
            "second": SimpleModule("B")
        })

        items = dict(mdict.items())
        assert "first" in items
        assert items["first"].value == "A"

    def test_moduledict_update(self):
        """Test ModuleDict update."""
        mdict = ModuleDict({"first": SimpleModule("A")})
        mdict.update({"second": SimpleModule("B")})

        assert len(mdict) == 2
        assert "second" in mdict

    def test_moduledict_pop(self):
        """Test ModuleDict pop."""
        mdict = ModuleDict({
            "first": SimpleModule("A"),
            "second": SimpleModule("B")
        })

        popped = mdict.pop("first")
        assert popped.value == "A"
        assert len(mdict) == 1
        assert "first" not in mdict

    def test_moduledict_clear(self):
        """Test ModuleDict clear."""
        mdict = ModuleDict({
            "first": SimpleModule("A"),
            "second": SimpleModule("B")
        })

        mdict.clear()
        assert len(mdict) == 0

    def test_moduledict_get(self):
        """Test ModuleDict get."""
        mdict = ModuleDict({"first": SimpleModule("A")})

        assert mdict.get("first").value == "A"
        assert mdict.get("nonexistent") is None
        assert mdict.get("nonexistent", "default") == "default"

    def test_moduledict_set(self):
        """Test ModuleDict set."""
        mdict = ModuleDict()
        mdict.set("key1", SimpleModule("A"))

        assert "key1" in mdict
        assert mdict["key1"].value == "A"

    def test_moduledict_iter(self):
        """Test ModuleDict iteration."""
        mdict = ModuleDict({
            "first": SimpleModule("A"),
            "second": SimpleModule("B")
        })

        keys = list(mdict)
        assert "first" in keys
        assert "second" in keys
