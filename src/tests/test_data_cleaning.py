import pytest
from src.utils.utils import contains_numeric, contains_substring


def test_contains_numeric():
    assert contains_numeric("abc123") == True
    assert contains_numeric("abc") == False


def test_contains_substring():
    assert contains_substring("hello world", ["world"]) == True
    assert contains_substring("hello world", ["goodbye"]) == False
