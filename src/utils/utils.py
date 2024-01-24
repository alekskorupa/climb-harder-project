from typing import List


def contains_numeric(value: str) -> bool:
    return any(char.isdigit() for char in value)


def contains_substring(input_string: str, substring_list: List[str]) -> bool:
    return any(substring in input_string for substring in substring_list)
