import re

from typing import List, Dict 
from collections import Counter

digit_pattern = re.compile(r"\d+")
only_digit_pattern = re.compile(r"^\d+$")
footnote_pattern = re.compile(r"^([a-zA-ZßöüäÖÄÜ.,;:?!\"\'“]){2,}[0-9]+$")


def common_error_replacements(word: str) -> str:
    common_errors = {
        "o¨": "ö",
        "a¨": "ä",
        "u¨": "ü",
        "¨o": "ö",
        "¨a": "ä",
        "¨u": "ü",
    }
    for k, v in common_errors.items():
        word = word.replace(k, v)

    return word

def group(l: List, group_range: int, sorting_index: int) -> List:
    """Groups vertical and horizontal lines by specific group range. This is needed to combine several lines into one line.
    
    Args:
        l (list): list of vertical or horizontal lines
        group_range (int): offset in positive and negative direction to group lines which meet in range
        sorting_index (int): 0 -> vertical lines, 1 -> horizontal lines

    Returns:
        list of lines grouped together by range
    """
    groups = []
    this_group = []
    i = 0
    l = list(l)
    l = sorted(l, key=lambda lines: lines[sorting_index])
    while i < len(l):
        a = l[i]
        if len(this_group) == 0:
            if i == len(l) - 1:
                # add last group
                this_group.append(a)
                break
            this_group_start = a
        if (
            a[sorting_index] <= this_group_start[sorting_index] + group_range
            and a[sorting_index] >= this_group_start[sorting_index] - group_range
        ):
            this_group.append(a)
        if a[sorting_index] < this_group_start[sorting_index] + group_range:
            i += 1
        else:
            groups.append(this_group)
            this_group = []
    groups.append(this_group)
    return [list(g) for g in groups if len(g) != 0]



def group_words(words: List[Dict], margin=8, group_by="bottom") -> List[List[Dict]]:
    groups = []
    this_group = []
    i = 0
    while i < len(words):
        word = words[i]
        if len(this_group) == 0:
            if i == len(words) - 1:
                this_group.append(word)
                break
            group_start = word[group_by]
        if word[group_by] >= group_start - margin and word[group_by] <= group_start + margin:
            this_group.append(word)
        else:
            groups.append(this_group)
            this_group = []
            i -= 1
        
        i += 1
    groups.append(this_group)
    return [list(g) for g in groups if len(g) != 0]



def get_bottom_coordinate_from_line(line: List[Dict]) -> float:
    bottoms = [word["bottom"] for word in line]
    return max(Counter(bottoms))


def get_height_frequency(line: List[Dict]) -> float:
    heights = [word["height"] for word in line]
    height_frequencies = Counter(heights)
    return sorted(height_frequencies.items(), key=lambda x: x[1], reverse=True)


def has_digit(word: str) -> bool:
    return bool(digit_pattern.search(word))


def is_digit(word: str) -> bool:
    return bool(only_digit_pattern.search(word))


def mark_superscript(word: str, marker: str) -> str:
    if footnote_pattern.search(word):
        digit_group = digit_pattern.search(word)
        digit = digit_group.group()
        return word.replace(digit, f"{marker}{digit}{marker}")
    else:
        return word

def mark_footnote_start(line: str) -> str:
    line[0]["text"] = f"FOOTNOTE:{line[0]['text']}"
    # line[-1]["text"] = f"{line[-1]['text']}FOOTNOTEEND"
    return line

def mark_footnote_end(line: str) -> str:
    line[-1]["text"] = f"{line[-1]['text']}FOOTNOTEEND"
    return line

# def mark_footnote()
# def mark_index(word: str) -> str:
