"""
Nickname lookup table for first-name normalization during corruption.

Exports
-------
NICKNAMES : Dict[str, List[str]]
    Canonical name → list of common nicknames / short forms.

NICKNAME_TO_CANONICAL : Dict[str, str]
    Nickname → canonical name (for reverse lookup / normalization).

get_nickname(name: str) -> str
    Return a random nickname if one exists, otherwise return the name unchanged.
"""

from __future__ import annotations

import random
from typing import Dict, List

# ── Primary table: canonical → [nicknames] ───────────────────────────────────

NICKNAMES: Dict[str, List[str]] = {
    # ── Male ──────────────────────────────────────────────────────────────────
    "James": ["Jim", "Jimmy", "Jamie"],
    "Robert": ["Rob", "Bob", "Bobby", "Robbie"],
    "William": ["Will", "Bill", "Billy", "Willy", "Liam"],
    "John": ["Johnny", "Jack", "Jon"],
    "Michael": ["Mike", "Mikey", "Mick"],
    "David": ["Dave", "Davey"],
    "Richard": ["Rich", "Dick", "Rick", "Ricky"],
    "Charles": ["Charlie", "Chuck", "Chas"],
    "Joseph": ["Joe", "Joey", "Jo"],
    "Thomas": ["Tom", "Tommy"],
    "Christopher": ["Chris", "Topher"],
    "Daniel": ["Dan", "Danny"],
    "Matthew": ["Matt", "Matty"],
    "Anthony": ["Tony", "Ant"],
    "Mark": ["Marc"],
    "Donald": ["Don", "Donnie"],
    "Steven": ["Steve", "Stevie"],
    "Paul": ["Pablo"],
    "Andrew": ["Andy", "Drew"],
    "Kenneth": ["Ken", "Kenny"],
    "Joshua": ["Josh"],
    "Kevin": ["Kev"],
    "Brian": ["Bri", "Bryan"],
    "George": ["Georgie", "Geo"],
    "Timothy": ["Tim", "Timmy"],
    "Ronald": ["Ron", "Ronnie"],
    "Edward": ["Ed", "Eddie", "Ted", "Ned"],
    "Jason": ["Jay"],
    "Jeffrey": ["Jeff"],
    "Ryan": ["Ry"],
    "Jacob": ["Jake"],
    "Gary": ["Gar"],
    "Nicholas": ["Nick", "Nico", "Nicky"],
    "Eric": ["Erik"],
    "Stephen": ["Steve", "Stevie"],
    "Jonathan": ["Jon", "Jonny"],
    "Larry": ["Lawrence"],
    "Scott": ["Scotty"],
    "Frank": ["Francis", "Frankie"],
    "Raymond": ["Ray"],
    "Patrick": ["Pat", "Paddy"],
    "Peter": ["Pete"],
    "Samuel": ["Sam", "Sammy"],
    "Benjamin": ["Ben", "Benny"],
    "Alexander": ["Alex", "Alec"],
    "Henry": ["Harry", "Hank"],
    "Albert": ["Al", "Bert"],
    "Frederick": ["Fred", "Freddie"],
    "Harold": ["Harry", "Hal"],
    "Walter": ["Walt", "Wally"],
    "Arthur": ["Art"],
    "Lawrence": ["Larry", "Laurie"],
    "Gerald": ["Jerry", "Gerry"],
    "Roger": ["Rog"],
    "Dennis": ["Den", "Denny"],

    # ── Female ────────────────────────────────────────────────────────────────
    "Mary": ["Marie", "Molly", "May", "Mamie"],
    "Patricia": ["Pat", "Patty", "Trish", "Tricia"],
    "Jennifer": ["Jen", "Jenny", "Jenn"],
    "Linda": ["Lyn", "Lindy"],
    "Barbara": ["Barb", "Babs", "Bea"],
    "Susan": ["Sue", "Susie", "Suzy"],
    "Jessica": ["Jess", "Jessie"],
    "Sarah": ["Sara", "Sally"],
    "Karen": ["Kari", "Kare"],
    "Lisa": ["Liz", "Lissy"],
    "Nancy": ["Nan", "Nannie"],
    "Betty": ["Beth", "Bette", "Bet"],
    "Margaret": ["Maggie", "Meg", "Peggy", "Marge"],
    "Sandra": ["Sandy", "Sandi"],
    "Ashley": ["Ash"],
    "Dorothy": ["Dot", "Dottie", "Dora"],
    "Kimberly": ["Kim"],
    "Emily": ["Em", "Emmie"],
    "Donna": ["Don"],
    "Michelle": ["Shelly", "Mich"],
    "Carol": ["Carrie", "Caro"],
    "Amanda": ["Mandy", "Manda"],
    "Melissa": ["Mel", "Missy"],
    "Deborah": ["Deb", "Debbie"],
    "Stephanie": ["Steph"],
    "Rebecca": ["Becca", "Becky", "Bex"],
    "Sharon": ["Shar"],
    "Laura": ["Laur", "Laurie"],
    "Cynthia": ["Cindy", "Cyndi"],
    "Kathleen": ["Kathy", "Kate", "Kath"],
    "Elizabeth": ["Liz", "Beth", "Eliza", "Bette", "Libby"],
    "Catherine": ["Cathy", "Kate", "Cat"],
    "Christine": ["Chris", "Christy"],
    "Angela": ["Angie"],
    "Helen": ["Nell", "Nellie"],
    "Diane": ["Di"],
    "Alice": ["Ali"],
    "Julie": ["Jules"],
    "Joyce": ["Joy"],
    "Virginia": ["Ginny", "Ginger"],
    "Victoria": ["Vicky", "Tori"],
    "Shirley": ["Shirl"],
    "Judy": ["Jude"],
    "Teresa": ["Terry", "Tess"],
    "Carolyn": ["Carol", "Lynn"],
    "Janet": ["Jan"],
    "Maria": ["Marie"],
    "Ruth": ["Ruthie"],
    "Evelyn": ["Evie", "Lynn"],
}

# ── Reverse table: nickname → canonical ───────────────────────────────────────

NICKNAME_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _nicks in NICKNAMES.items():
    for _nick in _nicks:
        # If a nickname maps to multiple canonicals, keep first encountered
        if _nick not in NICKNAME_TO_CANONICAL:
            NICKNAME_TO_CANONICAL[_nick] = _canonical


# ── Public helper ─────────────────────────────────────────────────────────────

def get_nickname(name: str, rng: random.Random | None = None) -> str:
    """
    Return a randomly chosen nickname for `name`, or `name` itself if none exists.

    Parameters
    ----------
    name : str
        The canonical (or any) first name.
    rng : random.Random | None
        Optional seeded RNG; falls back to the global random module.

    Returns
    -------
    str
        A nickname, or the original name if no mapping is found.
    """
    _rng = rng or random
    # Try exact match first
    if name in NICKNAMES and NICKNAMES[name]:
        return _rng.choice(NICKNAMES[name])
    # Try title-cased version
    titled = name.title()
    if titled in NICKNAMES and NICKNAMES[titled]:
        return _rng.choice(NICKNAMES[titled])
    return name
