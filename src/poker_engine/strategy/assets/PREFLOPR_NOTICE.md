# PreflopR asset notice

`preflopr-explicit-rfi-ranges.json` contains the explicit 6-handed and
9-handed open-raise hand lists extracted from
[`bmorrow10/preflopR`](https://github.com/bmorrow10/preflopR), pinned at commit
`aed511d0451aea33a14f7e9204595fc2211f233f`.

The upstream repository is licensed under the MIT License. Copyright (c) 2026.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The asset is used only as a heuristic training fallback. The upstream author
states that these ranges are GTO-approximate and are not solver-derived.

PokerSense deliberately excludes upstream's generated 3–5 and 7–8 player
fallback mappings and the last-resort BB mapping. Those mappings do not
represent independently authored per-player-count strategies.
