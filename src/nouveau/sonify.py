"""
Sonification: read a poem as a musical score.

MappingFn = (Line) -> list[Note]   same composable-factory pattern as ContextFn

Scales, roots, mappings, and tempos are all knobs. Plugging different mappings
into the same poem produces different music; plugging the same mapping into
different poems lets you listen for structure.

Musical score factories close the recursive loop back into generators.py:
pitch_entropy_scorer() and rest_density_scorer() return ScoreFactory callables
with the same interface as novelty_scorer() — so corpora can be ranked, filtered,
and threshold-switched by sonic properties.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from nouveau.generators import _get_sentiment_analyzer, count_syllables

if TYPE_CHECKING:
    from nouveau.poem import Line, Poem

Note = tuple[int | None, int]          # (midi pitch or None = rest, duration ticks)
MappingFn = Callable[["Line"], list[Note]]
ScoreFactory = Callable[["Poem"], Callable[[str], float]]

# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------

SCALES: dict[str, tuple[int, ...]] = {
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "whole_tone":       (0, 2, 4, 6, 8, 10),
    "chromatic":        tuple(range(12)),
    "blues":            (0, 3, 5, 6, 7, 10),
    "diminished":       (0, 2, 3, 5, 6, 8, 9, 11),
    "major":            (0, 2, 4, 5, 7, 9, 11),
    "minor":            (0, 2, 3, 5, 7, 8, 10),
}

_NOTE_NAMES = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

_VOWEL_DEGREE = {"a": 0, "e": 1, "i": 2, "o": 3, "u": 4, "y": 2}


def parse_note(name: str) -> int:
    """Parse a note name like 'A3', 'C#4', 'Bb2' into a MIDI pitch number."""
    name = name.strip()
    for n, offset in sorted(_NOTE_NAMES.items(), key=lambda x: -len(x[0])):
        if name.upper().startswith(n.upper()):
            octave_str = name[len(n):]
            octave = int(octave_str) if octave_str else 4
            return 12 * (octave + 1) + offset
    raise ValueError(f"cannot parse note name: {name!r}")


def scale_pitch(degree: int, root: int, scale: tuple[int, ...], octave_shift: int = 0) -> int:
    octave, idx = divmod(abs(degree), len(scale))
    semitones = scale[idx]
    return max(0, min(127, root + semitones + 12 * (octave + octave_shift)))


# ---------------------------------------------------------------------------
# Mapping factories
# ---------------------------------------------------------------------------

def make_vowel_mapping(
    root: int = 57,
    scale: str = "minor_pentatonic",
    beat: int = 240,
    sentiment_register: bool = True,
) -> MappingFn:
    """Vowel identity -> scale degree; syllables -> duration; mood -> register.

    The first vowel of each word picks the pitch. Words without vowels are
    rests. Empty lines are long rests. A minor pentatonic with sentiment
    register means: the loop poems become phase pieces locked to the same
    few pitches; the varied ones roam the octave.
    """
    s = SCALES[scale]
    analyzer = _get_sentiment_analyzer()

    def _map(line: "Line") -> list[Note]:
        text = line.text.strip()
        if not text:
            return [(None, beat * 8)]
        shift = 0
        if sentiment_register:
            c = analyzer.polarity_scores(text)["compound"]
            shift = 1 if c > 0.2 else (-1 if c < -0.2 else 0)
        notes: list[Note] = []
        for word in text.split():
            stripped = word.lower().strip(".,!?;:'\"()[]—–-…")
            vowels = [c for c in stripped if c in _VOWEL_DEGREE]
            dur = max(1, min(count_syllables(word), 4)) * beat
            if not vowels:
                notes.append((None, dur))
            else:
                degree = _VOWEL_DEGREE[vowels[0]] + len(vowels) - 1
                notes.append((scale_pitch(degree, root, s, shift), dur))
        return notes

    return _map


def make_sentiment_mapping(
    root: int = 57,
    scale: str = "minor_pentatonic",
    beat: int = 240,
    window: int = 3,
) -> MappingFn:
    """Per-word sentiment compound -> pitch height within scale.

    High positive -> high pitch; negative -> low. The melodic contour
    follows the emotional arc of the text word by word. Loops become
    flatlines; volatile text produces actual contour.
    """
    s = SCALES[scale]
    analyzer = _get_sentiment_analyzer()

    def _map(line: "Line") -> list[Note]:
        text = line.text.strip()
        if not text:
            return [(None, beat * 8)]
        words = text.split()
        notes: list[Note] = []
        for word in words:
            compound = analyzer.polarity_scores(word)["compound"]
            degree = int((compound + 1.0) / 2.0 * (len(s) * 2 - 1))
            dur = max(1, min(count_syllables(word), 4)) * beat
            notes.append((scale_pitch(degree, root, s), dur))
        return notes

    return _map


def make_length_mapping(
    root: int = 57,
    scale: str = "minor_pentatonic",
    beat: int = 240,
    max_chars: int = 12,
) -> MappingFn:
    """Word length (chars) -> pitch height; syllables -> duration.

    Short words are low; long compound words climb. The loops of 'Go'
    (2 chars) become a single droning low note; 'FLEXIGATORS' peaks.
    """
    s = SCALES[scale]

    def _map(line: "Line") -> list[Note]:
        text = line.text.strip()
        if not text:
            return [(None, beat * 8)]
        notes: list[Note] = []
        for word in text.split():
            stripped = word.strip(".,!?;:'\"()[]—–-…")
            chars = len(stripped)
            degree = int(chars / max_chars * (len(s) * 2 - 1))
            dur = max(1, min(count_syllables(word), 4)) * beat
            notes.append((scale_pitch(degree, root, s), dur))
        return notes

    return _map


def make_contour_mapping(
    root: int = 57,
    scale: str = "minor_pentatonic",
    beat: int = 240,
    direction: str = "arc",   # "arc" | "ascending" | "descending" | "wave"
) -> MappingFn:
    """Position in line -> pitch, shaping a melodic contour.

    'arc' peaks in the middle; 'ascending' climbs; 'descending' falls;
    'wave' oscillates. Repeated lines produce the same arc — loops become
    repeating melodic gestures rather than sustained pitches.
    """
    s = SCALES[scale]

    def _map(line: "Line") -> list[Note]:
        text = line.text.strip()
        if not text:
            return [(None, beat * 8)]
        words = text.split()
        n = len(words)
        notes: list[Note] = []
        for i, word in enumerate(words):
            t = i / max(n - 1, 1)
            if direction == "arc":
                pos = 1.0 - abs(2 * t - 1)
            elif direction == "ascending":
                pos = t
            elif direction == "descending":
                pos = 1.0 - t
            else:  # wave
                pos = (math.sin(2 * math.pi * t) + 1) / 2
            degree = int(pos * (len(s) * 2 - 1))
            dur = max(1, min(count_syllables(word), 4)) * beat
            notes.append((scale_pitch(degree, root, s), dur))
        return notes

    return _map


def make_interval_mapping(
    root: int = 57,
    scale: str = "minor_pentatonic",
    beat: int = 240,
) -> MappingFn:
    """Each word shifts pitch by the interval suggested by its vowel change.

    Rather than absolute pitch, tracks pitch *movement* — like reading the
    poem as a sequence of intervals rather than notes. Convergent text
    (loops) drifts to the octave limits; divergent text stays centered.
    """
    s = SCALES[scale]
    degree = [len(s)]  # mutable cell — pitch position accumulates across words

    def _map(line: "Line") -> list[Note]:
        text = line.text.strip()
        if not text:
            degree[0] = len(s)  # reset on silence
            return [(None, beat * 8)]
        notes: list[Note] = []
        for word in text.split():
            stripped = word.lower().strip(".,!?;:'\"()[]—–-…")
            vowels = [_VOWEL_DEGREE.get(c, 0) for c in stripped if c in _VOWEL_DEGREE]
            if vowels:
                delta = vowels[-1] - vowels[0]
                degree[0] = max(0, min(len(s) * 3, degree[0] + delta))
            dur = max(1, min(count_syllables(word), 4)) * beat
            notes.append((scale_pitch(degree[0], root, s), dur))
        return notes

    return _map


# Default named instances (zero-knob for CLI)
MAPPINGS: dict[str, MappingFn] = {
    "vowel":     make_vowel_mapping(),
    "sentiment": make_sentiment_mapping(),
    "length":    make_length_mapping(),
    "arc":       make_contour_mapping(direction="arc"),
    "ascending": make_contour_mapping(direction="ascending"),
    "descending":make_contour_mapping(direction="descending"),
    "wave":      make_contour_mapping(direction="wave"),
    "interval":  make_interval_mapping(),
}

def _contour_factory(direction: str):
    return lambda root=57, scale="minor_pentatonic", beat=240, **_: \
        make_contour_mapping(root=root, scale=scale, beat=beat, direction=direction)


MAPPING_FACTORIES = {
    "vowel":      make_vowel_mapping,
    "sentiment":  make_sentiment_mapping,
    "length":     make_length_mapping,
    "contour":    make_contour_mapping,
    "arc":        _contour_factory("arc"),
    "ascending":  _contour_factory("ascending"),
    "descending": _contour_factory("descending"),
    "wave":       _contour_factory("wave"),
    "interval":   make_interval_mapping,
}


# ---------------------------------------------------------------------------
# Feature extraction (feeds back into ScoreFactory)
# ---------------------------------------------------------------------------

def note_features(notes: list[Note]) -> dict[str, float]:
    """Extract musical features from a note list for scoring."""
    pitched = [p for p, _ in notes if p is not None]
    rests = [d for p, d in notes if p is None]
    total_dur = sum(d for _, d in notes) or 1

    if not pitched:
        return {
            "pitch_entropy": 0.0,
            "pitch_range": 0,
            "distinct_pitches": 0,
            "rest_ratio": 1.0,
            "mean_pitch": 0.0,
        }

    # Shannon entropy over pitch distribution
    counts: dict[int, int] = {}
    for p in pitched:
        counts[p] = counts.get(p, 0) + 1
    total = len(pitched)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0

    return {
        "pitch_entropy":    entropy / max_entropy if max_entropy else 0.0,
        "pitch_range":      max(pitched) - min(pitched),
        "distinct_pitches": len(counts),
        "rest_ratio":       sum(rests) / total_dur,
        "mean_pitch":       sum(pitched) / len(pitched),
    }


# ---------------------------------------------------------------------------
# Musical score factories (ScoreFactory-compatible, close the recursive loop)
# ---------------------------------------------------------------------------

def pitch_entropy_scorer(
    mapping: MappingFn | None = None,
    invert: bool = False,
) -> ScoreFactory:
    """Score a poem by the entropy of its sonification.

    Low entropy = repetitive loops (few distinct pitches); high = varied.
    By default lower cost = higher entropy (rewards variety).
    Set invert=True to reward repetition (lower cost = more loop-like).

    This closes the recursive loop: corpora can be ranked/filtered by
    how monotonous or melodically rich they sonify to.
    """
    m = mapping or MAPPINGS["vowel"]

    def make_score(poem: "Poem") -> Callable[[str], float]:
        def score(text: str) -> float:
            from nouveau.poem import Line
            notes = m(Line(author="score", text=text))
            features = note_features(notes)
            e = features["pitch_entropy"]
            return (1.0 - e) if not invert else e
        return score

    return make_score


def rest_density_scorer(target: float = 0.2, mapping: MappingFn | None = None) -> ScoreFactory:
    """Score a poem by how close its rest ratio is to a target.

    target=0.0 penalizes any silence; target=1.0 rewards total silence.
    target=0.2 selects poems with some breath — text that neither floods
    nor empties.
    """
    m = mapping or MAPPINGS["vowel"]

    def make_score(poem: "Poem") -> Callable[[str], float]:
        def score(text: str) -> float:
            from nouveau.poem import Line
            notes = m(Line(author="score", text=text))
            features = note_features(notes)
            return abs(features["rest_ratio"] - target)
        return score

    return make_score


def pitch_range_scorer(target: int = 12, mapping: MappingFn | None = None) -> ScoreFactory:
    """Score by distance from a target pitch range in semitones.

    target=12 = one octave. Selects poems that span a particular register
    width rather than staying on one note or leaping wildly.
    """
    m = mapping or MAPPINGS["vowel"]

    def make_score(poem: "Poem") -> Callable[[str], float]:
        def score(text: str) -> float:
            from nouveau.poem import Line
            notes = m(Line(author="score", text=text))
            features = note_features(notes)
            return abs(features["pitch_range"] - target) / 12.0
        return score

    return make_score


# ---------------------------------------------------------------------------
# MIDI writer
# ---------------------------------------------------------------------------

def _vlq(n: int) -> bytes:
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def write_midi(
    notes: list[Note],
    path: Path,
    tempo_bpm: int = 72,
    ticks_per_quarter: int = 480,
    velocity: int = 80,
    channel: int = 0,
) -> Path:
    track = bytearray()
    # tempo meta event
    us_per_beat = 60_000_000 // tempo_bpm
    track += b"\x00\xff\x51\x03" + struct.pack(">I", us_per_beat)[1:]

    pending = 0
    for pitch, duration in notes:
        if pitch is None:
            pending += duration
            continue
        p = max(0, min(127, pitch))
        ch = channel & 0x0F
        track += _vlq(pending) + bytes([0x90 | ch, p, velocity])
        track += _vlq(duration) + bytes([0x80 | ch, p, 0])
        pending = 0

    track += _vlq(pending) + b"\xff\x2f\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_quarter)
    chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    path.write_bytes(header + chunk)
    return path


def poem_to_notes(poem: "Poem", mapping: MappingFn) -> list[Note]:
    return [note for line in poem.lines for note in mapping(line)]


def sonify(
    poem: "Poem",
    path: Path,
    mapping: str | MappingFn = "vowel",
    scale: str = "minor_pentatonic",
    root: str = "A3",
    tempo_bpm: int = 72,
    beat: int = 240,
) -> Path:
    """Sonify a poem to a MIDI file. All knobs exposed.

    mapping: name in MAPPINGS or a MappingFn callable
    scale:   name in SCALES
    root:    note name ('A3', 'C#4', etc.)
    tempo_bpm, beat: timing
    """
    if isinstance(mapping, str):
        factory = MAPPING_FACTORIES.get(mapping)
        if factory is None:
            raise ValueError(f"unknown mapping {mapping!r}. choices: {list(MAPPING_FACTORIES)}")
        r = parse_note(root)
        m = factory(root=r, scale=scale, beat=beat)
    else:
        m = mapping

    notes = poem_to_notes(poem, m)
    return write_midi(notes, path, tempo_bpm=tempo_bpm)
