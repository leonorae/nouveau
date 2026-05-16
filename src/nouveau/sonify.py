"""
Sonification: read a poem as a musical score.

A poem is already a structure of repetition, length, and sentiment. This
module maps that structure to pitch and rhythm and emits a Standard MIDI
File — no dependencies, stdlib only. Looping poems become minimalist
phase pieces; empty lines (silence is valid output here) become rests.

Mapping (first pass — every rule here is a knob):
  word        -> note
  vowel       -> scale degree (a/e/i/o/u over a minor pentatonic)
  syllables   -> note duration
  line mood   -> register shift (VADER compound: bright up, dark down)
  empty line  -> a held rest
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from nouveau.generators import _get_sentiment_analyzer, count_syllables

if TYPE_CHECKING:
    from nouveau.poem import Poem

# A minor pentatonic, root = A3 (MIDI 57). The scale that makes anything
# sound intentional — a forgiving net under a model that cannot play.
_ROOT = 57
_SCALE = (0, 3, 5, 7, 10)
_VOWEL_DEGREE = {"a": 0, "e": 1, "i": 2, "o": 3, "u": 4, "y": 2}

Note = tuple[int | None, int]  # (midi pitch or None for rest, duration in ticks)


def _vlq(n: int) -> bytes:
    """Encode an int as a MIDI variable-length quantity."""
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def _scale_pitch(degree: int, octave_shift: int = 0) -> int:
    """Map a scale-degree index onto a MIDI pitch, wrapping octaves."""
    octave, idx = divmod(degree, len(_SCALE))
    return _ROOT + _SCALE[idx] + 12 * (octave + octave_shift)


def poem_to_notes(poem: "Poem", beat: int = 240) -> list[Note]:
    """Translate a poem into a flat note list.

    beat is ticks per eighth note (default division is 480 ticks/quarter).
    """
    analyzer = _get_sentiment_analyzer()
    notes: list[Note] = []

    for line in poem.lines:
        text = line.text.strip()
        if not text:
            notes.append((None, beat * 4))  # silence becomes a whole rest
            continue

        compound = analyzer.polarity_scores(text)["compound"]
        if compound > 0.2:
            octave_shift = 1
        elif compound < -0.2:
            octave_shift = -1
        else:
            octave_shift = 0

        for word in text.split():
            stripped = word.lower().strip(".,!?;:'\"()[]—–-")
            vowels = [c for c in stripped if c in _VOWEL_DEGREE]
            duration = max(1, min(count_syllables(word), 4)) * beat
            if not vowels:
                notes.append((None, duration))  # wordless sound -> rest
                continue
            degree = _VOWEL_DEGREE[vowels[0]] + len(vowels) - 1
            notes.append((_scale_pitch(degree, octave_shift), duration))

    return notes


def write_midi(notes: list[Note], path: Path, tempo_bpm: int = 72) -> Path:
    """Write a flat note list to a single-track Standard MIDI File."""
    track = bytearray()
    track += b"\x00\xff\x51\x03" + struct.pack(">I", 60_000_000 // tempo_bpm)[1:]

    pending = 0  # accumulated rest time, applied as delta before the next note
    for pitch, duration in notes:
        if pitch is None:
            pending += duration
            continue
        p = max(0, min(127, pitch))
        track += _vlq(pending) + bytes([0x90, p, 80])
        track += _vlq(duration) + bytes([0x80, p, 0])
        pending = 0

    track += _vlq(pending) + b"\xff\x2f\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
    chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    path.write_bytes(header + chunk)
    return path


def sonify(poem: "Poem", path: Path, tempo_bpm: int = 72) -> Path:
    """Read a poem as music and write it to a MIDI file."""
    return write_midi(poem_to_notes(poem), path, tempo_bpm=tempo_bpm)
