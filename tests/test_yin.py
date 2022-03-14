""" Yin Pitch Estimator Tests

License:
    MIT License
    Copyright © 2022 Brent M. Spell

"""

import hypothesis
import hypothesis.strategies as hypstrat
import numpy as np
from hypothesis.extra import numpy as hypnum

import torchyin

SAMPLE_RATES = [8000, 16000, 24000, 44100, 48000]
π = np.pi


@hypothesis.given(
    dtype=hypnum.floating_dtypes(sizes=[32, 64], endianness="="),
    batch=hypstrat.integers(min_value=0, max_value=3),
    time=hypstrat.integers(min_value=0, max_value=48000),
    sample_rate=hypstrat.sampled_from(SAMPLE_RATES),
)
def test_nonperiodics(dtype, batch, time, sample_rate):
    shape = [batch, time] if batch > 0 else [time]

    # silence should never have a pitch
    signal = np.zeros(shape, dtype=dtype)
    pitch = torchyin.estimate(signal, sample_rate=48000)
    assert (pitch == 0).all()

    # white noise should never have a pitch
    signal = np.random.normal(size=shape).astype(dtype)
    pitch = torchyin.estimate(signal, sample_rate=48000)
    assert (pitch == 0).all()


@hypothesis.given(
    dtype=hypnum.floating_dtypes(sizes=[32, 64], endianness="="),
    batch=hypstrat.integers(min_value=0, max_value=3),
    frequency=hypstrat.floats(min_value=20, max_value=20000),
    sample_rate=hypstrat.sampled_from(SAMPLE_RATES),
)
def test_periodics(dtype, batch, frequency, sample_rate):
    frequency = min(frequency, sample_rate // 4)
    time = int(2 * sample_rate // frequency)

    # generate a sinusoid
    signal = np.sin(2 * π * frequency / sample_rate * np.arange(time))
    if batch > 0:
        signal = np.repeat(signal[np.newaxis], batch, axis=0)

    # estimate the pitch of the signal
    pitch = torchyin.estimate(signal, sample_rate=sample_rate, threshold=0.5)
    assert batch == 0 or pitch.shape[0] == batch
    assert pitch.shape[-1] == 1
    assert not (pitch == 0).any()

    # compare log pitch for ballpark estimate
    expect = frequency
    actual = pitch
    assert np.allclose(np.log1p(expect), np.log1p(actual), rtol=1)


@hypothesis.given(
    frequency=hypstrat.floats(min_value=20, max_value=20000),
    sample_rate=hypstrat.sampled_from(SAMPLE_RATES),
)
def test_harmonics(frequency, sample_rate):
    frequency = min(frequency, sample_rate // 4)
    time = int(2 * sample_rate // frequency)

    # generate a sawtooth, all harmonics
    phase = np.cumsum(np.full([time], frequency / sample_rate)) % 1.0
    signal = 2 * phase - 1

    # estimate the pitch of the signal
    pitch = torchyin.estimate(signal, sample_rate=sample_rate, threshold=0.5)
    assert pitch.shape[-1] == 1
    assert not (pitch == 0).any()

    # compare log pitch for ballpark estimate
    expect = frequency
    actual = pitch
    assert np.allclose(np.log1p(expect), np.log1p(actual), rtol=1)


def test_piano():
    # a higher sample rate is needed for higher notes
    FS = 96000
    NOTES = 2 ** ((np.arange(88) - 48) / 12) * 440

    # generate a batch of the standard 88 piano key frequencies
    time = np.arange(2 * int(FS / 20))
    signal = np.vstack([np.sin(2 * π * f / FS * time) for f in NOTES])

    # compute the mae of the pitch estimate from the note frequencies
    pitch = torchyin.estimate(signal, sample_rate=FS).squeeze(-1).numpy()
    errors = np.abs(pitch[np.newaxis, :] - NOTES[:, np.newaxis])

    # verify that each note is closer to its correct value than to any other note
    expect = np.arange(len(NOTES))
    actual = errors.argmin(-1)
    assert (expect == actual).all()
