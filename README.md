# CHORUS

CHORUS Harmonizes Outputs from Reconstructed Upmixed Signals.

v0 is an offline Python reference demo. It reads stereo audio and writes three stereo stems:

- `center.wav`: `[Lc, Rc]`
- `only.wav`: `[Lo, Ro]`
- `surround.wav`: `[Ls, Rs]`

The reference transform is STFT. FrFT and wavelet adapters are experimental.
