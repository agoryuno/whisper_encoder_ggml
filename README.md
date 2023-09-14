# whisper_encoder_ggml

This is a stripped down version of [whisper.cpp](https://github.com/ggerganov/whisper.cpp) that only includes the encoder.

To build execute `./build`.

To run the executable (first see model file prep instructions below) do: `encoder-cli model-file.pth audio-file.wav`

The `encoder-cli` executable returns a JSON-formatted string to stdout.

Note that the encoder will ignore audio files that are less than 1 second in duration.

## Preparing the model file

This code also needs a stripped down Whisper checkpoint file with the decoder part removed and converted to the GGML format.
To do this, you can use this Google Colab notebook: https://colab.research.google.com/drive/1NO_J8-7ZwziBvKsLOr6B9lI32ewRkxOU?usp=sharing
