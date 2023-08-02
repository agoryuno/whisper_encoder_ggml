import struct
from pathlib import Path
import os
import requests

import torch
import numpy as np

MODEL_URL = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
MEL_FILTERS_URL = "https://github.com/openai/whisper/raw/main/whisper/assets/mel_filters.npz"

FNAME_OUT = Path("./ggml-encoder.bin")


def download_file(url=MODEL_URL, filename="whisper.chkpt"):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)


def redownload_whisper(
                       chkpt_url=MODEL_URL, 
                       mel_url=MEL_FILTERS_URL,
                       outname="whisper.chkpt", 
                       mel_file="mel_filters.npz",
                       ):
    # Remove files
    os.remove(outname)
    os.remove(mel_file)

    download_file(chkpt_url, outname)
    download_file(mel_url, mel_file)


def load_data(chkpt_file="whisper.chkpt", mel_file="mel_filters.npz"):
    # Load the model checkpoint
    checkpoint = torch.load(chkpt_file, map_location="cpu")

    # Filter out params that don't pertain to the encoder
    enc_pars = ('n_mels', 'n_audio_ctx', 'n_audio_state',
            'n_audio_head', 'n_audio_layer')
    params = {p : checkpoint["dims"][p] for p in enc_pars}

    # Filter out blocks that don't pertain to the encoder
    encoder = {k:v for k,v in checkpoint["model_state_dict"].items()
               if k.startswith('encoder.') }

    # Load MEL filters
    with np.load(mel_file, allow_pickle=True) as f:
        filters = f[f'mel_{params["n_mels"]}']
    
    return encoder, filters, params

def write_ggml(encoder, filters, params, ggml_out_path=FNAME_OUT):
    fout = ggml_out_path.open("wb")

    fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex

    for v in params.values():
        fout.write(struct.pack("i", v))
    fout.write(struct.pack("i", True)) # signifies the use of float16

    # write mel filters
    fout.write(struct.pack("i", filters.shape[0]))
    fout.write(struct.pack("i", filters.shape[1]))
    for i in range(filters.shape[0]):
        for j in range(filters.shape[1]):
            fout.write(struct.pack("f", filters[i][j]))

    # write model blocks
    for name in encoder.keys():
        data = encoder[name].squeeze().numpy()
        print("Processing variable: " , name ,  " with shape: ", data.shape)

        # reshape conv bias from [n] to [n, 1]
        if name in ["encoder.conv1.bias", "encoder.conv2.bias"]:
            data = data.reshape(data.shape[0], 1)
            print(f"  Reshaped variable: {name} to shape: ", data.shape)

        n_dims = len(data.shape)

        # looks like the whisper models are in f16 by default
        # so we need to convert the small tensors to f32 until we fully support f16 in ggml
        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype = 1
        if n_dims < 2 or \
                name == "encoder.conv1.bias"   or \
                name == "encoder.conv2.bias"   or \
                name == "encoder.positional_embedding" or \
                name == "decoder.positional_embedding":
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype = 0

        str_ = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str_), ftype))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str_)

        # data
        data.tofile(fout)

    fout.close()