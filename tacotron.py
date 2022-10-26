from setup import *
from scipy.io.wavfile import write
import numpy as np

pronounciation_dictionary=False
show_graphs=True

#model = get_Tactron2(model_path)
#hifigan = get_hifigan(model_path)




def ARPA(text, punctuation=r"!?,.;", EOS_Token=True):
    out = ''
    for word_ in text.split(" "):
        word=word_; end_chars = ''
        while any(elem in word for elem in punctuation) and len(word) > 1:
            if word[-1] in punctuation: end_chars = word[-1] + end_chars; word = word[:-1]
            else: break
        try:
            word_arpa = thisdict[word.upper()]
            word = "{" + str(word_arpa) + "}"
        except KeyError: pass
        out = (out + " " + word + end_chars).strip()
    if EOS_Token and out[-1] != ";": out += ";"
    return out

def has_MMI(STATE_DICT):
    return any(True for x in STATE_DICT.keys() if "mi." in x)
    
def get_Tactron2(model_path,device):
    # Load Tacotron2 and Config
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.max_decoder_steps = 3000 # Max Duration
    hparams.gate_threshold = 0.25 # Model must be 25% sure the clip is over before ending generation
    model = Tacotron2(hparams)
    print("loading tacotron from ", model_path, "...")
    state_dict = torch.load(model_path, map_location=torch.device(device))['state_dict']
    if has_MMI(state_dict):
        raise Exception("ERROR: This notebook does not currently support MMI models.")
    model.load_state_dict(state_dict)
    _ = model.eval()
    return model, hparams

def get_hifigan(model_path,name):

  # Download HiFi-GAN
  hifigan_pretrained_model = model_path
  #gdown.download(d+MODEL_ID, hifigan_pretrained_model, quiet=False)

  # Load HiFi-GAN
  hifi_gan_path = "C:\\Users\\USER\\Desktop\\Tacotron_Env\\hifi-gan"

  conf = "C:\\Users\\USER/Desktop\\Tacotron_Env\\hifi-gan\\" + name

  with open(conf) as f:
      json_config = json.loads(f.read())
  h = AttrDict(json_config)
  torch.manual_seed(h.seed)
  hifigan = Generator(h).to(torch.device("cpu"))
  state_dict_g = torch.load(hifigan_pretrained_model, map_location=torch.device("cpu"))
  hifigan.load_state_dict(state_dict_g["generator"])
  hifigan.eval()
  hifigan.remove_weight_norm()
  denoiser = Denoiser(hifigan, mode="normal")
  return hifigan, h, denoiser



hifigan, h, denoiser = get_hifigan("C:\\Users\\USER\\Desktop\\Tacotron_Env\\hifimodel_config_v1","config_v1.json")

hifigan_sr, h2, denoiser_sr = get_hifigan("C:\\Users\\USER\\Desktop\\Tacotron_Env\\hifimodel_config_32k","config_32k.json")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, hparams = get_Tactron2("C:\\Users\\USER\\Desktop\\Tacotron_Env\\mytacotron",device)

max_duration = 50
model.decoder.max_decoder_steps = max_duration * 80
stop_threshold = 0.5
model.decoder.gate_threshold = stop_threshold
superres_strength = 4.0


def end_to_end_infer(text, pronounciation_dictionary):
    for i in [x for x in text.split("\n") if len(x)]:
        if not pronounciation_dictionary:
            if i[-1] != ";": i=i+";" 
        else: i = ARPA(i)
        with torch.no_grad(): # save VRAM by not including gradients
            sequence = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
            y_g_hat = hifigan(mel_outputs_postnet.float())
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
            audio_denoised2 = denoiser(audio.view(1, -1), strength=35)[:, 0].cpu().numpy().reshape(-1)
            # Resample to 32k
            audio_denoised = audio_denoised.cpu().numpy().reshape(-1)

            normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
            audio_denoised = audio_denoised * normalize
            wave = resampy.resample(
                audio_denoised,
                h.sampling_rate,
                h2.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            # HiFi-GAN super-resolution
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device("cpu"))
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                h2.n_fft,
                h2.num_mels,
                h2.sampling_rate,
                h2.hop_size,
                h2.win_size,
                h2.fmin,
                h2.fmax,
            )
            y_g_hat2 = hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze()
            audio2 = audio2 * MAX_WAV_VALUE
            audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0]

            # High-pass filter, mixing and denormalizing
            audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
            b = scipy.signal.firwin(
                101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
            )
            y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
            y *= superres_strength
            y_out = y.astype(np.int16)
            y_padded = np.zeros(wave_out.shape)
            y_padded[: y_out.shape[0]] = y_out
            sr_mix = wave_out + y_padded
            sr_mix = sr_mix / normalize

            print("Original 22kz:")
            print("original audio: ", audio_denoised2)

            print("Super-Res 32kz:")
            print("sr_mix: ",sr_mix.astype(np.int16))

    return audio_denoised2, sr_mix


def main():
    text = "hello dear, wassup"
    audio, enhanced_audio  = end_to_end_infer(text, pronounciation_dictionary)
    sample_rate = 22050
    
    print(audio.shape)
    


    print("saving wave file")

    write("first_audio1.wav",sample_rate,audio.astype(np.int16))
    print("done")

if __name__ == "__main__":
    main()