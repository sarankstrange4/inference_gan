from tqdm.notebook import tqdm
from os.path import exists, join, basename, splitext
import os
import time
import matplotlib
import matplotlib.pylab as plt
import gdown
import sys
sys.path.append('C:\\Users\\USER\\Desktop\\Tacotron_Env\\hifi-gan')
sys.path.append("C:\\Users\\USER\\Desktop\\Tacotron_Env\\tacotron2")


import numpy as np
import torch
import json
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser
import resampy
import scipy.signal


print("done")

def dict_setup():

  thisdict = {}
  for line in reversed((open('C:\\Users\\USER\\Desktop\\Tacotron_Env\\merged.dict.txt', "r").read()).splitlines()):
      thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()
  
  return thisdict


