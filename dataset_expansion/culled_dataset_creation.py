import csv
from PIL import Image
from scipy import misc
import numpy as np



SPECIES_OF_INTEREST = ["Styrax japonica", "Prunus virginiana", "Prunus sargentii", "Cryptomeria japonica", "Aesculus pavi" ]
IMSIZE = 300
DATA_DIR = "mutated"


def noisy(noise_typ,image):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = image
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy


def generate_allfiles(file):
    k = 1
    n = 1
    with open(file,'rb') as tsvin:
        # l = tsvin.readline()
        # fieldnames = l.split("\t")
        # fieldnames.append(fieldnames[1].rstrip("\n"))

        allfiles = dict()
        lablesdict = set()
        out = None
        reader = csv.reader(tsvin.readlines(), delimiter='\t')
        for i, line in enumerate(reader):
            if i > 0:
                if line[-2] in SPECIES_OF_INTEREST:
                    im = misc.imresize(misc.imread(line[1]), (IMSIZE, IMSIZE, 3))

                    r = im[:,:,0].flatten()
                    g = im[:,:,1].flatten()
                    b = im[:,:,2].flatten()
                    label = [SPECIES_OF_INTEREST.index(line[-2])]
                    if out is None:
                        out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)

                        print out.shape
                    else:
                        out = np.concatenate((out, np.array(list(label) + list(r) + list(g) + list(b), np.uint8)), axis=0)


        np.random.shuffle(out)
        print out.shape
        splits = np.split(out, 4)
        for i, split in enumerate(splits[:-1]):
            split.tofile("{}/out{}.bin".format(DATA_DIR,str(i)))
        splits[-1].tofile("eval.bin")





if __name__ == "__main__":
    generate_allfiles("leafsnap-dataset-images.txt")

