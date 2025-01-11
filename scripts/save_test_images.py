import numpy as np
import os
from PIL import Image

#The same for all subjects
images = np.load('data/processed_data/nsd_test_stim.npy')

test_images_dir = 'data/spike_stimuli/test_images/'

if not os.path.exists(test_images_dir):
   os.makedirs(test_images_dir)
for i in range(len(images)):
    im = Image.fromarray(images[i].astype(np.uint8))
    im.save('{}/{}.png'.format(test_images_dir,i))


