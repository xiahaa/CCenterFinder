import imageio
import glob
import os
folder = './output'
filenames = glob.glob(os.path.join(folder, '*.png'))
filenames.sort()
print(filenames)
# filenames = [folder + '/' + filename for filename in filenames]
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('./output/circle_cga.gif', images, mode='I', duration = 0.1)