from PIL import Image
import glob
import numpy as np

# Create the frames
imgs = glob.glob("plots/3d_plots_*.png")
#imgs = glob.glob("plots/bic_*.png")
frames = [0]*len(imgs)
for i in imgs:
    new_frame = Image.open(i)
    n = int(i[i.index('time')+4:-4])-1
    frames[n] = new_frame

# Save into a GIF file
frames[0].save('3d_pots.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=100)
