from PIL import Image
import glob

# Create the frames
frames = []
imgs = glob.glob("3d_plots_time*.png")
imgs.sort()
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('3dplots.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)
