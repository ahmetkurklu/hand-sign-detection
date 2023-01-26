import Augmentor

<<<<<<< HEAD
p = Augmentor.Pipeline("images_rph/A/")
=======
p = Augmentor.Pipeline("image2/train/N/")
>>>>>>> 75105a3411ac30f5680d35b15c013b5ef7128a20
p.zoom(probability=0.3,min_factor=0.8,max_factor=1.5)

p.flip_top_bottom(probability=0.4)
p.random_brightness(probability=0.3,min_factor=0.3,max_factor=1.2)
p.random_distortion(probability=1,grid_width=4,grid_height=4,magnitude=8)

p.sample(300)