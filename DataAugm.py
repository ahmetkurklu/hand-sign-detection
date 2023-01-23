import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def visualize(original, augmented,type):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title(f'{type} image')
    plt.imshow(augmented)
    plt.show()

image_path = "..\images_rph\A\A_1.png"
imagepil = Image.open(image_path)

image_string = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image_string,channels=3)
img = tf.expand_dims(image, 0)

flipped = tf.image.flip_left_right(image)
visualize(imagepil, imagepil,"flipped")

grayscaled = tf.image.rgb_to_grayscale(image)
visualize(image, tf.squeeze(grayscaled),"grayscaled")

saturated = tf.image.adjust_saturation(image, 3)
visualize(image, saturated,"saturated")

bright = tf.image.adjust_brightness(image, 0.4)
visualize(image, bright,"brightness")

rotated = tf.image.rot90(image)
visualize(image, rotated,"rotated")

cropped = tf.image.central_crop(image, central_fraction=0.5)
visualize(image,cropped,"cropped")

