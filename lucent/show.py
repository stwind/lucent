import io

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from IPython import display

from lucent.tensor import channels_last


def show_array(img, figsize=(3, 3)):
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_axis_off()
    with plt.rc_context({"savefig.pad_inches": 0}):
        plt.show()


def show_arrays(imgs, figsize=(12, 5), nrows=1):
    ncols = len(imgs) // nrows
    fig = plt.figure(figsize=figsize)

    for i, img in enumerate(imgs):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(img)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def show_tensors(imgs, **kwargs):
    show_array(channels_last(make_grid(imgs)), **kwargs)


def show_pil_image(img):
    bio = io.BytesIO()
    img.save(bio, format="png")
    display.display(display.Image(bio.getvalue(), format="png", retina=True))
