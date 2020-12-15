import io

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from IPython import display

from lucent.utils import to_hwc


def show_array(img, figsize=(3, 3)):
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_axis_off()
    with plt.rc_context({"savefig.pad_inches": 0}):
        plt.show()


def show_arrays(imgs, figsize=(12, 5), nrows=1):
    n = len(imgs)
    ncols = n // nrows
    _, ax = plt.subplots(nrows, ncols, figsize=figsize)

    for i, img in enumerate(imgs):
        ax[i].imshow(img)
        ax[i].set_axis_off()

    plt.tight_layout()
    plt.show()


def show_tensors(imgs, **kwargs):
    show_array(to_hwc(make_grid(imgs)), **kwargs)


def show_pil_image(img):
    bio = io.BytesIO()
    img.save(bio, format="png")
    display.display(display.Image(bio.getvalue(), format="png", retina=True))
