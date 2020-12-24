from fastprogress.fastprogress import progress_bar


def compute_mean_std(loader):
    mean, std = 0.0, 0.0
    for images, _ in progress_bar(loader):
        n_batch = images.size(0)
        images = images.view(n_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std