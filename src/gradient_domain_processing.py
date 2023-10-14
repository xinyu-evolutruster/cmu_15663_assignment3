import numpy as np
import skimage
from scipy import signal as signal
from matplotlib import pyplot as plt


def gradient_x(img):
    new_img = np.pad(img, ((1, 0), (0, 0)))
    return np.diff(new_img, axis=0)


def gradient_y(img):
    new_img = np.pad(img, ((0, 0), (1, 0)))
    return np.diff(new_img, axis=1)


def divergence(grad_x, grad_y):
    grad_xx = gradient_x(grad_x)
    grad_yy = gradient_y(grad_y)
    return grad_xx + grad_yy


def laplacian(img):
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])
    new_img = signal.convolve2d(
        img, kernel, mode='same', boundary='fill', fillvalue=0)
    return new_img


def create_boundary_mask(h, w):
    boundary_mask = np.ones((h, w)).astype(np.uint8)
    for i in range(h):
        boundary_mask[i, 0] = 0
        boundary_mask[i, w-1] = 0
    for i in range(w):
        boundary_mask[0, i] = 0
        boundary_mask[h-1, i] = 0

    return boundary_mask


def create_boundary_img(img, boundary_mask):
    I_boundary = np.zeros_like(img).astype(np.float32)
    I_boundary[~boundary_mask] = img[~boundary_mask]

    return I_boundary


def gradient_integration(grad_x, grad_y, boundary_mask, I_boundary, N=10000, eps=0.005):
    I_init = np.zeros_like(grad_x).astype(np.float32)

    I_star = boundary_mask * I_init + (1 - boundary_mask) * I_boundary
    r = boundary_mask * (divergence(grad_x, grad_y) - laplacian(I_star))
    d = r
    delta_new = (r * r).sum()
    n = 0

    while np.sqrt((r * r).sum()) > eps and n < N:
        q = laplacian(d)
        eta = delta_new / (d * q).sum()
        I_star = I_star + boundary_mask * (eta * d)
        r = boundary_mask * (r - eta * q)
        delta_old = delta_new
        delta_new = (r * r).sum()
        beta = delta_new / delta_old
        d = r + beta * d
        n = n + 1
        # print("n = {}, diff = {}".format(n, np.sqrt((r*r).sum())))

    return I_star


def differentiate_reintegrate(img):
    integrated_img = np.zeros_like(img).astype(np.float32)
    h, w = integrated_img.shape[:2]
    for c in range(0, 3):
        img_c = img[:, :, c]
        grad_x = gradient_x(img_c)
        grad_y = gradient_y(img_c)

        boundary_mask = create_boundary_mask(h, w)
        I_boundary = create_boundary_img(img_c, boundary_mask)

        integrated_img[:, :, c] = gradient_integration(
            grad_x, grad_y, boundary_mask, I_boundary)

    return integrated_img


def fuse_gradient_field(amb_img, flash_img, sigma=40, tau_s=0.9):

    h, w = amb_img.shape[:2]
    boundary_mask = create_boundary_mask(h, w)

    integrated_img = np.zeros_like(amb_img)

    for c in range(3):
        amb_img_c = amb_img[:, :, c]
        flash_img_c = flash_img[:, :, c]

        grad_amb_x = gradient_x(amb_img_c)
        grad_amb_y = gradient_y(amb_img_c)

        grad_flash_x = gradient_x(flash_img_c)
        grad_flash_y = gradient_y(flash_img_c)

        M = np.abs(grad_flash_x * grad_amb_x + grad_flash_y * grad_amb_y)
        flash_den = np.sqrt(grad_flash_x**2 + grad_flash_y**2)
        amb_den = np.sqrt(grad_amb_x**2 + grad_amb_y**2)
        M = M / (flash_den * amb_den)
        M[amb_den == 0] = 0.5
        M[flash_den == 0] = 0.5

        w_s = np.tanh(sigma * (flash_img_c - tau_s))

        # normalize to [0,1]
        w_s = (w_s - w_s.min()) / (w_s.max() - w_s.min())

        grad_star_x = w_s * grad_amb_x + \
            (1 - w_s) * (M * grad_flash_x + (1 - M) * grad_amb_x)
        grad_star_y = w_s * grad_amb_y + \
            (1 - w_s) * (M * grad_flash_y + (1 - M) * grad_amb_y)

        I_boundary = create_boundary_img(
            (amb_img_c + flash_img_c) / 2.0, boundary_mask)
        integrated_c = gradient_integration(
            grad_star_x, grad_star_y, boundary_mask, I_boundary, eps=0.001)
        integrated_img[:, :, c] = integrated_c

        print("finished fuse gradient for channel {}".format(c))

    return integrated_img


def main():
    img_path = '../data/museum/museum_ambient.png'
    flash_img_path = '../data/museum/museum_flash.png'
    img = skimage.io.imread(img_path)[:, :, :3] / 255
    flash_img = skimage.io.imread(flash_img_path)[:, :, :3] / 255

    print("img shape: ", img.shape)

    self_img_path = '../data/self/cha_book.jpg'
    self_flash_img_path = '../data/self/cha_book_flash.jpg'
    self_img = skimage.io.imread(self_img_path)[:, :, :3] / 255
    self_flash_img = skimage.io.imread(self_flash_img_path)[:, :, :3] / 255

    self_img = skimage.transform.resize(
        self_img, (self_img.shape[0] // 8, self_img.shape[1] // 8),
        anti_aliasing=True)
    self_flash_img = skimage.transform.resize(
        self_flash_img, (self_flash_img.shape[0] // 8, self_flash_img.shape[1] // 8), anti_aliasing=True)
    self_img = self_img.astype(np.float32)
    self_flash_img = self_flash_img.astype(np.float32)

    # integrated_img = differentiate_reintegrate(img)

    # integrated_img = fuse_gradient_field(img, flash_img, sigma=40, tau_s=0.55)
    # print(integrated_img.max())
    # print(integrated_img.min())
    # integrated_img = np.clip(integrated_img * 1.1, a_min=0, a_max=1)

    # plt.imshow(integrated_img)
    # plt.axis('off')
    # plt.savefig('./fused_integrated.png', dpi=200,
    #             bbox_inches='tight', pad_inches=0)

    tau_s = [0.3]
    sigmas = [10, 20, 40, 80]
    for tau in tau_s:
        for sigma in sigmas:
            integrated_img = fuse_gradient_field(
                img, flash_img, sigma=sigma, tau_s=tau)
            print(integrated_img.max())
            print(integrated_img.min())
            integrated_img = np.clip(integrated_img * 1.1, a_min=0, a_max=1)

            plt.imshow(integrated_img)
            plt.axis('off')
            # plt.show()
            plt.savefig('./integrated_{}_{}.png'.format(tau, sigma), dpi=200,
                        bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
