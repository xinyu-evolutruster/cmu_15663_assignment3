import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpn as interpn
import cv2
import skimage


def g_intensity(img, sigma):
    return np.exp(-img**2 / (2 * sigma**2))


def bilateral_filter(img, flash_img=None, kernel_size=3, sigma_r=0.05, sigma_g=40, lamb=0.01):
    filtered_img = np.zeros_like(img).astype(np.float64)
    h, w = filtered_img.shape[:2]

    # process the channels separately
    for c in range(3):

        img_channel = img[:, :, c]

        flash_img_channel = flash_img[:, :, c]
        flash_maxI = flash_img_channel.max() + lamb
        flash_minI = flash_img_channel.min() - lamb

        nb_segments = int(np.ceil((flash_maxI - flash_minI) / sigma_r))

        r = (flash_maxI - flash_minI) / nb_segments

        i_j = 0.0
        J_js = []
        z = []

        for j in range(nb_segments + 1):

            i_j = flash_minI + j * r
            z.append(flash_minI + j * r)
            G_j = g_intensity(flash_img_channel - i_j, sigma_r)

            K_j = cv2.GaussianBlur(G_j, (kernel_size, kernel_size), sigma_g)
            H_j = G_j * img_channel

            H_j = cv2.GaussianBlur(H_j, (kernel_size, kernel_size), sigma_g)

            J_j = H_j / K_j
            J_j[K_j == 0] = 1
            J_js.append(J_j)

        J_js = np.transpose(np.array(J_js), (1, 2, 0))

        x = np.arange(0, h)
        y = np.arange(0, w)
        z = np.array(z)

        xi = []
        for i in range(h):
            for j in range(w):
                xi.append((i, j, img_channel[i, j]))
        xi = np.array(xi).reshape(h, w, 3)

        filtered_img[:, :, c] = interpn(
            (x, y, z), values=J_js, xi=xi, method='linear')

    return filtered_img


def detail_transfer(amb_img, flash_img, sigma_r=0.05, eps=0.001):
    flash_img_base = bilateral_filter(
        flash_img, flash_img, kernel_size=5, sigma_r=sigma_r)

    amb_img_nr = bilateral_filter(
        amb_img, flash_img, kernel_size=5, sigma_r=sigma_r)

    return amb_img_nr * (flash_img + eps) / (flash_img_base + eps)


def linearize_img(img):
    mask = img <= 0.0404482
    new_img = np.zeros_like(img)
    new_img[mask] = img[mask] / 12.92
    new_img[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4

    return new_img


def morph_operations(img, opening_kernel, closing_kernel, dilation_kernel):
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, opening_kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, closing_kernel)
    return cv2.dilate(img, dilation_kernel)


def specularity_masking(amb_img, flash_img):
    amb_img_lin = linearize_img(amb_img)
    flash_img_lin = linearize_img(flash_img)

    # iso correction
    amb_img_lin = amb_img_lin * (200 / 1600)

    amb_img_lin = cv2.cvtColor(
        amb_img_lin[:, :, ::-1].astype(np.float32), cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    flash_img_lin = cv2.cvtColor(
        flash_img_lin[:, :, ::-1].astype(np.float32), cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    tau_shad = 0.0005
    spec_thresh = 0.9

    shadow_mask = np.abs(flash_img_lin - amb_img_lin) <= tau_shad
    shadow_img = np.zeros_like(amb_img_lin)
    shadow_img[shadow_mask] = 1

    speckle_mask = (flash_img_lin - amb_img_lin) > spec_thresh
    speckle_img = np.zeros_like(amb_img_lin)
    speckle_img[speckle_mask] = 1

    # morphological filtering operations
    opening_kernel = np.ones((3, 3), np.uint8)
    closing_kernel = np.ones((8, 8), np.uint8)
    dilation_kernel = np.ones((15, 15), np.uint8)

    shadow_img = morph_operations(
        shadow_img, opening_kernel, closing_kernel, dilation_kernel)
    speckle_img = morph_operations(
        speckle_img, opening_kernel, closing_kernel, dilation_kernel)

    final_mask = shadow_img.copy()
    final_mask[speckle_img == 1] = 1
    final_mask = cv2.GaussianBlur(final_mask, (15, 15), 50)

    return final_mask


def main():

    img_path = '../data/lamp/lamp_ambient.tif'
    flash_img_path = '../data/lamp/lamp_flash.tif'
    img = skimage.io.imread(img_path)
    flash_img = skimage.io.imread(flash_img_path)

    img = skimage.transform.resize(
        img, (img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True)
    flash_img = skimage.transform.resize(
        flash_img, (flash_img.shape[0] // 2, flash_img.shape[1] // 2), anti_aliasing=True)

    # self image
    # self_img_path = '../data/self/kingohger.jpg'
    # flash_self_img_path = '../data/self/kingohger_flash.jpg'
    # self_img = skimage.io.imread(self_img_path)
    # flash_self_img = skimage.io.imread(flash_self_img_path)
    # self_img = skimage.transform.resize(
    #     self_img, (self_img.shape[0] // 2, self_img.shape[1] // 2),
    #     anti_aliasing=True)
    # flash_self_img = skimage.transform.resize(
    #     flash_self_img, (flash_self_img.shape[0] // 2, flash_self_img.shape[1] // 2), anti_aliasing=True)

    # normalize to 0-1
    img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)
    flash_img = ((flash_img - flash_img.min()) / (flash_img.max() -
                 flash_img.min())).astype(np.float32)

    # Lamp image
    # bilateral filtering
    A_base = bilateral_filter(img, img, kernel_size=5, sigma_r=0.05)
    plt.imshow(A_base)
    plt.axis('off')
    plt.show()

    # joint bilateral filtering
    A_NR = bilateral_filter(
        img, flash_img, kernel_size=3, sigma_r=0.02)

    plt.imshow(A_NR)
    plt.axis('off')
    plt.show()
    # plt.savefig('./A_joint.png', dpi=400, bbox_inches='tight', pad_inches=0)

    # detail transfer
    A_detail = detail_transfer(img, flash_img, sigma_r=0.05)

    plt.imshow(A_detail)
    plt.axis('off')
    plt.show()
    # plt.savefig('./A_detail.png', dpi=400, bbox_inches='tight', pad_inches=0)

    mask = specularity_masking(img, flash_img)

    plt.imshow(mask)
    plt.axis('off')
    plt.show()
    # plt.savefig('./self_mask.png', dpi=200, bbox_inches='tight', pad_inches=0)

    mask = np.repeat(mask[..., np.newaxis], 3, axis=2)

    final_img = (1 - mask) * A_detail + mask * A_base

    plt.imshow(final_img)
    plt.axis('off')
    plt.show()
    # plt.savefig('./final_img.png', dpi=400, bbox_inches='tight', pad_inches=0)

    # plt.imshow(self_img)
    # plt.axis('off')
    # plt.show()

    # # bilateral filtering
    # A_base = bilateral_filter(
    #     self_img, self_img, kernel_size=5, sigma_r=0.05)

    # plt.imshow(A_base)
    # plt.axis('off')
    # plt.show()

    # # self image
    # A_detail = detail_transfer(self_img, flash_self_img, sigma_r=0.05)

    # mask = specularity_masking(self_img, flash_self_img)
    # mask = np.repeat(mask[..., np.newaxis], 3, axis=2)
    # final_img = (1 - mask) * A_detail + mask * A_base

    # plt.imshow(final_img)
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    main()
