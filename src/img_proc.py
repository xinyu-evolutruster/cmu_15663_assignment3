import numpy as np
from matplotlib import pyplot as plt
import skimage

# img_path = './self_img_charbook.png'
# flash_img_path = './self_flash_img_charbook.png'

# fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
# axes[0].imshow(skimage.io.imread(img_path))
# axes[0].axis('off')
# axes[0].set_title('No-flash image')
# axes[1].imshow(skimage.io.imread(flash_img_path))
# axes[1].axis('off')
# axes[1].set_title('Flash image')

# plt.savefig('./no_flash_flash_imgs.png', bbox_inches='tight', pad_inches=0)

taus = [0.3, 0.5, 0.7, 0.9]
sigmas = [10, 20, 40, 80]
img_path = './integrated_{}_{}.png'

for tau in taus:
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), constrained_layout=True)
    idx = 0
    for sigma in sigmas:
        path = img_path.format(tau, sigma)
        axes[idx].imshow(skimage.io.imread(path))
        axes[idx].axis('off')
        axes[idx].set_title('tau_s = {}, sigma = {}'.format(tau, sigma))
        idx += 1
    plt.savefig('./par_{}_{}.png'.format(tau, sigma))

# self_img_path = '../data/self/kingohger.jpg'
# flash_self_img_path = '../data/self/kingohger_flash.jpg'
# self_img = skimage.io.imread(self_img_path)
# flash_self_img = skimage.io.imread(flash_self_img_path)
# self_img = skimage.transform.resize(
#     self_img, (self_img.shape[0] // 2, self_img.shape[1] // 2),
#     anti_aliasing=True)
# flash_self_img = skimage.transform.resize(
#     flash_self_img, (flash_self_img.shape[0] // 2, flash_self_img.shape[1] // 2), anti_aliasing=True)

# fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
# axes[0].imshow(self_img)
# axes[0].axis('off')
# axes[0].set_title('No-flash image')
# axes[1].imshow(flash_self_img)
# axes[1].axis('off')
# axes[1].set_title('Flash image')

# plt.savefig('./no_flash_flash_imgs.png', bbox_inches='tight', pad_inches=0)

# detail_img_path = '../data/lamp/results/A_detail_diff2base.png'
# final_img_path = '../data/lamp/results/A_final_diff2base.png'
# NR_img_path = '../data/lamp/results/A_NR_diff2base.png'

# fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
# axes[0].imshow(skimage.io.imread(NR_img_path))
# axes[0].axis('off')
# axes[0].set_title('A_NR')
# axes[1].imshow(skimage.io.imread(detail_img_path))
# axes[1].axis('off')
# axes[1].set_title('A_Detail')
# axes[2].imshow(skimage.io.imread(final_img_path))
# axes[2].axis('off')
# axes[2].set_title('A_Final')

# plt.savefig('./diff_images.png', bbox_inches='tight', pad_inches=0)
