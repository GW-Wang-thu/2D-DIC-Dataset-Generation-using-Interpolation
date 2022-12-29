import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from displacement_generation import deformation
from interpolation import interpolator_N as intorpolator


class img_loader():
    def __init__(self, imdir):
        files = os.listdir(imdir)
        self.all_imgs = [os.path.join(imdir, file) for file in files if file.endswith("_img.bmp")]
        self.all_masks = [os.path.join(imdir, file) for file in files if file.endswith("_mask.bmp")]

    def load_sample(self):
        i = 0
        if np.random.rand() < 0.3 :
            while i == 0:
                idx = np.random.randint(0, len(self.all_imgs))
                temp_img = cv2.imread(self.all_imgs[idx], cv2.IMREAD_GRAYSCALE)
                temp_mask = cv2.imread(self.all_masks[idx], cv2.IMREAD_GRAYSCALE)
                imsize0 = temp_img.shape
                if imsize0[0] > 1024:
                    i = 1
        else:
            idx = np.random.randint(0, len(self.all_imgs))
            temp_img = cv2.imread(self.all_imgs[idx], cv2.IMREAD_GRAYSCALE)
            temp_mask = cv2.imread(self.all_masks[idx], cv2.IMREAD_GRAYSCALE)
            imsize0 = temp_img.shape
        height = min(max(((imsize0[0]//512))*256, 256), 1792)
        width = min(max(((imsize0[1]//512))*256, 256), 1792)
        imsize = (height, width)
        stp_x = np.random.randint(0, imsize0[0] - height)
        stp_y = np.random.randint(0, imsize0[1] - width)
        return temp_img[stp_x:stp_x+height, stp_y:stp_y+width], temp_mask[stp_x:stp_x+height, stp_y:stp_y+width], imsize


def generate_dataset(num=500):
    save_dir = r"F:\DATASET\LargeDeformation_Trial1\dataset\\"
    my_displacements = deformation(imsize=(256, 256))
    my_img_loader = img_loader(imdir=r"F:\DATASET\LargeDeformation_Trial1\raw_imgs\Z_imgs\\")
    my_interpolator = interpolator(imsize=(256, 256))
    for i in range(num):
        if os.path.exists(save_dir + str(i) + "_img_m.bmp"):
            continue
        temp_img, temp_mask, temp_size = my_img_loader.load_sample()
        my_displacements.imsize = temp_size
        my_interpolator.imsize = temp_size
        print(temp_size)
        if np.random.rand() < 0.7:
            if np.random.rand() < 0.7:
                pos, disp, def_mask = my_displacements.get_random_displacement(amp=np.random.randint(2, 10))
            else:
                pos, disp, def_mask = my_displacements.get_random_displacement(amp=np.random.randint(1, 11) / 6)
        else:
            if np.random.rand() < 0.7:
                pos, disp, def_mask = my_displacements.get_random_displacement_no_crack(amp=np.random.randint(2, 10))
            else:
                pos, disp, def_mask = my_displacements.get_random_displacement_no_crack(amp=np.random.randint(1, 11) / 6)
        temp_ref_img = my_interpolator.interpolation(pos[0], pos[1], temp_img * def_mask)
        temp_ref_mask = my_interpolator.interpolation(pos[0], pos[1], temp_mask * def_mask)
        temp_def_img = temp_img * def_mask
        temp_u = disp[0]
        temp_v = disp[1]

        np.save(save_dir + str(i) + "_imgs.npy", np.array([temp_ref_mask, temp_ref_img, temp_def_img]))
        np.save(save_dir + str(i) + "_disps.npy", np.array([temp_u, temp_v]))
        cv2.imwrite(save_dir + str(i) + "_img_r.bmp", temp_ref_img)
        cv2.imwrite(save_dir + str(i) + "_img_d.bmp", temp_def_img)
        cv2.imwrite(save_dir + str(i) + "_img_m.bmp", temp_ref_mask)
        plt.figure(figsize=(8, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(temp_u)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(temp_v)
        plt.colorbar()
        plt.savefig(save_dir + str(i) + "_disps.png")
        plt.close()


def swap_dataset():
    save_dir = r"G:\\"
    data_dir = r"F:\DATASET\LargeDeformation_Trial1\dataset\\"
    train_percent = 0.8
    train_id_list = []
    valid_id_list = []
    if os.path.exists(data_dir + "Train\\id_list.csv"):
        train_id_list = np.loadtxt(data_dir + "Train\\id_list.csv", dtype="int32").tolist()
    if os.path.exists(data_dir + "Valid\\id_list.csv"):
        valid_id_list = np.loadtxt(data_dir + "Valid\\id_list.csv", dtype="int32").tolist()
    all_data = os.listdir(data_dir)
    all_imgs = [os.path.join(data_dir, file) for file in all_data if file.endswith("_imgs.npy")]
    all_disps = [os.path.join(data_dir, file) for file in all_data if file.endswith("_disps.npy")]

    k = 0
    for i in range(len(all_imgs) + len(train_id_list) + len(valid_id_list)):
        if i in train_id_list or i in valid_id_list:
            continue
        if np.random.rand() < train_percent:
            temp_type = "Train"
            train_id_list.append(k)
        else:
            temp_type = "Valid"
            valid_id_list.append(k)
        temp_imgs = np.load(all_imgs[k])
        temp_imgs[0, :, :] = temp_imgs[0, :, :] > 10
        temp_disps = np.load(all_disps[k])
        np.save(save_dir + temp_type + "//" + str(i) + "_img&disp.npy", np.concatenate([temp_imgs, temp_disps], axis=0))
        k += 1
    np.savetxt(save_dir + "Train/id_list.csv", np.array(train_id_list))
    np.savetxt(save_dir + "Valid/id_list.csv", np.array(valid_id_list))



if __name__ == '__main__':
    generate_dataset(num=500)
    # swap_dataset()

