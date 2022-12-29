import cv2
import numpy as np
import torch
import sympy as sy
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d


class interpolator():

    def __init__(self):
        self.kernal_matrix = torch.from_numpy(np.array([[-0.5, 1.5, -1.5, 0.5],
                                                        [1.0, -2.5, 2.0, -0.5],
                                                        [-0.5, 0, 0.5, 0],
                                                        [0.0, 1.0, 0.0, 0.0]], dtype="float32")).cuda()

    def interpolation(self, u_pos, v_pos, init_array, imsize, img_mode=True):
        target_imsize = u_pos.shape

        u_pos = torch.from_numpy(u_pos).flatten().cuda()
        v_pos = torch.from_numpy(v_pos).flatten().cuda()
        gray_array = torch.from_numpy(init_array).flatten().cuda()

        pos_x_0 = torch.floor(u_pos).to(torch.long)
        mask_x_0 = (pos_x_0 >= 0) * (pos_x_0 <= imsize[0]-1)

        pos_y_0 = torch.floor(v_pos).to(torch.long)
        mask_y_0 = (pos_y_0 >= 0) * (pos_y_0 <= imsize[1]-1)

        pos_x_m1 = pos_x_0 - 1
        mask_x_m1 = (pos_x_m1 >= 0) * (pos_x_m1 <= imsize[0]-1)

        pos_y_m1 = pos_y_0 - 1
        mask_y_m1 = (pos_y_m1 >= 0) * (pos_y_m1 <= imsize[1]-1)

        pos_x_1 = pos_x_0 + 1
        mask_x_1 = (pos_x_1 >= 0) * (pos_x_1 <= imsize[0]-1)

        pos_y_1 = pos_y_0 + 1
        mask_y_1 = (pos_y_1 >= 0) * (pos_y_1 <= imsize[1]-1)

        pos_x_2 = pos_x_0 + 2
        mask_x_2 = (pos_x_2 >= 0) * (pos_x_2 <= imsize[0]-1)

        pos_y_2 = pos_y_0 + 2
        mask_y_2 = (pos_y_2 >= 0) * (pos_y_2 <= imsize[1]-1)

        pos_x_0 = (pos_x_0 * mask_x_0)
        pos_y_0 = (pos_y_0 * mask_y_0)
        pos_x_m1 = (pos_x_m1 * mask_x_m1)
        pos_y_m1 = (pos_y_m1 * mask_y_m1)
        pos_x_1 = (pos_x_1 * mask_x_1)
        pos_y_1 = (pos_y_1 * mask_y_1)
        pos_x_2 = (pos_x_2 * mask_x_2)
        pos_y_2 = (pos_y_2 * mask_y_2)

        tx_vect = u_pos - torch.floor(u_pos)
        ty_vect = v_pos - torch.floor(v_pos)

        g_m1_m1 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_m1) * mask_x_m1 * mask_y_m1
        g_m1_0 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_0) * mask_x_m1 * mask_y_0
        g_m1_1 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_1) * mask_x_m1 * mask_y_1
        g_m1_2 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_2) * mask_x_m1 * mask_y_2
        g_0_m1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_m1) * mask_x_0 * mask_y_m1
        g_0_0 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_0) * mask_x_0 * mask_y_0
        g_0_1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_1) * mask_x_0 * mask_y_1
        g_0_2 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_2) * mask_x_0 * mask_y_2
        g_1_m1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_m1) * mask_x_1 * mask_y_m1
        g_1_0 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_0) * mask_x_1 * mask_y_0
        g_1_1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_1) * mask_x_1 * mask_y_1
        g_1_2 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_2) * mask_x_1 * mask_y_2
        g_2_m1 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_m1) * mask_x_2 * mask_y_m1
        g_2_0 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_0) * mask_x_2 * mask_y_0
        g_2_1 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_1) * mask_x_2 * mask_y_1
        g_2_2 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_2) * mask_x_2 * mask_y_2

        b_m1 = self.__p(tx_vect, torch.column_stack([g_m1_m1, g_0_m1, g_1_m1, g_2_m1]))
        b_0 = self.__p(tx_vect, torch.column_stack([g_m1_0, g_0_0, g_1_0, g_2_0]))
        b_1 = self.__p(tx_vect, torch.column_stack([g_m1_1, g_0_1, g_1_1, g_2_1]))
        b_2 = self.__p(tx_vect, torch.column_stack([g_m1_2, g_0_2, g_1_2, g_2_2]))
        g_inter = self.__p(ty_vect, torch.column_stack([b_m1, b_0, b_1, b_2]))
        img_float = g_inter.unflatten(dim=0, sizes=target_imsize).detach().cpu().numpy()
        if img_mode:
            img = (img_float - 255) * (img_float <= 255) + 255
            img = img * (img > 0)
            return img.astype("uint8")
        else:
            return img_float

    def __p(self, t_vect, gray_map):
        t_vect_array = torch.column_stack([t_vect**3, t_vect**2, t_vect, torch.ones_like(t_vect).cuda()])
        a = torch.matmul(t_vect_array, self.kernal_matrix)
        b = torch.sum(a * gray_map, dim=-1)                            # n * 4 dot 4 * 4
        return b


class interpolator_N():

    def __init__(self):
        # a = 0.5
        # self.kernal_matrix = torch.from_numpy(np.array([[-0.5, 1.5, -1.5, 0.5],
        #                                                 [1.0, -2.5, 2.0, -0.5],
        #                                                 [-0.5, 0, 0.5, 0],
        #                                                 [0.0, 1.0, 0.0, 0.0]], dtype="float32")).cuda()
        # bicubic
        self.kernal_matrix_bicubic = torch.from_numpy(np.array([[-0.75, 1.25, -1.25, 0.75],
                                                        [1.5, -2.25, 1.5, -0.75],
                                                        [-0.75, 0, 0.75, 0],
                                                        [0.0, 1.0, 0.0, 0.0]], dtype="float32")).cuda()
        # b_spline
        self.kernal_matrix_spline = torch.from_numpy(np.array([[-0.166666666666667, 0.5, -0.5, 0.166666666666667],
                                                        [0.5, -1.0, 0.5, 0],
                                                        [-0.5, 0, 0.5, 0],
                                                        [0.166666666666667, 0.666666666666667, 0.166666666666667, 0]], dtype="float32")).cuda()
        # bilinear
        self.kernal_matrix_spline = torch.from_numpy(np.array([[-0.166666666666667, 0.5, -0.5, 0.166666666666667],
                                                        [0.5, -1.0, 0.5, 0],
                                                        [-0.5, 0, 0.5, 0],
                                                        [0.166666666666667, 0.666666666666667, 0.166666666666667, 0]], dtype="float32")).cuda()

    def interpolation(self, u_pos, v_pos, init_array, imsize, img_mode=True, kernel="bicubic"):
        if kernel == 'bicubic':
            kernel_matrix = self.kernal_matrix_bicubic
        else:
            kernel_matrix = self.kernal_matrix_spline

        target_imsize = u_pos.shape

        u_pos = torch.from_numpy(u_pos).flatten().cuda()
        v_pos = torch.from_numpy(v_pos).flatten().cuda()
        gray_array = torch.from_numpy(init_array).flatten().cuda()

        pos_x_0 = torch.floor(u_pos).to(torch.long)
        mask_x_0 = (pos_x_0 >= 0) * (pos_x_0 <= imsize[0]-1)

        pos_y_0 = torch.floor(v_pos).to(torch.long)
        mask_y_0 = (pos_y_0 >= 0) * (pos_y_0 <= imsize[1]-1)

        pos_x_m1 = pos_x_0 - 1
        mask_x_m1 = (pos_x_m1 >= 0) * (pos_x_m1 <= imsize[0]-1)

        pos_y_m1 = pos_y_0 - 1
        mask_y_m1 = (pos_y_m1 >= 0) * (pos_y_m1 <= imsize[1]-1)

        pos_x_1 = pos_x_0 + 1
        mask_x_1 = (pos_x_1 >= 0) * (pos_x_1 <= imsize[0]-1)

        pos_y_1 = pos_y_0 + 1
        mask_y_1 = (pos_y_1 >= 0) * (pos_y_1 <= imsize[1]-1)

        pos_x_2 = pos_x_0 + 2
        mask_x_2 = (pos_x_2 >= 0) * (pos_x_2 <= imsize[0]-1)

        pos_y_2 = pos_y_0 + 2
        mask_y_2 = (pos_y_2 >= 0) * (pos_y_2 <= imsize[1]-1)

        pos_x_0 = (pos_x_0 * mask_x_0)
        pos_y_0 = (pos_y_0 * mask_y_0)
        pos_x_m1 = (pos_x_m1 * mask_x_m1)
        pos_y_m1 = (pos_y_m1 * mask_y_m1)
        pos_x_1 = (pos_x_1 * mask_x_1)
        pos_y_1 = (pos_y_1 * mask_y_1)
        pos_x_2 = (pos_x_2 * mask_x_2)
        pos_y_2 = (pos_y_2 * mask_y_2)

        tx_vect = u_pos - torch.floor(u_pos)
        ty_vect = v_pos - torch.floor(v_pos)

        g_m1_m1 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_m1) * mask_x_m1 * mask_y_m1
        g_m1_0 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_0) * mask_x_m1 * mask_y_0
        g_m1_1 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_1) * mask_x_m1 * mask_y_1
        g_m1_2 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_2) * mask_x_m1 * mask_y_2
        g_0_m1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_m1) * mask_x_0 * mask_y_m1
        g_0_0 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_0) * mask_x_0 * mask_y_0
        g_0_1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_1) * mask_x_0 * mask_y_1
        g_0_2 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_2) * mask_x_0 * mask_y_2
        g_1_m1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_m1) * mask_x_1 * mask_y_m1
        g_1_0 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_0) * mask_x_1 * mask_y_0
        g_1_1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_1) * mask_x_1 * mask_y_1
        g_1_2 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_2) * mask_x_1 * mask_y_2
        g_2_m1 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_m1) * mask_x_2 * mask_y_m1
        g_2_0 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_0) * mask_x_2 * mask_y_0
        g_2_1 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_1) * mask_x_2 * mask_y_1
        g_2_2 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_2) * mask_x_2 * mask_y_2

        t_x_vect = torch.column_stack([tx_vect**3, tx_vect**2, tx_vect, torch.ones_like(tx_vect).cuda()])
        t_y_vect = torch.column_stack([ty_vect**3, ty_vect**2, ty_vect, torch.ones_like(ty_vect).cuda()])
        b = torch.matmul(t_x_vect, kernel_matrix).unsqueeze(2)
        a = torch.matmul(t_y_vect, kernel_matrix).unsqueeze(1)

        ratios = torch.reshape(torch.bmm(b, a), (-1, 16))
        cated_grays = torch.column_stack([g_m1_m1, g_m1_0, g_m1_1, g_m1_2,
                                          g_0_m1, g_0_0, g_0_1, g_0_2,
                                          g_1_m1, g_1_0, g_1_1, g_1_2,
                                          g_2_m1, g_2_0, g_2_1, g_2_2])
        g_inter = torch.sum(ratios * cated_grays, dim=-1)
        img_float = g_inter.unflatten(dim=0, sizes=target_imsize).detach().cpu().numpy()
        if img_mode:
            img = (img_float - 255) * (img_float <= 255) + 255
            img = img * (img >= 0)
            return img.astype("uint8")
        else:
            return img_float

    def interpolation_list(self, u_pos, v_pos, init_array_list, imsize, img_mode=True, kernel="bicubic"):
        if kernel == 'bicubic':
            kernel_matrix = self.kernal_matrix_bicubic
        else:
            kernel_matrix = self.kernal_matrix_spline
        target_imsize = u_pos.shape

        u_pos = torch.from_numpy(u_pos).flatten().cuda()
        v_pos = torch.from_numpy(v_pos).flatten().cuda()

        pos_x_0 = torch.floor(u_pos).to(torch.long)
        mask_x_0 = (pos_x_0 >= 0) * (pos_x_0 <= imsize[0]-1)

        pos_y_0 = torch.floor(v_pos).to(torch.long)
        mask_y_0 = (pos_y_0 >= 0) * (pos_y_0 <= imsize[1]-1)

        pos_x_m1 = pos_x_0 - 1
        mask_x_m1 = (pos_x_m1 >= 0) * (pos_x_m1 <= imsize[0]-1)

        pos_y_m1 = pos_y_0 - 1
        mask_y_m1 = (pos_y_m1 >= 0) * (pos_y_m1 <= imsize[1]-1)

        pos_x_1 = pos_x_0 + 1
        mask_x_1 = (pos_x_1 >= 0) * (pos_x_1 <= imsize[0]-1)

        pos_y_1 = pos_y_0 + 1
        mask_y_1 = (pos_y_1 >= 0) * (pos_y_1 <= imsize[1]-1)

        pos_x_2 = pos_x_0 + 2
        mask_x_2 = (pos_x_2 >= 0) * (pos_x_2 <= imsize[0]-1)

        pos_y_2 = pos_y_0 + 2
        mask_y_2 = (pos_y_2 >= 0) * (pos_y_2 <= imsize[1]-1)

        pos_x_0 = (pos_x_0 * mask_x_0)
        pos_y_0 = (pos_y_0 * mask_y_0)
        pos_x_m1 = (pos_x_m1 * mask_x_m1)
        pos_y_m1 = (pos_y_m1 * mask_y_m1)
        pos_x_1 = (pos_x_1 * mask_x_1)
        pos_y_1 = (pos_y_1 * mask_y_1)
        pos_x_2 = (pos_x_2 * mask_x_2)
        pos_y_2 = (pos_y_2 * mask_y_2)

        tx_vect = u_pos - torch.floor(u_pos)
        ty_vect = v_pos - torch.floor(v_pos)

        t_x_vect = torch.column_stack([tx_vect ** 3, tx_vect ** 2, tx_vect, torch.ones_like(tx_vect).cuda()])
        t_y_vect = torch.column_stack([ty_vect ** 3, ty_vect ** 2, ty_vect, torch.ones_like(ty_vect).cuda()])
        b = torch.matmul(t_x_vect, kernel_matrix).unsqueeze(2)
        a = torch.matmul(t_y_vect, kernel_matrix).unsqueeze(1)

        ratios = torch.reshape(torch.bmm(b, a), (-1, 16))

        results = []
        for i in range(len(init_array_list)):
            gray_array = torch.from_numpy(init_array_list[i]).flatten().cuda()
            g_m1_m1 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_m1) * mask_x_m1 * mask_y_m1
            g_m1_0 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_0) * mask_x_m1 * mask_y_0
            g_m1_1 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_1) * mask_x_m1 * mask_y_1
            g_m1_2 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_2) * mask_x_m1 * mask_y_2
            g_0_m1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_m1) * mask_x_0 * mask_y_m1
            g_0_0 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_0) * mask_x_0 * mask_y_0
            g_0_1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_1) * mask_x_0 * mask_y_1
            g_0_2 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_2) * mask_x_0 * mask_y_2
            g_1_m1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_m1) * mask_x_1 * mask_y_m1
            g_1_0 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_0) * mask_x_1 * mask_y_0
            g_1_1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_1) * mask_x_1 * mask_y_1
            g_1_2 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_2) * mask_x_1 * mask_y_2
            g_2_m1 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_m1) * mask_x_2 * mask_y_m1
            g_2_0 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_0) * mask_x_2 * mask_y_0
            g_2_1 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_1) * mask_x_2 * mask_y_1
            g_2_2 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_2) * mask_x_2 * mask_y_2
            cated_grays = torch.column_stack([g_m1_m1, g_m1_0, g_m1_1, g_m1_2,
                                              g_0_m1, g_0_0, g_0_1, g_0_2,
                                              g_1_m1, g_1_0, g_1_1, g_1_2,
                                              g_2_m1, g_2_0, g_2_1, g_2_2])
            g_inter = torch.sum(ratios * cated_grays, dim=-1)
            img_float = g_inter.unflatten(dim=0, sizes=target_imsize).detach().cpu().numpy()

            if img_mode:
                img = (img_float - 255) * (img_float <= 255) + 255
                img = img * (img >= 0)
                results.append(img.astype("uint8"))
            else:
                results.append(img_float)
        return results

    def interpolation_bilinear(self, u_pos, v_pos, init_array, imsize, img_mode=True):

        target_imsize = u_pos.shape

        u_pos = torch.from_numpy(u_pos).flatten().cuda()
        v_pos = torch.from_numpy(v_pos).flatten().cuda()
        gray_array = torch.from_numpy(init_array).flatten().cuda()

        pos_x_0 = torch.floor(u_pos).to(torch.long)
        mask_x_0 = (pos_x_0 >= 0) * (pos_x_0 <= imsize[0] - 1)

        pos_y_0 = torch.floor(v_pos).to(torch.long)
        mask_y_0 = (pos_y_0 >= 0) * (pos_y_0 <= imsize[1] - 1)

        pos_x_1 = pos_x_0 + 1
        mask_x_1 = (pos_x_1 >= 0) * (pos_x_1 <= imsize[0] - 1)

        pos_y_1 = pos_y_0 + 1
        mask_y_1 = (pos_y_1 >= 0) * (pos_y_1 <= imsize[1] - 1)

        pos_x_0 = (pos_x_0 * mask_x_0)
        pos_y_0 = (pos_y_0 * mask_y_0)
        pos_x_1 = (pos_x_1 * mask_x_1)
        pos_y_1 = (pos_y_1 * mask_y_1)

        tx_vect = u_pos - torch.floor(u_pos)
        ty_vect = v_pos - torch.floor(v_pos)

        g_0_0 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_0) * mask_x_0 * mask_y_0
        g_0_1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_1) * mask_x_0 * mask_y_1
        g_1_0 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_0) * mask_x_1 * mask_y_0
        g_1_1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_1) * mask_x_1 * mask_y_1

        a_00 = g_0_0
        a_10 = - g_0_0 + g_1_0
        a_01 = - g_0_0 + g_0_1
        a_11 = g_0_0 - g_0_1 - g_1_0 + g_1_1

        g_inter = a_00 + a_10 * tx_vect + a_01 * ty_vect + a_11 * tx_vect * ty_vect
        img_float = g_inter.unflatten(dim=0, sizes=target_imsize).detach().cpu().numpy()
        if img_mode:
            img = (img_float - 255) * (img_float <= 255) + 255
            img = img * (img >= 0)
            return img.astype("uint8")
        else:
            return img_float

    def interpolation_bilinear_list(self, u_pos, v_pos, init_array_list, imsize, img_mode=True):

        target_imsize = u_pos.shape

        u_pos = torch.from_numpy(u_pos).flatten().cuda()
        v_pos = torch.from_numpy(v_pos).flatten().cuda()

        pos_x_0 = torch.floor(u_pos).to(torch.long)
        mask_x_0 = (pos_x_0 >= 0) * (pos_x_0 <= imsize[0] - 1)

        pos_y_0 = torch.floor(v_pos).to(torch.long)
        mask_y_0 = (pos_y_0 >= 0) * (pos_y_0 <= imsize[1] - 1)

        pos_x_1 = pos_x_0 + 1
        mask_x_1 = (pos_x_1 >= 0) * (pos_x_1 <= imsize[0] - 1)

        pos_y_1 = pos_y_0 + 1
        mask_y_1 = (pos_y_1 >= 0) * (pos_y_1 <= imsize[1] - 1)

        pos_x_0 = (pos_x_0 * mask_x_0)
        pos_y_0 = (pos_y_0 * mask_y_0)
        pos_x_1 = (pos_x_1 * mask_x_1)
        pos_y_1 = (pos_y_1 * mask_y_1)

        tx_vect = u_pos - torch.floor(u_pos)
        ty_vect = v_pos - torch.floor(v_pos)


        results = []
        for i in range(len(init_array_list)):
            gray_array = torch.from_numpy(init_array_list[i]).flatten().cuda()
            g_0_0 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_0) * mask_x_0 * mask_y_0
            g_0_1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_1) * mask_x_0 * mask_y_1
            g_1_0 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_0) * mask_x_1 * mask_y_0
            g_1_1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_1) * mask_x_1 * mask_y_1

            a_00 = g_0_0
            a_10 = - g_0_0 + g_1_0
            a_01 = - g_0_0 + g_0_1
            a_11 = g_0_0 - g_0_1 - g_1_0 + g_1_1

            g_inter = a_00 + a_10 * tx_vect + a_01 * ty_vect + a_11 * tx_vect * ty_vect
            img_float = g_inter.unflatten(dim=0, sizes=target_imsize).detach().cpu().numpy()

            if img_mode:
                img = (img_float - 255) * (img_float <= 255) + 255
                img = img * (img >= 0)
                results.append(img.astype("uint8"))
            else:
                results.append(img_float)
        return results


def solve_w():
    a = -0.4
    t = sy.symbols("t")
    w_m1 = a * (1+t)**3 - 5*a*(1+t)**2 + 8*a*(1+t) - 4*a
    w_0 = (a + 2) * t**3 - (a+3) * t**2 + 1
    w_1 = (a + 2) * (1-t)**3 - (a+3) * (1-t)**2 + 1
    w_2 = a * (2-t)**3 - 5*a*(2-t)**2 + 8*a*(2-t) - 4*a
    print(sy.series(w_m1, t, 0, 5))
    print(sy.series(w_0, t, 0, 5))
    print(sy.series(w_1, t, 0, 5))
    print(sy.series(w_2, t, 0, 5))


def solve_b_splin():
    t = sy.symbols("t")
    w_m1 = (1/6) * (2 - (1+t))**3
    w_0 = 2/3 - (1-t/2)*t**2
    w_1 = 2/3 - (1 - (1-t)/2)*(1-t)**2
    w_2 = (1/6) * (2 - (2-t))**3
    print(sy.series(w_m1, t, 0, 5))
    print(sy.series(w_0, t, 0, 5))
    print(sy.series(w_1, t, 0, 5))
    print(sy.series(w_2, t, 0, 5))

    x = sy.symbols("x")
    A = -0.75
    coeffs_0 = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
    coeffs_1 = ((A + 2)*x - (A + 3))*x*x + 1;
    coeffs_2 = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
    coeffs_3 = 1.0 - coeffs_0 - coeffs_1 - coeffs_2;

    print(sy.series(coeffs_0, x, 0, 5))
    print(sy.series(coeffs_1, x, 0, 5))
    print(sy.series(coeffs_2, x, 0, 5))
    print(sy.series(coeffs_3, x, 0, 5))


def eval_method():
    my_img = cv2.imread("../demoimg/Demoimg2.bmp", cv2.IMREAD_GRAYSCALE)
    my_interpolator = interpolator(imsize=my_img.shape)
    v_pos, u_pos = np.meshgrid(np.arange(0, my_img.shape[1]), np.arange(0, my_img.shape[0]))
    u_pos = u_pos.astype("float64") * 0.8
    v_pos = v_pos.astype("float64") * 0.8
    now = time.perf_counter()
    img = my_interpolator.interpolation(u_pos, v_pos, my_img)
    print(time.perf_counter() - now)
    cv2.imwrite("../demoimg/Reiszed.bmp", img)
    now = time.perf_counter()
    cv2.imwrite("../demoimg/Reiszed1.bmp",
                cv2.resize(my_img, dsize=None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)[:my_img.shape[0],
                :my_img.shape[1]])
    print(time.perf_counter() - now)
    solve_w()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # my_img = cv2.imread("../demoimg/Demoimg2.bmp", cv2.IMREAD_GRAYSCALE)
    # my_interpolator = interpolator(imsize=my_img.shape)
    # v_pos, u_pos = np.meshgrid(np.arange(0, my_img.shape[1]), np.arange(0, my_img.shape[0]))
    # deform_x = cv2.resize(30 * np.random.random(size=(6, 4))-10, dsize=(my_img.shape[1], my_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # deform_y = cv2.resize(50 * np.random.random(size=(6, 4))-20, dsize=(my_img.shape[1], my_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # u_pos = u_pos.astype("float64")*1.0 + deform_x
    # v_pos = v_pos.astype("float64")*1.0 + deform_y
    # img = my_interpolator.interpolation(u_pos, v_pos, my_img)
    # plt.imshow(deform_x)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(deform_y)
    # plt.colorbar()
    # plt.show()
    # cv2.imwrite("../demoimg/DemoLena.bmp", img)
    # solve_w()
    solve_b_splin()