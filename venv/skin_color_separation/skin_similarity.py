import cv2
import math
import numpy as np


def y_cg_cr(image):

    # 转换矩阵
    matrix = np.array([[65.481, 128.553, 24.966], [-81.085, 112, -30.915], [112, -93.768, -18.214]])

    height, width, m = image.shape
    image = image.astype(float)

    # 求YCgCr色彩空间图像new_image
    new_image = np.zeros(image.shape)
    for i in range(height):
        for j in range(width):
            new_image[i][j] = [16, 128, 128] + np.dot(matrix, image[i][j]) / 256

    # 切除亮度分量y,留下色度分量cg_cr
    new_image = new_image[:, :, 1:]

    # 求色度分量均值m
    m = np.sum(np.sum(new_image, axis=0), axis=0) / (height*width)

    # 求色度分量协方差c
    c_matrix = np.zeros((height, width, 2, 2))
    for i in range(height):
        for j in range(width):
            c_matrix[i][j] = np.asmatrix((new_image[i][j]-m)).T * np.asmatrix(new_image[i][j]-m)
    c_matrix = np.array(c_matrix)
    c = np.sum(np.sum(c_matrix, axis=0), axis=0) / (height*width)

    # 求肤色相似度sim_image
    sim_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            temp = np.sum(np.asmatrix(new_image[i][j]-m) * (np.asmatrix(c)).I * (np.matrix(new_image[i][j])-m).T)
            sim_image[i][j] = math.exp(-0.5 * temp)
    return sim_image


if __name__ == '__main__':
    img = cv2.imread('../pictures/liu.jpg')
    sim_img = y_cg_cr(img)
    print(np.max(sim_img), np.min(sim_img))