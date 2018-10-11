"""
skin_color_separation.skin_threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~

这个模块提供一个Threshold类用来计算肤色分割所需要的动态阈值t
"""


class Threshold:
    # 肤色相似度矩阵
    __image = None
    # 默认分10级
    __level = 10
    # 像素分级容器
    basket = None
    # 阈值
    t = 0

    def __init__(self, image):
        """
        初始化image是一个二维的float类型的numpy数组.
        """
        self.__image = image
        self.height, self.width = self.__image.shape

    def set_level(self, lev):
        """重新设定__level"""
        self.__level = lev

    def compute(self):
        self.basket = [0] * self.__level
        for i in range(self.height):
            for j in range(self.width):
                group = int(self.__image[i][j] * self.__level)
                self.basket[group] += 1
        print(self.basket)

    def threshold(self):
        """运行前先利用compute()初始化basket，否则出错
        """
        # 肤色区域在整副图像中所占的比例w0
        w0 = [sum(self.basket[i:]) / (self.height * self.width) for i in range(self.__level)]

        # 背景区域在整副图像中所占的比例w1
        w1 = [sum(self.basket[:i]) / (self.height * self.width) for i in range(self.__level)]

        # 肤色区域均值u0
        kp = list(map(lambda x, y: x * y, self.basket, range(self.__level)))
        u0 = [sum(kp[i:]) / sum(self.basket[i:]) for i in range(self.__level)]

        # 背景区域均值u1
        u1 = [0.0]
        u1.extend([sum(kp[:i]) / sum(self.basket[:i]) for i in range(1, self.__level)])

        # 整个图像总均值u
        u = u0[0]

        # 肤色区域的类内方差q0
        q0 = []
        for i in range(self.__level):
            pq0 = list(map(lambda x, y: (x - u0[i]) * (x - u0[i]) * y, range(self.__level)[i:], self.basket[i:]))
            q0.append(sum(pq0))

        # 背景区域的类内方差q1
        q1 = []
        for i in range(self.__level):
            pq1 = list(map(lambda x, y: (x - u1[i]) * (x - u1[i]) * y, range(self.__level)[:i], self.basket[i:]))
            q1.append(sum(pq1))

        # 求阈值t
        temp1 = list(map(lambda w, x, y, z: w * (y - u) * (y - u) + x * (z - u) * (z - u), w0, w1, u0, u1))
        temp2 = list(map(lambda w, x, y, z: w * y * y + x * z * z, w0, w1, q0, q1))
        temp = list(map(lambda x, y: x / y, temp1, temp2))
        self.t = temp.index(max(temp))
        return self.t/self.__level


if __name__ == '__main__':
    import skin_similarity
    import cv2
    import numpy as np

    img = cv2.imread('../pictures/jin.jpg')
    skin_img = skin_similarity.y_cg_cr(img)
    obj = Threshold(skin_img)
    obj.compute()
    div = obj.threshold()
    print(div)
    final_img = 255 * (skin_img >= div)
    final_img = final_img.astype('uint8')
    print(final_img)
    cv2.namedWindow('image', 0)
    cv2.imshow('image', final_img)
    ke = np.ones((10, 10), np.uint8)
    opening = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, kernel=ke)
    cv2.namedWindow('image1', 0)
    cv2.imshow('image1', opening)
    cv2.waitKey(60000)
    cv2.destroyAllWindows()
