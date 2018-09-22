# coding=utf-8
import cv2
import numpy as np
import logging
from imutils import auto_canny, contours
from .exceptions import ImageException

logger = logging.getLogger('ImageProcessing')


class AnswerScanBase(object):
    @staticmethod
    def get_corner_node_list(poly_node_list):
        """
        获得多边形四个顶点的坐标
        """
        center_y, center_x = (np.sum(poly_node_list, axis=0) / 4)[0]
        top_left = bottom_left = top_right = bottom_right = None
        for node in poly_node_list:
            x = node[0, 1]
            y = node[0, 0]
            if x < center_x and y < center_y:
                top_left = node
            elif x < center_x and y > center_y:
                bottom_left = node
            elif x > center_x and y < center_y:
                top_right = node
            elif x > center_x and y > center_y:
                bottom_right = node
        return top_left, bottom_left, top_right, bottom_right

    def img_regulate(self, poly, base_img):
        """图片校偏"""
        # 计算多边形四个顶点，并且截图，然后处理截取后的图片
        top_left, bottom_left, top_right, bottom_right = self.get_corner_node_list(poly)
        # 多边形顶点和图片顶点，主要用于纠偏
        base_poly_nodes = np.float32([top_left[0], bottom_left[0], top_right[0], bottom_right[0]])
        base_nodes = np.float32([[0, 0],
                                 [base_img.shape[1], 0],
                                 [0, base_img.shape[0]],
                                 [base_img.shape[1], base_img.shape[0]]])
        transmtx = cv2.getPerspectiveTransform(base_poly_nodes, base_nodes)
        img_warp = cv2.warpPerspective(base_img, transmtx, (base_img.shape[1], base_img.shape[0]))
        return img_warp

    @staticmethod
    def get_bright_process_img(img):
        """调整图片亮度"""
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, -7)
        return img

    def get_max_area_cnt(self, img):
        """
        获得图片里面最大面积的轮廓
        """
        binary, cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=lambda c: cv2.contourArea(c))
        # 获取轮廓周长
        cnt_perimeter = cv2.arcLength(cnt, True)
        return cnt, cnt_perimeter

    def get_contours(self, base_img):
        """得到图片四角"""
        # 处理优化图片
        h = cv2.Sobel(base_img, cv2.CV_32F, 0, 1, -1)
        v = cv2.Sobel(base_img, cv2.CV_32F, 1, 0, -1)
        img = cv2.add(h, v)
        img = cv2.convertScaleAbs(img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        # 膨胀
        img = cv2.dilate(img, kernel, iterations=2)
        img = auto_canny(img)
        # 获取最大轮廓和轮廓周长
        cnt, cnt_perimeter = self.get_max_area_cnt(img)
        base_img_perimeter = (base_img.shape[0] + base_img.shape[1]) * 2

        # 答题卡框与整个图片周长比的阈值
        CNT_PERIMETER_THRESHOLD = 0.35
        if not cnt_perimeter > CNT_PERIMETER_THRESHOLD * base_img_perimeter:
            logger.error("[get_contours] 获取答题卡失败，请重新上传图片")
            raise ImageException("获取答题卡失败，请重新上传图片")

        # 10%，即0.1的精确度,忽略细节
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        # 计算多边形的顶点，并看是否是四个顶点
        poly_node_list = cv2.approxPolyDP(cnt, epsilon, True)
        if not poly_node_list.shape[0] == 4:
            logger.error("[get_contours] 不支持该答题卡，请重新上传图片")
            raise ImageException("不支持该答题卡，请重新上传图片")
        return poly_node_list


class AnswerScan(AnswerScanBase):
    """
    答题卡识别类
    """

    def __init__(self, file_path, debug=False):
        self.__base_img = cv2.imread(file_path)
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            logger.addHandler(ch)

    def scan(self):
        # 缩放图片
        rows, cols, channels = self.__base_img.shape
        # 计算边框
        rim = int((cols / rows) * 39.99)

        if cols > 1000:
            cols = 1000
        if rows > 1333:
            rows = 1333
        base_img = cv2.resize(
            self.__base_img,
            (cols, rows),
            interpolation=cv2.INTER_CUBIC
        )
        # 灰度化
        img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        poly_node_list = self.get_contours(img)
        # 图片纠偏
        if len(poly_node_list) == 0:
            logger.error("[scan] 答题卡四角获取失败,请重新上传图片")
            raise ImageException("答题卡四角获取失败,请重新上传图片")
        img = self.img_regulate(poly_node_list, base_img)

        # 调整图片的亮度
        img = self.get_bright_process_img(img)

        # 截取中间去除边框
        img = img[rim:-rim, rim:-rim]

        question_cnts = self.get_qustion_cnts(img)
        answer_cnts = self.get_answer_cnts(img)
        answer_list = self.proofreading_ans(img, question_cnts, answer_cnts)
        if self.debug:
            cv2.destroyAllWindows()
        if len(answer_list) == 0:
            raise ImageException("[scan] 识别失败")
        logger.info("[scan] 识别成功 %s" % str(answer_list))
        return answer_list

    def get_qustion_cnts(self, img):
        """查找选项框以及前面题号的轮廓"""
        # 定义结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        # 形态学闭运算填充目标内的孔洞
        quest_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # 腐蚀寻找题目
        quest_img = cv2.erode(quest_img, np.ones((4, 4), np.uint8), iterations=3)

        # 黑白反色
        choice_img = np.zeros(img.shape, np.uint8)
        cv2.bitwise_not(quest_img, choice_img)

        binary, cnts, h = cv2.findContours(choice_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 排除垃圾轮廓
        cnt_list = [cv2.contourArea(cnt) for cnt in cnts]
        avg_cnts = np.mean(cnt_list)
        for i, cnt in enumerate(cnts):
            if cv2.contourArea(cnt) < avg_cnts * 0.5:
                cnts.pop(i)

        if self.debug:
            # 新建黑板
            temp_white = np.zeros(choice_img.shape, np.uint8)
            cv2.drawContours(temp_white, cnts, -1, (255, 0, 0), 1)
            cv2.imshow('temp', temp_white)
            cv2.waitKey(0)
        return cnts

    def get_answer_cnts(self, img):
        """获取答案轮廓"""

        img = cv2.GaussianBlur(img, (5, 5), 0)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                            cv2.THRESH_BINARY_INV, 9, 5)

        # 腐蚀去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        choice_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # 腐蚀答案
        kernel = np.ones((2, 2), np.uint8)
        choice_img = cv2.erode(choice_img, kernel, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        choice_img = cv2.morphologyEx(choice_img, cv2.MORPH_OPEN, kernel)

        binary, cnts, h = cv2.findContours(choice_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

        if self.debug:
            img = np.zeros(choice_img.shape, np.uint8)
            cv2.drawContours(img, cnts, -1, (255, 0, 0), -1)
            cv2.imshow('temp', img)
            cv2.waitKey(0)

        return cnts

    def proofreading_ans(self, img, quest_cnts, ans_cnts):
        """解析答案"""
        # 画一个纯黑色的面板，并涂上题目的轮廓，用于调试用显示
        empty_img = np.zeros(img.shape, np.uint8)
        cv2.drawContours(empty_img, quest_cnts, -1, (255, 0, 0), 1)
        # 画一个纯黑色的面板，绘制答案轮廓
        processed_img = np.zeros(img.shape, np.uint8)
        cv2.drawContours(processed_img, ans_cnts, -1, (255, 0, 0), -1)

        # 以从顶部到底部将轮廓进行排序
        quest_cnts_row, cnts_top_bottom_pos = contours.sort_contours(quest_cnts, method="top-to-bottom")

        # 答案存储数组
        answer_list = []
        # 答案选择范围
        answer_range = ['', '', 'A', 'B', 'C', 'D']
        # 题目答案数
        ans_num = 4
        # 一行包含的题目轮廓
        question_row_num = (3 * (ans_num + 1))
        # 均值评估答案面积阀值
        all_area_index = 0
        all_area = 0
        # 每个题目有5个选项，所以5个气泡一组循环处理
        qust_index = 1
        for i in np.arange(0, len(quest_cnts_row), question_row_num):
            # 从左到右为当前题目的气泡轮廓排序
            quest_cnt_qust, cnt_left_right_pos = contours.sort_contours(
                quest_cnts_row[i:i + question_row_num], method="left-to-right")

            ans_index = 1
            answer_area = []
            for col in quest_cnt_qust:
                # 截取单个选项
                x, y, w, h = cv2.boundingRect(col)
                # top_left = (x, y)
                # top_right = (x + w, y)
                # bottom_left = (x, y + h)
                # bottom_right = (x + w, y + h)
                roi_img = processed_img[y:y + h, x:x + w]
                if ans_index == 1:
                    logger.debug("题目%d" % qust_index)
                    if self.debug:
                        cv2.putText(empty_img, str(qust_index), (x, int(y + (h * 3 / 4))),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), thickness=4, lineType=8)
                    ans_index = ans_index + 1
                    qust_index = qust_index + 1
                    continue

                binary, cnts, h = cv2.findContours(roi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    # 找不到轮廓，该选项为答案
                    answer_area.append({'index': ans_index, 'area': 0, "qnst_area": cv2.contourArea(col), "cnt": [col]})
                else:
                    # 找到最大轮廓，计算面积
                    cnt = max(cnts, key=lambda c: cv2.contourArea(c))
                    area = cv2.contourArea(cnt)
                    qnst_area = cv2.contourArea(col)
                    answer_area.append({'index': ans_index, 'area': area, "qnst_area": qnst_area, "cnt": [col]})

                if ans_index >= ans_num + 1:
                    # 扫过一题
                    get_answer = 0
                    answer_area = sorted(answer_area, key=lambda d: d['area'])
                    if answer_area[0]['area'] == 0:
                        get_answer = answer_area[0]['index']
                        logger.debug('题目%d  答案:%d' % (qust_index, answer_area[0]['index']))
                        if self.debug:
                            cv2.drawContours(empty_img, answer_area[0]['cnt'], -1, (255, 0, 0), -1)
                    else:
                        for ans in answer_area:
                            # 面积阀值，根据上次结果摆动
                            threshold = 0.18
                            if all_area:
                                threshold = threshold * 0.4 + (all_area / all_area_index + 0.18) * 0.6
                            if ans['area'] < ans['qnst_area'] * threshold:
                                all_area = all_area + (ans['area'] / ans['qnst_area'])
                                all_area_index = all_area_index + 1
                                # logger.debug("阀值%lf" % (all_area / all_area_index))
                                logger.debug('可能是%d为答案，答案面积%lf,题目面积%lf，占比%lf' % (
                                    ans['index'], ans['area'], ans['qnst_area'], ans['area'] / ans['qnst_area']))
                                if self.debug:
                                    cv2.drawContours(empty_img, ans['cnt'], -1, (255, 0, 0), -1)
                                get_answer = ans['index']
                                break
                            else:
                                logger.debug('可能%d不是答案，答案面积%lf,题目面积%lf，占比%lf' % (
                                    ans['index'], ans['area'], ans['qnst_area'], ans['area'] / ans['qnst_area']))
                    answer_list.append(answer_range[get_answer])
                    # 清空答案面积，复位索引
                    answer_area = []
                    ans_index = 1
                else:
                    ans_index = ans_index + 1
        if self.debug:
            cv2.imshow('mask', empty_img)
            cv2.waitKey(0)
        return answer_list

    def check_answer(self, answer_list, right_answer_list):
        right_list = []
        error_list = []
        empty_list = []
        for right_answer in right_answer_list:
            index = right_answer['id'] - 1
            if len(answer_list) < index:
                empty_list.append(right_answer['id'])
                continue
            if answer_list[index] == right_answer['answer']:
                right_list.append(right_answer['id'])
            else:
                error_list.append(right_answer['id'])
        logger.debug("答对了：%s" % str(right_list))
        logger.debug("答错了：%s" % str(error_list))
        logger.debug("没答题：%s" % str(empty_list))
        logger.debug("准确率：%lf" % ((len(right_list) / float(len(right_answer_list))) * 100.0))
        return right_list, error_list, empty_list
