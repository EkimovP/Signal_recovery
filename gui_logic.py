from gui import Ui_Dialog
from FastFourierTransform import FFT
from PyQt5 import QtCore
from graph import Graph
from drawer import Drawer as drawer

from PyQt5.QtWidgets import QFileDialog
import random
import cv2
import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


# Гауссовы купола
def dome_2d(amplitude, x, x_0, sigma_x, y, y_0, sigma_y):
    return amplitude * math.exp(-(
            (
                    ((x - x_0) * (x - x_0)) /
                    (2 * sigma_x * sigma_x)
            ) + (
                    ((y - y_0) * (y - y_0)) /
                    (2 * sigma_y * sigma_y)
            )))


# Перевод цветного изображения в серое
def black_white_image(color_picture):
    height, width, _ = color_picture.shape
    gray_image = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel = color_picture[i, j]
            gray_image[i, j] = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
    maximum_intensity = np.max(gray_image)
    multiplier = 255 / maximum_intensity
    gray_image = gray_image * multiplier
    return gray_image


# Функция (шума) для нормального распределения по Гауссу
def uniform_distribution():
    repeat = 12
    val = 0
    for i in range(repeat):
        val += random.random()  # значение от 0.0 до 1.0
    return val / repeat


# КЛАСС АЛГОРИТМА ПРИЛОЖЕНИЯ
class GuiProgram(Ui_Dialog):

    def __init__(self, dialog):
        # Создаем окно
        Ui_Dialog.__init__(self)
        # Дополнительные функции окна
        dialog.setWindowFlags(  # Передаем флаги создания окна
            QtCore.Qt.WindowCloseButtonHint |  # Закрытие
            QtCore.Qt.WindowMaximizeButtonHint |  # Во весь экран (развернуть)
            QtCore.Qt.WindowMinimizeButtonHint  # Свернуть
        )
        self.setupUi(dialog)  # Устанавливаем пользовательский интерфейс

        # ПОЛЯ КЛАССА
        # Параметры 1 графика - Исходное изображение
        self.graph_1 = Graph(
            layout=self.layout_plot,
            widget=self.widget_plot,
            name_graphics="График №1. Исходное изображение"
        )
        # Параметры 2 графика - Аппаратная функция
        self.graph_2 = Graph(
            layout=self.layout_plot_1,
            widget=self.widget_plot_1,
            name_graphics="График №2. Аппаратная функция"
        )
        # Параметры 3 графика - Спектр исходного изображения
        self.graph_3 = Graph(
            layout=self.layout_plot_2,
            widget=self.widget_plot_2,
            name_graphics="График №3. Спектр исходного изображения"
        )
        # Параметры 4 графика - Спектр аппаратной функции
        self.graph_4 = Graph(
            layout=self.layout_plot_3,
            widget=self.widget_plot_3,
            name_graphics="График №4. Спектр аппаратной функции"
        )
        # Изображения этапов обработки
        self.color_picture = None
        self.gray_picture = None
        self.original_picture = None
        self.noise_image = None
        self.kernel_image = None
        self.spectrum_image = None
        self.spectrum_core = None
        self.module_spectrum_repositioned_all = None
        self.blurred_image = None
        self.reconstructed_signal = None

        # ДЕЙСТВИЯ ПРИ ВКЛЮЧЕНИИ
        # Смена режима отображения картинки
        self.radioButton_color_picture.clicked.connect(self.change_picture_to_colored)
        self.radioButton_gray_picture.clicked.connect(self.change_picture_to_gray)
        # Выбрана картинка или график
        self.radioButton_generation_domes.clicked.connect(self.creating_dome)
        self.radioButton_loading_pictures.clicked.connect(self.display_picture)

        # Алгоритм обратки
        # Создание куполов
        self.pushButton_display_domes.clicked.connect(self.creating_dome)
        # Загрузка картинки
        self.pushButton_loading_pictures.clicked.connect(self.load_image)
        # Создание аппаратной функции
        self.pushButton_creating_hardware_function.clicked.connect(self.creating_hardware_function)
        # Построение спектра изображения и аппаратной функции
        self.pushButton_construction_spectrums.clicked.connect(self.spectrum_image_and_core)
        # Размытие изображения
        self.pushButton_start_processing.clicked.connect(self.convolution)
        # Добавление шума
        self.pushButton_display_noise.clicked.connect(self.noise_convolution)
        # Восстановление сигнала
        self.pushButton_start_signal_recovery.clicked.connect(self.signal_recovery)

    # ОБРАБОТКА ИНТЕРФЕЙСА
    # Смена режима отображения изображения
    # Выбрано отображение цветного изображения
    def change_picture_to_colored(self, state):
        if state and self.color_picture is not None:
            drawer.image_color_2d(self.graph_1, self.color_picture)

    # Выбрано отображение серого изображения
    def change_picture_to_gray(self, state):
        if state and self.original_picture is not None:
            drawer.image_gray_2d(self.graph_1, self.original_picture)

    # Отобразить изображение
    def display_picture(self):
        # Изображения нет - не отображаем
        if self.color_picture is None:
            return
        # Проверяем вид отображаемого изображения
        if self.radioButton_color_picture.isChecked():
            drawer.image_color_2d(self.graph_1, self.color_picture)
        else:
            drawer.image_gray_2d(self.graph_1, self.original_picture)

    # Построение спектра
    def building_spectrum(self, original_image_for_the_spectrum, the_center=False):
        # Переводим изображение в комплексное
        complex_image = np.array(original_image_for_the_spectrum, dtype=complex)
        # Считаем спектр
        original_image_for_the_spectrum = FFT.matrix_fft(complex_image)
        # Берем модуль, для отображения
        module_picture_spectrum = abs(original_image_for_the_spectrum)
        module_picture_spectrum[0, 0] = 0
        # Строим спектр в центре, если это необходимо
        if the_center:
            module_picture_spectrum = self.building_spectrum_in_the_center(module_picture_spectrum)
        return module_picture_spectrum, original_image_for_the_spectrum

    # Построение спектра в центру
    def building_spectrum_in_the_center(self, module_picture_spectrum):
        # Матрица со спектром посередине
        height, width = module_picture_spectrum.shape
        middle_h = height // 2
        middle_w = width // 2
        module_spectrum_repositioned_image = np.zeros((height, width))
        # Меняем по главной диагонали
        module_spectrum_repositioned_image[0:middle_h, 0:middle_w] = \
            module_picture_spectrum[middle_h:height, middle_w:width]
        module_spectrum_repositioned_image[middle_h:height, middle_w:width] = \
            module_picture_spectrum[0:middle_h, 0:middle_w]
        # Меняем по главной диагонали
        module_spectrum_repositioned_image[middle_h:height, 0:middle_w] = \
            module_picture_spectrum[0:middle_h, middle_w:width]
        module_spectrum_repositioned_image[0:middle_h, middle_w:width] = \
            module_picture_spectrum[middle_h:height, 0:middle_w]
        return module_spectrum_repositioned_image

    # АЛГОРИТМ РАБОТЫ ПРОГРАММЫ
    # (1 Гауссовы купола) Вычислить график
    def creating_dome(self):
        # Запрашиваем размер области
        width_area = int(self.lineEdit_width_area.text())
        height_area = int(self.lineEdit_height_area.text())
        # Запрашиваем параметры куполов
        # Первый купол
        amplitude_1 = float(self.lineEdit_amplitude_1.text())
        x0_1 = float(self.lineEdit_x0_1.text())
        sigma_x_1 = float(self.lineEdit_sigma_x_1.text())
        y0_1 = float(self.lineEdit_y0_1.text())
        sigma_y_1 = float(self.lineEdit_sigma_y_1.text())
        # Второй купол
        amplitude_2 = float(self.lineEdit_amplitude_2.text())
        x0_2 = float(self.lineEdit_x0_2.text())
        sigma_x_2 = float(self.lineEdit_sigma_x_2.text())
        y0_2 = float(self.lineEdit_y0_2.text())
        sigma_y_2 = float(self.lineEdit_sigma_y_2.text())
        # Третий купол
        amplitude_3 = float(self.lineEdit_amplitude_3.text())
        x0_3 = float(self.lineEdit_x0_3.text())
        sigma_x_3 = float(self.lineEdit_sigma_x_3.text())
        y0_3 = float(self.lineEdit_y0_3.text())
        sigma_y_3 = float(self.lineEdit_sigma_y_3.text())
        # Создаем пустую матрицу пространства
        self.original_picture = np.zeros((height_area, width_area))
        # Для каждой точки матрицы считаем сумму куполов
        for x in range(width_area):
            for y in range(height_area):
                self.original_picture[y, x] = dome_2d(amplitude_1, x, x0_1, sigma_x_1, y, y0_1, sigma_y_1) + \
                                              dome_2d(amplitude_2, x, x0_2, sigma_x_2, y, y0_2, sigma_y_2) + \
                                              dome_2d(amplitude_3, x, x0_3, sigma_x_3, y, y0_3, sigma_y_3)
        # Выводим изображение
        drawer.graph_color_2d(self.graph_1, self.original_picture)

    # (1 изображение) Загрузить изображение
    def load_image(self):
        # Вызов окна выбора файла
        filename, filetype = QFileDialog.getOpenFileName(None,
                                                         "Выбрать файл изображения",
                                                         ".",
                                                         "All Files(*)")
        # filename = "image11.png"
        # Загружаем изображение
        self.color_picture = cv2.imread(filename, cv2.IMREAD_COLOR)
        # Конвертируем в серое
        self.gray_picture = black_white_image(self.color_picture)
        self.original_picture = self.gray_picture
        # Отображаем изображение
        self.display_picture()

    # # (2) Накладываем шум
    # def noise(self):
    #     # Нет исходны данных - сброс
    #     if self.original_picture is None:
    #         return
    #     self.noise_image = self.original_picture.copy()
    #     # Считаем энергию изображения
    #     energy_pictures = 0
    #     for x in self.noise_image:
    #         for y in x:
    #             energy_pictures += y * y
    #     # Создаем изображение чисто шума
    #     height, width = self.noise_image.shape
    #     picture_noise = np.zeros((height, width))
    #     energy_noise = 0
    #     for x in range(width):
    #         for y in range(height):
    #             val = uniform_distribution()
    #             # Записываем пиксель шума
    #             picture_noise[y, x] = val
    #             # Копим энергию шума
    #             energy_noise += val * val
    #     # Запрашиваем процент шума
    #     noise_percentage = float(self.lineEdit_noise.text()) / 100
    #     # Считаем коэффициент/множитель шума
    #     noise_coefficient = math.sqrt(noise_percentage *
    #                                   (energy_pictures / energy_noise))
    #     # К пикселям изображения добавляем пиксель шума
    #     for x in range(width):
    #         for y in range(height):
    #             self.noise_image[y, x] += int(noise_coefficient * picture_noise[y, x])
    #     # Отображаем итог
    #     # выбираем график или изображение
    #     if self.radioButton_loading_pictures.isChecked():
    #         drawer.image_gray_2d(self.graph_1, self.noise_image)
    #     else:
    #         drawer.graph_color_2d(self.graph_1, self.noise_image)

    # (2) Создаем аппаратную функцию (ядро)
    def creating_hardware_function(self):
        # Запрашиваем параметры аппаратной функции в виде купола
        amplitude_kernels = float(self.lineEdit_amplitude_kernels.text())
        x0_kernels = float(self.lineEdit_x0_kernels.text())
        sigma_x_kernels = float(self.lineEdit_sigma_x_kernels.text())
        y0_kernels = float(self.lineEdit_y0_kernels.text())
        sigma_y_kernels = float(self.lineEdit_sigma_y_kernels.text())
        # Создаем пустую матрицу пространства
        height_area, width_area = self.original_picture.shape
        self.kernel_image = np.zeros((height_area, width_area))
        for x in range(width_area):
            for y in range(height_area):
                self.kernel_image[y, x] = dome_2d(amplitude_kernels,
                                                  x, x0_kernels, sigma_x_kernels,
                                                  y, y0_kernels, sigma_y_kernels)
        drawer.image_gray_2d(self.graph_2, self.kernel_image)

    # (3) Строим спектр изображения и аппаратной функции (ядра)
    def spectrum_image_and_core(self):
        if self.original_picture is None and self.kernel_image is None:
            return
        # Строим спектр изображения
        module_picture_spectrum, self.spectrum_image = self.building_spectrum(self.original_picture, True)
        # Отображаем спектр изображения
        drawer.image_gray_2d(self.graph_3, module_picture_spectrum)
        # Строим спектр аппаратной функции
        module_core_spectrum, self.spectrum_core = self.building_spectrum(self.kernel_image, True)
        # Отображаем спектр аппаратной функции
        drawer.image_gray_2d(self.graph_4, module_core_spectrum)

    # (4) Производим свертку аппаратной функции (ядра) с исходным изображением
    # через перемножение их комплексных спектров
    def convolution(self):
        # Свертка сигнала и фильтра
        self.blurred_image = self.spectrum_image * self.spectrum_core
        # Берем модуль, для отображения
        module_all_spectrum = abs(self.blurred_image)
        module_all_spectrum[0, 0] = 0
        # Отображаем спектр свертки
        # self.graph_1.name_graphics = "График 5. Свертка сигнала и фильтра"
        # drawer.image_gray_2d(self.graph_1, module_all_spectrum)
        # self.graph_1.name_graphics = "График №1. Исходное изображение"
        # Отображаем спектр свертки в центре
        self.module_spectrum_repositioned_all = self.building_spectrum_in_the_center(module_all_spectrum)
        self.graph_1.name_graphics = "График 5. Свертка сигнала и фильтра"
        drawer.image_gray_2d(self.graph_1, self.module_spectrum_repositioned_all)
        self.graph_1.name_graphics = "График №1. Исходное изображение"
        # Отображаем изображение свертки
        convolution_image = (np.fft.ifft2(self.spectrum_image * self.spectrum_core)).real
        self.graph_2.name_graphics = "График 6. Размытое изображение"
        drawer.image_gray_2d(self.graph_2, convolution_image)
        self.graph_2.name_graphics = "График №2. Аппаратная функция"

    # (5) Добавляем шум к свертке
    def noise_convolution(self):
        # Нет исходны данных - сброс
        if self.blurred_image is None:
            return
        self.noise_image = self.blurred_image.copy()
        # Считаем энергию изображения
        energy_pictures = 0
        for x in self.noise_image:
            for y in x:
                energy_pictures += y * y
        # Создаем изображение чисто шума
        height, width = self.noise_image.shape
        picture_noise = np.zeros((height, width))
        energy_noise = 0
        for x in range(width):
            for y in range(height):
                val = uniform_distribution()
                # Записываем пиксель шума
                picture_noise[y, x] = val
                # Копим энергию шума
                energy_noise += val * val
        # Запрашиваем процент шума
        noise_percentage = float(self.lineEdit_noise.text()) / 100
        # Считаем коэффициент/множитель шума
        noise_coefficient = math.sqrt(noise_percentage * (energy_pictures.real / energy_noise))
        print(noise_coefficient)
        print(picture_noise)
        # К пикселям изображения добавляем пиксель шума
        for x in range(width):
            for y in range(height):
                self.noise_image[y, x] += int(noise_coefficient * picture_noise[y, x])
        # Отображаем итог
        module_picture_spectrum_convolution = abs(self.noise_image)
        module_picture_spectrum_convolution[0, 0] = 0
        # Отображаем спектр свертки с шумом в центре
        module_picture_spectrum_convolution = self.building_spectrum_in_the_center(module_picture_spectrum_convolution)
        self.graph_2.name_graphics = "График 7. Свертка сигнала и фильтра c шумом"
        drawer.image_gray_2d(self.graph_2, module_picture_spectrum_convolution)
        self.graph_2.name_graphics = "График №2. Аппаратная функция"

    # (6) Восстанавливаем исходное изображение
    def signal_recovery(self):
        num_iterations = int(self.lineEdit_number_iterations.text())  # Количество итераций
        epsilon_break = float(self.lineEdit_division.text())  # Порог для восстановления сигнала
        # Копируем спектр свертки
        self.reconstructed_signal = self.noise_image.copy()
        # Шаг 1. Выполнить прямое ПФ аппаратной функции
        fft_kernel = np.fft.fft2(self.kernel_image)
        # Шаг 2. Выполнить прямое ПФ изображения
        fft_image = np.fft.fft2(self.original_picture)
        # Шаг 3. Производится деление Фурье-образов, где это возможно.
        # Если деление не возможно, оставляются отсчёты Фурье-образа свёртки,
        # или производится зануление соответствующих отсчётов
        # Инициализируем массив для результата
        result = np.zeros_like(fft_image, dtype=np.complex128)
        # Выполняем деление Фурье-образов, где это возможно, иначе зануляем
        list_of_zeroed_indices_i = []
        list_of_zeroed_indices_j = []
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                # if fft_kernel[i, j].real == 0 and fft_kernel[i, j].imag == 0:
                if (fft_kernel[i, j].real < epsilon_break and fft_kernel[i, j].imag < epsilon_break
                        and self.reconstructed_signal[i, j].real < epsilon_break
                        and self.reconstructed_signal[i, j].imag < epsilon_break):
                    result[i, j] = 0. + 0.j  # self.reconstructed_signal[i, j]
                    list_of_zeroed_indices_i.append(i)
                    list_of_zeroed_indices_j.append(j)
                else:
                    result[i, j] = self.reconstructed_signal[i, j] / fft_kernel[i, j]
        for _ in range(num_iterations):
            # Шаг 4. Выполняется обратное ПФ -> временная область
            # Только реальная часть
            result_image = np.fft.ifft2(result)
            # Шаг 5. В восстановленном изображении зануляются отрицательные элементы
            result_image = np.maximum(result_image, 0)
            # Шаг 6. Выполняется прямое ПФ -> частотная область
            fft_result_image = np.fft.fft2(result_image.astype("complex"))
            # Шаг 7. В точках, где произведено деление, значение Фурье-образа заменяются результатом деления.
            # В остальных точках значения текущих отсчётов Фурье-образа сохраняются
            for i in range(len(list_of_zeroed_indices_i)):
                result[list_of_zeroed_indices_i[i], list_of_zeroed_indices_j[i]] = \
                    fft_result_image[list_of_zeroed_indices_i[i], list_of_zeroed_indices_j[i]]
            # Шаг 8. Переход к шагу 4
        # Отображаем восстановленный график
        self.graph_3.name_graphics = "График №6. Восстановленное изображение"
        # drawer.image_gray_2d(self.graph_3, np.abs(np.fft.ifft2(result)))
        drawer.image_gray_2d(self.graph_3, np.fft.ifft2(result).real)
        self.graph_3.name_graphics = "График №3. Спектр исходного изображения"
        # Отображаем исходный график
        self.graph_4.name_graphics = "График №1. Исходное изображение"
        drawer.image_gray_2d(self.graph_4, np.abs(self.original_picture))
        self.graph_4.name_graphics = "График №4. Спектр аппаратной функции"
