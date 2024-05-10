import numpy as np
import math


# Переводит действительное значение в комплексное
# Мнимую реальная = действительной; мнимая = 0
def actual_to_complex(actual_val):
    complex_val = actual_val.astype(complex)
    return complex_val


# Вычисление поворачивающего модуля e^(-i*2*PI*k/N)
def turning_module_exp(k, n):
    if k % n == 0:
        return 1
    argument = -2 * math.pi * k / n
    return complex(math.cos(argument), math.sin(argument))


class FFT:
    # Прямое Фурье Преобразование
    @staticmethod
    def fft(in_x):
        size_in_x = in_x.size
        half_size = size_in_x // 2
        out_x = 1j * np.arange(size_in_x)

        if size_in_x == 2:
            out_x[0] = in_x[0] + in_x[1]
            out_x[1] = in_x[0] - in_x[1]

        else:
            x_even = 1j * np.arange(half_size)  # четные
            x_odd = 1j * np.arange(half_size)  # не четные

            # Производим разделение на чет и не чет
            for i in range(half_size):
                x_even[i] = in_x[2 * i]
                x_odd[i] = in_x[2 * i + 1]

            # Отправляем чет и не чет в рекурсию
            x_even = FFT.fft(x_even)
            x_odd = FFT.fft(x_odd)

            # Считаем произведение поворачивающих модулей с нечетными
            right_syllable = 1j * np.arange(half_size)
            for i in range(half_size):
                right_syllable[i] = turning_module_exp(i, size_in_x) * x_odd[i]

            # Собираем обратно
            # левая половина: X_even[i] + w(i, N) * X_odd[i]
            # правая половина: X_even[i] - w(i, N) * X_odd[i]
            # Выход равен двум частям

            out_x = np.concatenate((
                x_even + right_syllable,
                x_even - right_syllable
            ), axis=None)

        #         right = 1j * np.arange(half_size) # четные
        #         left = 1j * np.arange(half_size) # не четные
        #         for i in range (half_size):
        #             left[i] = x_even[i] + w_coefficient[i]
        #         for i in range (half_size):
        #             right[i] = x_even[i] - turning_module_exp(i, size_in_x) * x_odd[i]

        #         out_x = np.concatenate((
        #             left,
        #             right
        #         ), axis=None)

        return out_x

    # Обратное Фурье Преобразование
    @staticmethod
    def my_reverse_fft(in_x):
        x = in_x.conjugate()  # (1) Комплексно сопряженное
        x = FFT.fft(x)  # (2) Фурье
        x = x.conjugate()  # (3) Комплексно сопряженное
        x = x / x.size  # (4) Делим re и im часть на кол-во отсчетов
        return x

    @staticmethod
    def matrix_fft(complex_matrix):
        height, width = complex_matrix.shape

        final_matrix = np.zeros((height, width), dtype=complex)
        for i in range(height):
            final_matrix[i] = FFT.fft(complex_matrix[i])

        final_matrix = final_matrix.transpose()

        for i in range(width):
            final_matrix[i] = FFT.fft(final_matrix[i])

        final_matrix = final_matrix.transpose()

        return (final_matrix)

    @staticmethod
    def matrix_fft_reverse(complex_matrix):
        height, width = complex_matrix.shape

        final_matrix = np.zeros((height, width), dtype=complex)
        for i in range(height):
            final_matrix[i] = FFT.my_reverse_fft(complex_matrix[i])

        final_matrix = final_matrix.transpose()

        for i in range(width):
            final_matrix[i] = FFT.my_reverse_fft(final_matrix[i])

        final_matrix = final_matrix.transpose()

        return (final_matrix)
