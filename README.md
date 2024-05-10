### [Восстановление сигнала из свертки]()

<img src=" " align="right" width=500>

Данный проект предназначен для исследования итерационного алгоритма восстановления.
<br><br>
В программе можно изобразить гауссовы купола, задав их параметры, или загрузить изображение.
После чего необходимо построить аппаратную функцию (ядро). 
Затем строится спектр исходного изображения и аппаратной функции с помощью метода 
быстрого преобразования Фурье (БПФ) и происходит свертка сигнала и фильтра, как процесс перемножения их
спектров. Затем можно добавить шум к свертке. После чего идет процесс восстановления.<br> 
Рассчитывается параметр восстановления, который позволяет оценить работу метода и сравнить исходное и 
восстановленное изображения.<br>
Проводится исследование зависимости параметра восстановления от заданного диапазона шума.

***Алгоритм программы***

1. **Ввод данных:** генерация гауссовых куполов или загрузка изображения. 
Так как используется БПФ, то необходимо брать исходное изображение *степени двойки*.
2. **Построение аппаратной функции (ядра).** 
3. **Построение спектров исходного изображения и аппаратной функции.**
4. **Свертка исходного изображения с аппаратной функцией.**
5. **Добавление шума к свертке.**
6. **Восстановление изображения с помощью итерационного алгоритма.**
7. **Расчет параметра восстановления.**
8. **Исследование зависимости параметра восстановления от шума.**
