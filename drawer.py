import cv2
from graph import Graph
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ШАБЛОНЫ ОТРИСОВКИ ГРАФИКОВ
# Очистка и подпись графика (вызывается в начале)
def cleaning_and_chart_graph(graph: Graph, title):
    graph.toolbar.home()  # Возвращаем зум в домашнюю позицию
    graph.toolbar.update()  # Очищаем стек осей
    # Очищаем график
    graph.axis.clear()
    # Задаем название графика
    graph.axis.set_title(title)


def cleaning_and_chart_graph_research(graph: Graph, x_label, y_label, title):
    graph.toolbar.home()  # Возвращаем зум в домашнюю позицию
    graph.toolbar.update()  # Очищаем стек осей
    # Очищаем график
    graph.axis.clear()
    # Задаем название осей
    graph.axis.set_xlabel(x_label)
    graph.axis.set_ylabel(y_label)
    # Задаем название графика
    graph.axis.set_title(title)


# Отрисовка (вызывается в конце)
def draw_graph(graph: Graph):
    # Убеждаемся, что все помещается внутри холста
    graph.figure.tight_layout()
    # Показываем новую фигуру в интерфейсе
    graph.canvas.draw()


# Отрисовка при отсутствии данных
def no_data(graph: Graph):
    graph.axis.text(0.5, 0.5, "Нет данных",
                    fontsize=14,
                    horizontalalignment='center',
                    verticalalignment='center')
    # Отрисовка, без подписи данных графиков
    draw_graph(graph)


# Класс художник. Имя холст (graph), рисует на нем данные
class Drawer:
    # ПАРАМЕТРЫ ГРАФИКОВ
    horizontal_axis_name_data = "Уровень шума [%]"
    vertical_axis_name_data = "Параметр восстановления [усл. ед.]"

    # ОТРИСОВКИ
    # (1) Цветной двухмерный график
    @staticmethod
    def graph_color_2d(
            graph: Graph,
            data
    ):

        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
        )

        # Рисуем график
        im = graph.axis.imshow(data, cmap='gray')
        # Если color bar нет - создаем, иначе обновляем
        if not graph.colorbar:
            divider = make_axes_locatable(graph.axis)
            cax = divider.append_axes("right", "10%", pad="3%")
            graph.colorbar = graph.figure.colorbar(im, orientation='vertical', cax=cax)
        else:
            graph.colorbar.update_normal(im)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)

    # (2) Цветная картинка
    @staticmethod
    def image_color_2d(
            graph: Graph,
            data
    ):
        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
        )

        # Рисуем график
        im = graph.axis.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        # Если color bar нет - создаем, иначе обновляем
        if not graph.colorbar:
            divider = make_axes_locatable(graph.axis)
            cax = divider.append_axes("right", "10%", pad="3%")
            graph.colorbar = graph.figure.colorbar(im, orientation='vertical', cax=cax)
        else:
            graph.colorbar.update_normal(im)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)

    # (3) Серая картинка
    @staticmethod
    def image_gray_2d(
            graph: Graph,
            data
    ):

        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
        )

        # Рисуем график
        im = graph.axis.imshow(data, cmap='gray')
        # Если color bar нет - создаем, иначе обновляем
        if not graph.colorbar:
            divider = make_axes_locatable(graph.axis)
            cax = divider.append_axes("right", "10%", pad="3%")
            graph.colorbar = graph.figure.colorbar(im, orientation='vertical', cax=cax)
        else:
            graph.colorbar.update_normal(im)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)

    @staticmethod
    def graph_2d(
            graph: Graph,
            x, y
    ):

        graph.reset()

        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph_research(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
            # Подпись осей
            x_label=Drawer.horizontal_axis_name_data, y_label=Drawer.vertical_axis_name_data
        )

        graph.axis.plot(x, y)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)
