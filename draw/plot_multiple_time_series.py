def plot_multiple_time_series(
    series_list: List[np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = "Сравнение временных рядов",
    xlabel: str = "Время",
    ylabel: str = "Значение",
    figsize: Tuple[int, int] = (14, 6),
    colors: Optional[List[str]] = None,
    line_alpha: float = 0.7,
    point_size: float = 2.0,
    point_alpha: float = 0.6,
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Визуализирует несколько временных рядов на одном графике для сравнения.

    Parameters
    ----------
    series_list : list of np.ndarray
        Список временных рядов для отображения.

    labels : list of str, optional
        Список подписей для каждого ряда. Должен совпадать по длине с series_list.

    title : str, default="Сравнение временных рядов"
        Заголовок графика.

    xlabel : str, default="Время"
        Подпись оси X.

    ylabel : str, default="Значение"
        Подпись оси Y.

    figsize : tuple, default=(14, 6)
        Размер фигуры.

    colors : list of str, optional
        Список цветов для каждого ряда. Если None, используются цвета по умолчанию.

    line_alpha : float, default=0.7
        Прозрачность линий.

    point_size : float, default=2.0
        Размер точек.

    point_alpha : float, default=0.6
        Прозрачность точек.

    show_grid : bool, default=True
        Показывать ли сетку.

    grid_alpha : float, default=0.3
        Прозрачность сетки.

    save_path : str, optional
        Путь для сохранения.

    show : bool, default=True
        Показывать ли график.

    Returns
    -------
    None

    Examples
    --------
    >>> ts1 = generate_time_series(length=200, seasonal_periods=[24])
    >>> ts2 = generate_time_series(length=200, seasonal_periods=[24, 168])
    >>> plot_multiple_time_series([ts1, ts2], labels=["Суточная", "Суточная + недельная"])
    """

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=figsize)

    # Если цвета не указаны, используем стандартную палитру
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(series_list)))

    # Если подписи не указаны, генерируем
    if labels is None:
        labels = [f"Ряд {i+1}" for i in range(len(series_list))]

    # Отрисовываем каждый ряд
    time = np.arange(len(series_list[0]))

    for i, (series, label, color) in enumerate(zip(series_list, labels, colors)):
        ax.plot(time, series, color=color, alpha=line_alpha, linewidth=1.5, label=label)
        ax.scatter(time, series, color=color, s=point_size, alpha=point_alpha, zorder=5)

    # Настройка подписей и заголовка
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Сетка
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)

    # Легенда
    ax.legend(loc='best', fontsize=10)

    # Автоматическая настройка отступов
    plt.tight_layout()

    # Сохранение
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен в: {save_path}")

    # Отображение
    if show:
        plt.show()
    else:
        plt.close(fig)
