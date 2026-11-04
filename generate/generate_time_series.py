def generate_time_series(
    length: int,
    trend_type: Optional[str] = None,
    trend_params: Optional[Dict[str, float]] = None,
    seasonal_periods: Optional[List[int]] = None,
    seasonal_amplitudes: Optional[List[float]] = None,
    seasonal_phases: Optional[List[float]] = None,
    noise_type: Optional[str] = None,
    noise_params: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Генерирует синтетический временной ряд с заданными компонентами.

    Функция создает временной ряд как сумму трех компонент:
        y(t) = trend(t) + seasonality(t) + noise(t)

    Parameters
    ----------
    length : int
        Длина генерируемого временного ряда (количество точек).

    trend_type : str, optional
        Тип тренда. Возможные значения:
        - 'linear'   : линейный тренд
        - 'power'    : степенной тренд
        - 'log'      : логарифмический тренд
        - None       : без тренда

    trend_params : dict, optional
        Параметры тренда. Должен содержать ключи в зависимости от типа:
        - Для 'linear': {'slope': float} - наклон прямой
        - Для 'power' : {'a': float, 'p': float} - коэффициент и степень (a * x^p)
        - Для 'log'   : {'a': float} - коэффициент (a * ln(x))
        Если не указан, используются значения по умолчанию.

    seasonal_periods : list of int, optional
        Список периодов сезонных гармоник.
        Например: [24, 168] для суточной и недельной сезонности.
        Если None, сезонность отсутствует.

    seasonal_amplitudes : list of float, optional
        Список амплитуд для каждой гармоники. Должен совпадать по длине с seasonal_periods.
        Если None, все амплитуды равны 1.0.

    seasonal_phases : list of float, optional
        Список фазовых сдвигов (в радианах) для каждой гармоники.
        Должен совпадать по длине с seasonal_periods.
        Если None, все фазы равны 0.

    noise_type : str, optional
        Тип шума. Возможные значения:
        - 'gaussian' : гауссовский шум
        - 'laplace'  : шум Лапласа
        - None       : без шума

    noise_params : dict, optional
        Параметры шума:
        - Для 'gaussian': {'mean': float, 'std': float} - среднее и стандартное отклонение
        - Для 'laplace' : {'loc': float, 'scale': float} - локация и масштаб
        Если не указан, используются значения по умолчанию.

    seed : int, optional
        Seed для генератора случайных чисел (для воспроизводимости).

    Returns
    -------
    np.ndarray
        Сгенерированный временной ряд длины `length`.

    Examples
    --------
    >>> # Ряд с линейным трендом, суточной сезонностью и гауссовским шумом
    >>> ts = generate_time_series(
    ...     length=1000,
    ...     trend_type='linear',
    ...     trend_params={'slope': 0.01},
    ...     seasonal_periods=[24],
    ...     seasonal_amplitudes=[2.0],
    ...     noise_type='gaussian',
    ...     noise_params={'mean': 0, 'std': 0.5},
    ...     seed=42
    ... )
    """

    # Устанавливаем seed для воспроизводимости
    if seed is not None:
        np.random.seed(seed)

    # Создаем временную ось (начинаем с 1, чтобы избежать log(0))
    t = np.arange(1, length + 1, dtype=float)

    # Инициализируем ряд нулями
    series = np.zeros(length)

    # --- 1. Добавляем тренд ---
    if trend_type is not None:
        if trend_params is None:
            trend_params = {}

        if trend_type == 'linear':
            slope = trend_params.get('slope', 0.01)
            trend = slope * t
            series += trend

        elif trend_type == 'power':
            a = trend_params.get('a', 0.001)
            p = trend_params.get('p', 1.5)
            trend = a * (t ** p)
            series += trend

        elif trend_type == 'log':
            a = trend_params.get('a', 1.0)
            trend = a * np.log(t)
            series += trend

        else:
            raise ValueError(
                f"Неизвестный тип тренда: '{trend_type}'. "
                f"Доступные: 'linear', 'power', 'log'"
            )

    # --- 2. Добавляем сезонность ---
    if seasonal_periods is not None:
        n_harmonics = len(seasonal_periods)

        # Устанавливаем амплитуды по умолчанию
        if seasonal_amplitudes is None:
            seasonal_amplitudes = [1.0] * n_harmonics
        elif len(seasonal_amplitudes) != n_harmonics:
            raise ValueError(
                f"Длина seasonal_amplitudes ({len(seasonal_amplitudes)}) "
                f"не совпадает с длиной seasonal_periods ({n_harmonics})"
            )

        # Устанавливаем фазы по умолчанию
        if seasonal_phases is None:
            seasonal_phases = [0.0] * n_harmonics
        elif len(seasonal_phases) != n_harmonics:
            raise ValueError(
                f"Длина seasonal_phases ({len(seasonal_phases)}) "
                f"не совпадает с длиной seasonal_periods ({n_harmonics})"
            )

        # Суммируем все гармоники
        seasonality = np.zeros(length)
        for period, amp, phase in zip(seasonal_periods, seasonal_amplitudes, seasonal_phases):
            # Используем синусоиду с заданным периодом, амплитудой и фазой
            harmonic = amp * np.sin(2 * np.pi * t / period + phase)
            seasonality += harmonic

        series += seasonality

    # --- 3. Добавляем шум ---
    if noise_type is not None:
        if noise_params is None:
            noise_params = {}

        if noise_type == 'gaussian':
            mean = noise_params.get('mean', 0.0)
            std = noise_params.get('std', 0.1)
            noise = np.random.normal(mean, std, length)

        elif noise_type == 'laplace':
            loc = noise_params.get('loc', 0.0)
            scale = noise_params.get('scale', 0.1)
            noise = np.random.laplace(loc, scale, length)

        else:
            raise ValueError(
                f"Неизвестный тип шума: '{noise_type}'. "
                f"Доступные: 'gaussian', 'laplace'"
            )

        series += noise

    return series
