def create_missing_values(
    series: np.ndarray,
    missing_type: str = 'point',
    point_percentage: Optional[float] = None,
    continuous_max_length: Optional[int] = None,
    continuous_percentage: Optional[float] = None,
    continuous_indices: Optional[List[List[int]]] = None,
    mixed_point_percentage: Optional[float] = None,
    mixed_continuous_max_length: Optional[int] = None,
    mixed_continuous_percentage: Optional[float] = None,
    mixed_continuous_indices: Optional[List[List[int]]] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создает пропуски во временном ряду различными способами.

    Функция поддерживает три режима работы:
    1. 'point' - точечные пропуски (случайные одиночные точки)
    2. 'continuous' - продолжительные пропуски (блоки пропусков заданной длины)
    3. 'mixed' - смешанные пропуски (комбинация точечных и продолжительных)

    Parameters
    ----------
    series : np.ndarray
        Входной временной ряд (1D массив).

    missing_type : str, default='point'
        Тип создаваемых пропусков:
        - 'point'      : только точечные пропуски
        - 'continuous' : только продолжительные пропуски
        - 'mixed'      : смешанные пропуски (точки + блоки)

    point_percentage : float, optional
        Процент точечных пропусков от длины ряда (0-100).
        Используется при missing_type='point'.

    continuous_max_length : int, optional
        Максимальная длина одного блока непрерывных пропусков.
        Используется при missing_type='continuous' или 'mixed'.

    continuous_percentage : float, optional
        Процент точек, которые будут удалены в блоках непрерывных пропусков (0-100).
        Используется при missing_type='continuous' с режимом 1.

    continuous_indices : list of lists, optional
        Список списков с индексами для удаления.
        Например: [[10, 11, 12], [25, 26, 27, 28]]
        Используется при missing_type='continuous' с режимом 2.

    mixed_point_percentage : float, optional
        Процент точечных пропусков от длины ряда (0-100).
        Используется при missing_type='mixed'.

    mixed_continuous_max_length : int, optional
        Максимальная длина блока непрерывных пропусков для смешанного режима.
        Используется при missing_type='mixed' с режимом 1 для непрерывных.

    mixed_continuous_percentage : float, optional
        Процент точек для непрерывных пропусков в смешанном режиме.
        Используется при missing_type='mixed' с режимом 1 для непрерывных.

    mixed_continuous_indices : list of lists, optional
        Список списков с индексами для непрерывных пропусков в смешанном режиме.
        Используется при missing_type='mixed' с режимом 2 для непрерывных.

    seed : int, optional
        Seed для воспроизводимости.

    Returns
    -------
    tuple (series_with_nan, mask)
        - series_with_nan : np.ndarray
            Временной ряд с пропущенными значениями (NaN).
        - mask : np.ndarray
            Бинарная маска (1 - валидные данные, 0 - пропуски).

    Raises
    ------
    ValueError
        Если переданы несовместимые параметры или неверный тип пропусков.

    Examples
    --------
    >>> # 1. Точечные пропуски (20% точек)
    >>> series_nan, mask = create_missing_values(
    ...     series=ts,
    ...     missing_type='point',
    ...     point_percentage=20,
    ...     seed=42
    ... )

    >>> # 2. Непрерывные пропуски (блоки до 10 дней, удаляем 15% точек)
    >>> series_nan, mask = create_missing_values(
    ...     series=ts,
    ...     missing_type='continuous',
    ...     continuous_max_length=10,
    ...     continuous_percentage=15,
    ...     seed=42
    ... )

    >>> # 3. Смешанные пропуски (10% точечных + блоки до 7 дней, удаляем 10% точек)
    >>> series_nan, mask = create_missing_values(
    ...     series=ts,
    ...     missing_type='mixed',
    ...     mixed_point_percentage=10,
    ...     mixed_continuous_max_length=7,
    ...     mixed_continuous_percentage=10,
    ...     seed=42
    ... )
    """

    # Устанавливаем seed
    if seed is not None:
        np.random.seed(seed)

    length = len(series)

    # Копируем ряд, чтобы не изменять исходный
    series_with_nan = series.copy()

    # Инициализируем маску (1 - данные есть, 0 - пропуск)
    mask = np.ones(length, dtype=bool)

    # --- 1. Точечные пропуски ---
    if missing_type == 'point':
        if point_percentage is None:
            raise ValueError("Для missing_type='point' необходимо указать point_percentage")

        if not (0 <= point_percentage <= 100):
            raise ValueError(f"point_percentage должен быть в диапазоне [0, 100], получено {point_percentage}")

        # Количество точек для удаления
        n_missing = int(length * point_percentage / 100)

        # Случайно выбираем индексы
        missing_indices = np.random.choice(length, size=n_missing, replace=False)

        # Устанавливаем NaN и обновляем маску
        series_with_nan[missing_indices] = np.nan
        mask[missing_indices] = False

    # --- 2. Продолжительные пропуски ---
    elif missing_type == 'continuous':
        # Проверяем, какой режим используется
        if continuous_indices is not None:
            # Режим 2: индексы заданы явно
            for indices in continuous_indices:
                # Проверяем, что индексы в пределах ряда
                for idx in indices:
                    if idx >= length or idx < 0:
                        raise ValueError(f"Индекс {idx} вне диапазона [0, {length-1}]")

                series_with_nan[indices] = np.nan
                mask[indices] = False

        elif continuous_max_length is not None and continuous_percentage is not None:
            # Режим 1: заданы max_length и процент удаляемых точек
            if not (0 <= continuous_percentage <= 100):
                raise ValueError(f"continuous_percentage должен быть в диапазоне [0, 100], получено {continuous_percentage}")

            if continuous_max_length <= 0:
                raise ValueError(f"continuous_max_length должен быть > 0, получено {continuous_max_length}")

            # Количество точек для удаления
            n_missing = int(length * continuous_percentage / 100)

            if n_missing == 0:
                print("Предупреждение: continuous_percentage слишком мал, пропуски не созданы")
                return series_with_nan, mask

            # Создаем блоки пропусков
            missing_indices = []
            total_missing = 0

            # Сначала генерируем блоки, пока не наберем нужное количество точек
            attempts = 0
            max_attempts = length * 10  # Защита от бесконечного цикла

            while total_missing < n_missing and attempts < max_attempts:
                attempts += 1

                # Случайная длина блока (1 до max_length)
                block_length = np.random.randint(1, continuous_max_length + 1)

                # Случайная начальная позиция
                start = np.random.randint(0, length - block_length + 1)

                # Блок индексов
                block_indices = list(range(start, start + block_length))

                # Проверяем, не пересекается ли с уже созданными пропусками
                # (используем set для быстрой проверки)
                existing_indices_set = set(missing_indices)
                block_set = set(block_indices)

                if not block_set.intersection(existing_indices_set):
                    missing_indices.extend(block_indices)
                    total_missing += block_length

            # Если набрали больше чем нужно, обрезаем лишние
            if total_missing > n_missing:
                extra = total_missing - n_missing
                # Удаляем случайные индексы из созданных
                indices_to_remove = np.random.choice(missing_indices, size=extra, replace=False)
                for idx in indices_to_remove:
                    missing_indices.remove(idx)

            # Применяем пропуски
            for idx in missing_indices:
                if idx < length:  # Проверка на случай обрезания
                    series_with_nan[idx] = np.nan
                    mask[idx] = False

        else:
            raise ValueError(
                "Для missing_type='continuous' нужно указать либо "
                "(continuous_max_length + continuous_percentage), "
                "либо continuous_indices"
            )

    # --- 3. Смешанные пропуски ---
    elif missing_type == 'mixed':
        # --- 3.1 Добавляем точечные пропуски ---
        if mixed_point_percentage is None:
            raise ValueError("Для missing_type='mixed' необходимо указать mixed_point_percentage")

        if not (0 <= mixed_point_percentage <= 100):
            raise ValueError(f"mixed_point_percentage должен быть в диапазоне [0, 100], получено {mixed_point_percentage}")

        n_point_missing = int(length * mixed_point_percentage / 100)

        if n_point_missing > 0:
            # Случайно выбираем индексы для точечных пропусков
            point_indices = np.random.choice(length, size=n_point_missing, replace=False)

            # Устанавливаем NaN и обновляем маску
            series_with_nan[point_indices] = np.nan
            mask[point_indices] = False

        # --- 3.2 Добавляем непрерывные пропуски ---
        # Проверяем, какой режим используется для непрерывных
        if mixed_continuous_indices is not None:
            # Режим 2: индексы заданы явно
            for indices in mixed_continuous_indices:
                # Проверяем, что индексы не пересекаются с уже созданными пропусками
                for idx in indices:
                    if idx >= length or idx < 0:
                        raise ValueError(f"Индекс {idx} вне диапазона [0, {length-1}]")
                    if not mask[idx]:  # Если уже есть пропуск
                        # Пропускаем этот индекс, чтобы не перезаписывать
                        continue

                # Применяем пропуски
                for idx in indices:
                    if mask[idx]:  # Если еще нет пропуска
                        series_with_nan[idx] = np.nan
                        mask[idx] = False

        elif mixed_continuous_max_length is not None and mixed_continuous_percentage is not None:
            # Режим 1: заданы max_length и процент удаляемых точек
            if not (0 <= mixed_continuous_percentage <= 100):
                raise ValueError(f"mixed_continuous_percentage должен быть в диапазоне [0, 100], получено {mixed_continuous_percentage}")

            if mixed_continuous_max_length <= 0:
                raise ValueError(f"mixed_continuous_max_length должен быть > 0, получено {mixed_continuous_max_length}")

            # Количество точек для удаления в непрерывных пропусках
            n_continuous_missing = int(length * mixed_continuous_percentage / 100)

            if n_continuous_missing > 0:
                # Создаем блоки пропусков, избегая уже созданных точечных пропусков
                continuous_indices = []
                total_missing = 0

                # Получаем индексы, где еще нет пропусков
                available_indices = np.where(mask)[0].tolist()

                if len(available_indices) == 0:
                    print("Предупреждение: все точки уже удалены, непрерывные пропуски не созданы")
                else:
                    attempts = 0
                    max_attempts = len(available_indices) * 5

                    # Сортируем доступные индексы для поиска блоков
                    available_indices = sorted(available_indices)

                    while total_missing < n_continuous_missing and attempts < max_attempts and len(available_indices) > 0:
                        attempts += 1

                        # Выбираем случайную начальную позицию из доступных
                        start_idx = np.random.choice(available_indices)

                        # Случайная длина блока (1 до max_length)
                        block_length = np.random.randint(1, mixed_continuous_max_length + 1)

                        # Проверяем, все ли индексы блока доступны
                        block_indices = []
                        current_idx = start_idx
                        for _ in range(block_length):
                            if current_idx in available_indices:
                                block_indices.append(current_idx)
                                current_idx += 1
                            else:
                                # Если встретили занятый индекс, пытаемся продолжить дальше
                                # Но ограничиваем поиск, чтобы не делать бесконечный цикл
                                if len(block_indices) == 0:
                                    break
                                # Ищем следующий доступный индекс после текущего
                                next_available = [idx for idx in available_indices if idx > current_idx]
                                if next_available:
                                    current_idx = next_available[0]
                                else:
                                    break

                        # Если блок получился достаточно большим (>0), добавляем
                        if len(block_indices) > 0:
                            continuous_indices.extend(block_indices)
                            total_missing += len(block_indices)

                            # Удаляем использованные индексы из доступных
                            for idx in block_indices:
                                if idx in available_indices:
                                    available_indices.remove(idx)

                    # Если набрали больше чем нужно, обрезаем
                    if total_missing > n_continuous_missing:
                        extra = total_missing - n_continuous_missing
                        indices_to_remove = np.random.choice(continuous_indices, size=extra, replace=False)
                        for idx in indices_to_remove:
                            if idx in continuous_indices:
                                continuous_indices.remove(idx)

                    # Применяем непрерывные пропуски
                    for idx in continuous_indices:
                        if idx < length and mask[idx]:
                            series_with_nan[idx] = np.nan
                            mask[idx] = False

        else:
            raise ValueError(
                "Для missing_type='mixed' нужно указать либо "
                "(mixed_continuous_max_length + mixed_continuous_percentage), "
                "либо mixed_continuous_indices"
            )

    else:
        raise ValueError(
            f"Неизвестный тип пропусков: '{missing_type}'. "
            f"Доступные: 'point', 'continuous', 'mixed'"
        )

    return series_with_nan, mask
