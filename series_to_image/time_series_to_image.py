class TimeSeries2Image:
    """
    Класс для преобразования временного ряда с пропусками в изображение.

    Основная идея:
    1. На вход подается временной ряд с NaN (пропусками)
    2. Ряд заполняется (интерполяция/среднее/медиана)
    3. Создается бинарная маска пропусков
    4. Оба компонента (ряд и маска) преобразуются в изображения
    5. Итог: изображение данных + маска (как отдельный канал)

    Поддерживаемые методы преобразования:
    1. Slice (сезонное) - временной ряд режется на периоды
    2. Спектрограмма (STFT) - комплексная спектрограмма (2 канала: Re, Im)
    3. Вейвлет-преобразование (CWT) - многомасштабный анализ

    Attributes
    ----------
    original_series : np.ndarray
        Исходный временной ряд с NaN (пропусками)
    filled_series : np.ndarray
        Заполненный временной ряд (без NaN)
    mask_series : np.ndarray
        Бинарная маска на уровне ряда (1 - данные есть, 0 - пропуск)

    image_data : np.ndarray
        Изображение, полученное из заполненного ряда
    image_mask : np.ndarray
        Изображение маски (соответствует image_data по размерности)

    metadata : dict
        Метаданные для обратного преобразования
    """

    def __init__(self, series: np.ndarray):
        """
        Инициализация класса временным рядом с пропусками.

        Parameters
        ----------
        series : np.ndarray
            Временной ряд (может содержать NaN)
        """
        self.original_series = series.copy()
        self.filled_series = None
        self.mask_series = None

        self.image_data = None
        self.image_mask = None
        self.image_data_spectrogram = None
        self.image_mask_spectrogram = None
        self.image_data_wavelet = None
        self.image_mask_wavelet = None

        self.metadata = {
            'slice': {},
            'spectrogram': {},
            'wavelet': {},
            'fill_method': None,
            'fill_params': {}
        }

        # Автоматически создаем маску и заполняем ряд при инициализации
        self._create_mask()

    # ==================== ВНУТРЕННИЕ МЕТОДЫ ====================

    def _create_mask(self) -> None:
        """
        Создает бинарную маску пропусков на основе original_series.
        1 - данные есть (не NaN), 0 - пропуск (NaN)
        """
        self.mask_series = ~np.isnan(self.original_series)

    def fill_missing(
        self,
        method: Literal['linear', 'mean', 'median', 'zero', 'constant'] = 'linear',
        fill_value: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Заполняет пропущенные значения во временном ряду.

        Parameters
        ----------
        method : str, default='linear'
            Метод заполнения пропусков:
            - 'linear'   : линейная интерполяция
            - 'mean'     : заполнение средним значением по всему ряду
            - 'median'   : заполнение медианой по всему ряду
            - 'zero'     : заполнение нулями
            - 'constant' : заполнение константой (задается fill_value)

        fill_value : float, optional
            Значение для заполнения при method='constant'

        **kwargs
            Дополнительные параметры для интерполяции (например, kind='cubic')

        Returns
        -------
        np.ndarray
            Заполненный временной ряд

        Notes
        -----
        При линейной интерполяции можно передать kind='cubic' для кубической сплайн-интерполяции.
        """
        series = self.original_series.copy()
        mask = self.mask_series

        if method == 'linear':
            # Линейная интерполяция
            indices = np.arange(len(series))
            valid_indices = indices[mask]
            valid_values = series[mask]

            if len(valid_indices) < 2:
                # Если меньше 2 валидных точек, используем среднее
                warnings.warn("Недостаточно валидных точек для интерполяции. Используется среднее.")
                method = 'mean'
            else:
                # Создаем интерполяционную функцию
                kind = kwargs.get('kind', 'linear')
                interp_func = interp1d(
                    valid_indices,
                    valid_values,
                    kind=kind,
                    bounds_error=False,
                    fill_value=(valid_values[0], valid_values[-1])
                )
                series[~mask] = interp_func(indices[~mask])

        if method == 'mean':
            mean_val = np.mean(series[mask]) if np.any(mask) else 0.0
            series[~mask] = mean_val

        elif method == 'median':
            median_val = np.median(series[mask]) if np.any(mask) else 0.0
            series[~mask] = median_val

        elif method == 'zero':
            series[~mask] = 0.0

        elif method == 'constant':
            if fill_value is None:
                raise ValueError("Для method='constant' необходимо указать fill_value")
            series[~mask] = fill_value

        self.filled_series = series
        self.metadata['fill_method'] = method
        self.metadata['fill_params'] = kwargs

        return series

    # ==================== МЕТОДЫ ПРЕОБРАЗОВАНИЯ ====================

    def to_slice(
        self,
        seasonality: int = 24,
        periods: Optional[int] = None,
        step: Optional[int] = None,
        rows_are_time: bool = True,
        fill_method: Literal['linear', 'mean', 'median', 'zero', 'constant'] = 'linear',
        fill_value: Optional[float] = None,
        **fill_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Преобразование временного ряда в изображение методом Slice.

        Временной ряд разбивается на периоды длиной seasonality и складывается в матрицу.
        Строки матрицы - это время внутри периода, столбцы - номера периодов.

        Parameters
        ----------
        seasonality : int, default=24
            Длина периода (сезонности).
            Например: 24 для суточной сезонности (часы), 168 для недельной

        periods : int, optional
            Количество периодов для включения в изображение.
            Если None, берутся все возможные периоды.

        step : int, optional
            Шаг скользящего окна. Если None, step = seasonality (без перекрытия).

        rows_are_time : bool, default=True
            Если True: строки = время внутри периода, столбцы = номера периодов.
            Если False: строки = номера периодов, столбцы = время внутри периода.

        fill_method : str, default='linear'
            Метод заполнения пропусков перед преобразованием.

        fill_value : float, optional
            Значение для заполнения при fill_method='constant'

        **fill_kwargs
            Дополнительные параметры для интерполяции

        Returns
        -------
        tuple (image_data, image_mask)
            - image_data : np.ndarray - изображение с заполненными данными
            - image_mask : np.ndarray - изображение маски (1 - данные, 0 - пропуск)
        """
        # Заполняем пропуски
        self.fill_missing(method=fill_method, fill_value=fill_value, **fill_kwargs)
        series = self.filled_series
        mask = self.mask_series.astype(float)
        length = len(series)

        # Определяем шаг
        if step is None:
            step = seasonality

        # Определяем количество периодов
        if periods is None:
            periods = (length - seasonality) // step + 1
            if periods <= 0:
                raise ValueError(
                    f"Длина ряда ({length}) слишком мала для seasonality={seasonality}"
                )

        # Проверяем, что данных достаточно
        max_start = step * (periods - 1) + seasonality
        if max_start > length:
            warnings.warn(
                f"Недостаточно данных: требуется {max_start}, доступно {length}. "
                f"Будет взято {periods} периодов."
            )
            periods = (length - seasonality) // step + 1

        # Создаем матрицы для данных и маски
        if rows_are_time:
            img_data = np.zeros((seasonality, periods))
            img_mask = np.zeros((seasonality, periods))

            for i in range(periods):
                start = i * step
                end = start + seasonality
                if end <= length:
                    img_data[:, i] = series[start:end]
                    img_mask[:, i] = mask[start:end]
                else:
                    available = length - start
                    img_data[:available, i] = series[start:]
                    img_data[available:, i] = 0.0
                    img_mask[:available, i] = mask[start:]
                    img_mask[available:, i] = 0.0
        else:
            img_data = np.zeros((periods, seasonality))
            img_mask = np.zeros((periods, seasonality))

            for i in range(periods):
                start = i * step
                end = start + seasonality
                if end <= length:
                    img_data[i, :] = series[start:end]
                    img_mask[i, :] = mask[start:end]
                else:
                    available = length - start
                    img_data[i, :available] = series[start:]
                    img_data[i, available:] = 0.0
                    img_mask[i, :available] = mask[start:]
                    img_mask[i, available:] = 0.0

        # Сохраняем результаты
        self.image_data = img_data
        self.image_mask = img_mask

        # Сохраняем метаданные
        self.metadata['slice'] = {
            'seasonality': seasonality,
            'periods': periods,
            'step': step,
            'rows_are_time': rows_are_time,
            'original_length': length,
            'img_shape': img_data.shape,
            'mask_shape': img_mask.shape
        }

        return img_data, img_mask

    def to_spectrogram(
        self,
        n_fft: int = 128,
        hop_length: int = 64,
        window: str = 'hann',
        center: bool = True,
        pad_mode: str = 'reflect',
        fill_method: Literal['linear', 'mean', 'median', 'zero', 'constant'] = 'linear',
        fill_value: Optional[float] = None,
        mask_mode: Literal['strict', 'threshold'] = 'strict',
        mask_threshold: float = 0.5,
        **fill_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Преобразование временного ряда в комплексную спектрограмму (2 канала).

        Используется Short-Time Fourier Transform (STFT).

        Parameters
        ----------
        n_fft : int, default=128
            Размер окна для STFT.

        hop_length : int, default=64
            Шаг между окнами.

        window : str, default='hann'
            Тип окна: 'hann', 'hamming', 'blackman', 'bartlett', 'boxcar'

        center : bool, default=True
            Если True, сигнал дополняется так, чтобы окна были центрированы.

        pad_mode : str, default='reflect'
            Режим дополнения сигнала.

        fill_method : str, default='linear'
            Метод заполнения пропусков перед преобразованием.

        fill_value : float, optional
            Значение для заполнения при fill_method='constant'

        mask_mode : str, default='strict'
            Режим создания маски для спектрограммы:
            - 'strict'   : если в окне есть хотя бы один пропуск → весь столбец = 0
            - 'threshold': если доля валидных точек в окне < mask_threshold → столбец = 0

        mask_threshold : float, default=0.5
            Порог для режима 'threshold' (0-1). Доля валидных точек в окне.

        **fill_kwargs
            Дополнительные параметры для интерполяции

        Returns
        -------
        tuple (image_data, image_mask)
            - image_data : np.ndarray - спектрограмма (2, n_freq, n_time)
            - image_mask : np.ndarray - маска для спектрограммы (n_freq, n_time)
        """
        from scipy.signal import stft

        # Заполняем пропуски
        self.fill_missing(method=fill_method, fill_value=fill_value, **fill_kwargs)
        series = self.filled_series
        mask = self.mask_series.astype(float)
        length = len(series)

        # Выполняем STFT
        f, t, Zxx = stft(
            series,
            fs=1.0,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            detrend=False,
            return_onesided=True,
            boundary='zeros' if not center else None,
            padded=center,
            axis=-1
        )

        # Создаем маску для спектрограммы
        # Определяем, какие окна содержат пропуски
        n_freq = Zxx.shape[0]
        n_time = Zxx.shape[1]

        # Для каждого временного окна считаем долю валидных точек
        window_valid_ratio = np.zeros(n_time)

        for i in range(n_time):
            # Определяем границы окна во временном ряду
            if center:
                start = i * hop_length
                end = start + n_fft
                # При center=True сигнал дополняется, поэтому окна могут выходить за границы
                # Валидными считаем только те точки, которые были в исходном сигнале
                valid_start = max(0, start)
                valid_end = min(length, end)
                window_mask = mask[valid_start:valid_end]
                window_len = valid_end - valid_start
            else:
                start = i * hop_length
                end = min(start + n_fft, length)
                window_mask = mask[start:end]
                window_len = len(window_mask)

            if window_len > 0:
                window_valid_ratio[i] = np.sum(window_mask) / window_len
            else:
                window_valid_ratio[i] = 0.0

        # Создаем маску для спектрограммы
        if mask_mode == 'strict':
            # Если в окне есть хотя бы один пропуск → весь столбец маски = 0
            mask_spec = np.ones((n_freq, n_time))
            for i in range(n_time):
                if window_valid_ratio[i] < 1.0:  # есть пропуски
                    mask_spec[:, i] = 0.0
        elif mask_mode == 'threshold':
            # Если доля валидных точек < порога → весь столбец = 0
            mask_spec = np.ones((n_freq, n_time))
            for i in range(n_time):
                if window_valid_ratio[i] < mask_threshold:
                    mask_spec[:, i] = 0.0
        else:
            raise ValueError(f"Неизвестный mask_mode: {mask_mode}. Доступны: 'strict', 'threshold'")

        # Сохраняем как 2-канальное изображение (Re, Im)
        # shape: (2, n_freq, n_time)
        img_data = np.zeros((2, n_freq, n_time), dtype=np.float32)
        img_data[0, :, :] = np.real(Zxx)
        img_data[1, :, :] = np.imag(Zxx)

        # Сохраняем результаты
        self.image_data_spectrogram = img_data
        self.image_mask_spectrogram = mask_spec

        # Сохраняем метаданные
        self.metadata['spectrogram'] = {
            'n_fft': n_fft,
            'hop_length': hop_length,
            'window': window,
            'center': center,
            'pad_mode': pad_mode,
            'mask_mode': mask_mode,
            'mask_threshold': mask_threshold,
            'freq': f.tolist(),
            'time': t.tolist(),
            'original_length': length,
            'img_shape': img_data.shape,
            'mask_shape': mask_spec.shape
        }

        return img_data, mask_spec

    def to_wavelet(
        self,
        wavelet: str = 'morl',
        scales: Optional[np.ndarray] = None,
        scales_range: Tuple[int, int] = (1, 128),
        num_scales: Optional[int] = None,
        sampling_period: float = 1.0,
        fill_method: Literal['linear', 'mean', 'median', 'zero', 'constant'] = 'linear',
        fill_value: Optional[float] = None,
        mask_mode: Literal['strict', 'threshold'] = 'strict',
        mask_threshold: float = 0.5,
        **fill_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Преобразование временного ряда с помощью CWT (непрерывное вейвлет-преобразование).

        Parameters
        ----------
        wavelet : str, default='morl'
            Тип вейвлета: 'morl' (Morlet), 'mexh' (Mexican Hat), 'paul' (Paul)

        scales : np.ndarray, optional
            Массив масштабов. Если None, создается из scales_range.

        scales_range : tuple, default=(1, 128)
            Диапазон масштабов (min, max).

        num_scales : int, optional
            Количество масштабов. Если None, используется 128.

        sampling_period : float, default=1.0
            Период дискретизации для коррекции частот.

        fill_method : str, default='linear'
            Метод заполнения пропусков перед преобразованием.

        fill_value : float, optional
            Значение для заполнения при fill_method='constant'

        mask_mode : str, default='strict'
            Режим создания маски:
            - 'strict'   : если в окне влияния есть пропуск → пиксель = 0
            - 'threshold': если доля валидных точек < порога → пиксель = 0

        mask_threshold : float, default=0.5
            Порог для режима 'threshold' (0-1).

        **fill_kwargs
            Дополнительные параметры для интерполяции

        Returns
        -------
        tuple (image_data, image_mask)
            - image_data : np.ndarray - коэффициенты CWT (scales, time)
            - image_mask : np.ndarray - маска (scales, time)
        """
        import pywt

        # Заполняем пропуски
        self.fill_missing(method=fill_method, fill_value=fill_value, **fill_kwargs)
        series = self.filled_series
        mask = self.mask_series.astype(float)
        length = len(series)

        # Создаем массив масштабов
        if scales is None:
            if num_scales is None:
                num_scales = 128
            scales = np.linspace(scales_range[0], scales_range[1], num_scales)

        # Выполняем CWT
        coefficients, frequencies = pywt.cwt(
            series,
            scales=scales,
            wavelet=wavelet,
            sampling_period=sampling_period,
            method='fft'
        )

        # Сохраняем амплитуды
        if np.iscomplexobj(coefficients):
            img_data = np.abs(coefficients).astype(np.float32)
        else:
            img_data = coefficients.astype(np.float32)

        # Создаем маску для CWT
        n_scales = len(scales)
        img_mask = np.ones((n_scales, length), dtype=np.float32)

        # Для каждого масштаба определяем радиус влияния
        for i, scale in enumerate(scales):
            # Радиус влияния пропорционален масштабу
            radius = int(scale * 2)  # эмпирическая оценка
            if radius < 1:
                radius = 1

            # Для каждой временной точки проверяем окрестность
            for t in range(length):
                start = max(0, t - radius)
                end = min(length, t + radius + 1)
                window_mask = mask[start:end]
                window_len = end - start

                if window_len > 0:
                    valid_ratio = np.sum(window_mask) / window_len
                else:
                    valid_ratio = 0.0

                if mask_mode == 'strict':
                    if valid_ratio < 1.0:
                        img_mask[i, t] = 0.0
                elif mask_mode == 'threshold':
                    if valid_ratio < mask_threshold:
                        img_mask[i, t] = 0.0
                else:
                    raise ValueError(f"Неизвестный mask_mode: {mask_mode}")

        # Сохраняем результаты
        self.image_data_wavelet = img_data
        self.image_mask_wavelet = img_mask

        # Сохраняем метаданные
        self.metadata['wavelet'] = {
            'wavelet': wavelet,
            'scales': scales.tolist(),
            'frequencies': frequencies.tolist(),
            'num_scales': len(scales),
            'sampling_period': sampling_period,
            'mask_mode': mask_mode,
            'mask_threshold': mask_threshold,
            'original_length': length,
            'img_shape': img_data.shape,
            'mask_shape': img_mask.shape
        }

        return img_data, img_mask

    def to_all(
        self,
        slice_kwargs: Optional[Dict] = None,
        spectrogram_kwargs: Optional[Dict] = None,
        wavelet_kwargs: Optional[Dict] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Преобразование временного ряда всеми тремя методами одновременно.

        Parameters
        ----------
        slice_kwargs : dict, optional
            Параметры для метода to_slice
        spectrogram_kwargs : dict, optional
            Параметры для метода to_spectrogram
        wavelet_kwargs : dict, optional
            Параметры для метода to_wavelet

        Returns
        -------
        dict
            Словарь с ключами: 'slice', 'spectrogram', 'wavelet'
            Значения: tuple (image_data, image_mask)
        """
        if slice_kwargs is None:
            slice_kwargs = {}
        if spectrogram_kwargs is None:
            spectrogram_kwargs = {}
        if wavelet_kwargs is None:
            wavelet_kwargs = {}

        result = {}

        result['slice'] = self.to_slice(**slice_kwargs)
        result['spectrogram'] = self.to_spectrogram(**spectrogram_kwargs)
        result['wavelet'] = self.to_wavelet(**wavelet_kwargs)

        return result

    # ==================== МЕТОДЫ ОБРАТНОГО ПРЕОБРАЗОВАНИЯ ====================

    def from_slice(
        self,
        image_data: Optional[np.ndarray] = None,
        use_mask: bool = True
    ) -> np.ndarray:
        """
        Обратное преобразование из Slice-изображения во временной ряд.

        Parameters
        ----------
        image_data : np.ndarray, optional
            Изображение с восстановленными данными.
            Если None, используется self.image_data.

        use_mask : bool, default=True
            Использовать ли маску при восстановлении.
            Если True, то данные из маски накладываются на восстановленное изображение.

        Returns
        -------
        np.ndarray
            Восстановленный временной ряд.
        """
        if image_data is None:
            image_data = self.image_data
            if image_data is None:
                raise ValueError("Нет Slice-изображения. Сначала выполните to_slice()")

        meta = self.metadata['slice']
        seasonality = meta.get('seasonality', 24)
        step = meta.get('step', seasonality)
        rows_are_time = meta.get('rows_are_time', True)
        original_length = meta.get('original_length', 0)

        # Транспонируем если нужно
        if not rows_are_time:
            image_data = image_data.T

        # Восстанавливаем ряд
        n_periods = image_data.shape[1]
        recovered_length = step * (n_periods - 1) + seasonality

        recovered = np.zeros(recovered_length)
        for i in range(n_periods):
            start = i * step
            end = start + seasonality
            recovered[start:end] = image_data[:, i]

        # Обрезаем до исходной длины (это важно сделать ДО применения маски)
        if original_length > 0:
            if len(recovered) > original_length:
                recovered = recovered[:original_length]
            elif len(recovered) < original_length:
                # Дополняем недостающие значения
                pad_length = original_length - len(recovered)
                recovered = np.pad(recovered, (0, pad_length), constant_values=np.nan)

        # Если используем маску, накладываем оригинальные данные
        if use_mask and self.image_mask is not None:
            # Преобразуем маску обратно в ряд
            mask_img = self.image_mask
            if not rows_are_time:
                mask_img = mask_img.T

            mask_series = np.zeros(recovered_length)
            for i in range(n_periods):
                start = i * step
                end = start + seasonality
                mask_series[start:end] = mask_img[:, i]

            # Обрезаем маску до длины recovered (которая уже равна original_length)
            if len(mask_series) > len(recovered):
                mask_series = mask_series[:len(recovered)]
            elif len(mask_series) < len(recovered):
                pad_length = len(recovered) - len(mask_series)
                mask_series = np.pad(mask_series, (0, pad_length), constant_values=0.0)

            # Там где были валидные данные — оставляем оригинал
            # Там где были пропуски — берем восстановленное
            mask_series_bool = mask_series > 0.5

            # Убеждаемся, что размеры совпадают
            if len(mask_series_bool) == len(self.original_series):
                recovered[mask_series_bool] = self.original_series[mask_series_bool]
            else:
                # Если размеры все еще не совпадают, берем минимальную длину
                min_len = min(len(mask_series_bool), len(self.original_series), len(recovered))
                mask_series_bool = mask_series_bool[:min_len]
                recovered[:min_len][mask_series_bool] = self.original_series[:min_len][mask_series_bool]

        return recovered


    def from_spectrogram(
        self,
        image_data: Optional[np.ndarray] = None,
        length: Optional[int] = None,
        use_mask: bool = True
    ) -> np.ndarray:
        """
        Обратное преобразование из спектрограммы во временной ряд.

        Parameters
        ----------
        image_data : np.ndarray, optional
            Спектрограмма размером (2, n_freq, n_time).
            Если None, используется self.image_data_spectrogram.

        length : int, optional
            Длина восстанавливаемого ряда.

        use_mask : bool, default=True
            Использовать ли маску при восстановлении.

        Returns
        -------
        np.ndarray
            Восстановленный временной ряд.
        """
        from scipy.signal import istft

        if image_data is None:
            image_data = self.image_data_spectrogram
            if image_data is None:
                raise ValueError("Нет спектрограммы. Сначала выполните to_spectrogram()")

        meta = self.metadata['spectrogram']
        n_fft = meta.get('n_fft', 128)
        hop_length = meta.get('hop_length', 64)
        window = meta.get('window', 'hann')
        center = meta.get('center', True)
        original_length = meta.get('original_length', 0)

        if length is None:
            length = original_length

        # Восстанавливаем комплексный спектр
        Zxx = image_data[0, :, :] + 1j * image_data[1, :, :]

        # Применяем обратное STFT
        _, recovered = istft(
            Zxx,
            fs=1.0,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            input_onesided=True,
            boundary='zeros' if not center else None,
            time_axis=-1,
            freq_axis=-2
        )

        # Обрезаем до нужной длины
        if length > 0 and len(recovered) > length:
            recovered = recovered[:length]

        # Если используем маску, накладываем оригинальные данные
        if use_mask and self.mask_series is not None:
            mask_series = self.mask_series[:len(recovered)]
            recovered[mask_series] = self.original_series[:len(recovered)][mask_series]

        return recovered

    def from_wavelet(
        self,
        image_data: Optional[np.ndarray] = None,
        length: Optional[int] = None,
        use_mask: bool = True
    ) -> np.ndarray:
        """
        Обратное преобразование из вейвлет-коэффициентов во временной ряд.

        Parameters
        ----------
        image_data : np.ndarray, optional
            Изображение с вейвлет-коэффициентами.
            Если None, используется self.image_data_wavelet.

        length : int, optional
            Длина восстанавливаемого ряда.

        use_mask : bool, default=True
            Использовать ли маску при восстановлении.

        Returns
        -------
        np.ndarray
            Восстановленный временной ряд.
        """
        import pywt

        if image_data is None:
            image_data = self.image_data_wavelet
            if image_data is None:
                raise ValueError("Нет вейвлет-изображения. Сначала выполните to_wavelet()")

        meta = self.metadata['wavelet']
        wavelet = meta.get('wavelet', 'morl')
        scales = np.array(meta.get('scales', []))
        sampling_period = meta.get('sampling_period', 1.0)
        original_length = meta.get('original_length', 0)

        if length is None:
            length = original_length

        if len(scales) == 0:
            raise ValueError("Нет информации о масштабах в метаданных")

        # Восстанавливаем сигнал из коэффициентов
        try:
            recovered = pywt.icwt(
                image_data.astype(np.complex128),
                scales=scales,
                wavelet=wavelet,
                sampling_period=sampling_period
            )
        except Exception as e:
            warnings.warn(f"Обратное вейвлет-преобразование не удалось: {e}")
            recovered = np.sum(image_data, axis=0)
            recovered = recovered / np.max(np.abs(recovered)) if np.max(np.abs(recovered)) > 0 else recovered

        # Обрезаем до нужной длины
        if length > 0 and len(recovered) > length:
            recovered = recovered[:length]

        # Если используем маску, накладываем оригинальные данные
        if use_mask and self.mask_series is not None:
            mask_series = self.mask_series[:len(recovered)]
            recovered[mask_series] = self.original_series[:len(recovered)][mask_series]

        return recovered

    def from_all(self) -> Dict[str, np.ndarray]:
        """
        Восстановление временных рядов из всех трех изображений.

        Returns
        -------
        dict
            Словарь с ключами: 'slice', 'spectrogram', 'wavelet'
        """
        result = {}

        try:
            result['slice'] = self.from_slice()
        except Exception as e:
            warnings.warn(f"Ошибка восстановления из Slice: {e}")
            result['slice'] = None

        try:
            result['spectrogram'] = self.from_spectrogram()
        except Exception as e:
            warnings.warn(f"Ошибка восстановления из спектрограммы: {e}")
            result['spectrogram'] = None

        try:
            result['wavelet'] = self.from_wavelet()
        except Exception as e:
            warnings.warn(f"Ошибка восстановления из вейвлета: {e}")
            result['wavelet'] = None

        return result

    # ==================== МЕТОДЫ СОХРАНЕНИЯ И ЗАГРУЗКИ ====================

    def save(self, path: str, save_images: bool = True) -> None:
        """
        Сохранение преобразованных изображений и метаданных на диск.

        Parameters
        ----------
        path : str
            Путь для сохранения (без расширения).
        save_images : bool, default=True
            Сохранять ли изображения.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) if os.path.dirname(path) else '.', exist_ok=True)

        # Сохраняем метаданные
        metadata_path = f"{path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"Метаданные сохранены в: {metadata_path}")

        if save_images:
            images = {}
            if self.image_data is not None:
                images['slice_data'] = self.image_data
                images['slice_mask'] = self.image_mask
            if self.image_data_spectrogram is not None:
                images['spectrogram_data'] = self.image_data_spectrogram
                images['spectrogram_mask'] = self.image_mask_spectrogram
            if self.image_data_wavelet is not None:
                images['wavelet_data'] = self.image_data_wavelet
                images['wavelet_mask'] = self.image_mask_wavelet

            if images:
                images_path = f"{path}_images.npz"
                np.savez_compressed(images_path, **images)
                print(f"Изображения сохранены в: {images_path}")
            else:
                print("Нет изображений для сохранения")

    def load(self, path: str, load_images: bool = True) -> None:
        """
        Загрузка преобразованных изображений и метаданных с диска.

        Parameters
        ----------
        path : str
            Путь к сохраненным данным (без расширения).
        load_images : bool, default=True
            Загружать ли изображения.
        """
        metadata_path = f"{path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Метаданные загружены из: {metadata_path}")
        else:
            warnings.warn(f"Файл метаданных не найден: {metadata_path}")

        if load_images:
            images_path = f"{path}_images.npz"
            if os.path.exists(images_path):
                data = np.load(images_path, allow_pickle=True)
                if 'slice_data' in data:
                    self.image_data = data['slice_data']
                    self.image_mask = data['slice_mask']
                if 'spectrogram_data' in data:
                    self.image_data_spectrogram = data['spectrogram_data']
                    self.image_mask_spectrogram = data['spectrogram_mask']
                if 'wavelet_data' in data:
                    self.image_data_wavelet = data['wavelet_data']
                    self.image_mask_wavelet = data['wavelet_mask']
                print(f"Изображения загружены из: {images_path}")
            else:
                warnings.warn(f"Файл изображений не найден: {images_path}")

    # ==================== МЕТОДЫ ВИЗУАЛИЗАЦИИ ====================

    def visualize(
        self,
        method: str = 'all',
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Визуализация преобразованных изображений с масками.

        Parameters
        ----------
        method : str, default='all'
            Какой метод визуализировать: 'slice', 'spectrogram', 'wavelet', 'all'
        figsize : tuple, default=(15, 10)
            Размер фигуры.
        """
        if method == 'all':
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            methods = ['slice', 'spectrogram', 'wavelet']
            titles = ['Slice (сезонное)', 'Спектрограмма (STFT)', 'Вейвлет (CWT)']

            for i, (m, title) in enumerate(zip(methods, titles)):
                self._plot_single_method(m, axes[i, 0], axes[i, 1], axes[i, 2], title)

            plt.tight_layout()
            plt.show()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            titles = {
                'slice': 'Slice (сезонное)',
                'spectrogram': 'Спектрограмма (STFT)',
                'wavelet': 'Вейвлет (CWT)'
            }
            self._plot_single_method(
                method, axes[0], axes[1], axes[2],
                titles.get(method, method)
            )
            plt.tight_layout()
            plt.show()

    def _plot_single_method(
        self,
        method: str,
        ax_data: plt.Axes,
        ax_mask: plt.Axes,
        ax_overlay: plt.Axes,
        title: str
    ) -> None:
        """
        Вспомогательный метод для отрисовки одного метода (данные + маска + overlay).
        """
        if method == 'slice':
            data = self.image_data
            mask = self.image_mask
            cmap = 'viridis'
            title_data = f'{title}\nДанные'
            title_mask = f'{title}\nМаска'
            title_overlay = f'{title}\nНаложение'
        elif method == 'spectrogram':
            data = self.image_data_spectrogram
            mask = self.image_mask_spectrogram
            cmap = 'inferno'
            title_data = f'{title}\nДанные (амплитуда)'
            title_mask = f'{title}\nМаска'
            title_overlay = f'{title}\nНаложение'

            # Для спектрограммы показываем амплитуду
            if data is not None:
                data = np.sqrt(data[0]**2 + data[1]**2)
        elif method == 'wavelet':
            data = self.image_data_wavelet
            mask = self.image_mask_wavelet
            cmap = 'coolwarm'
            title_data = f'{title}\nДанные'
            title_mask = f'{title}\nМаска'
            title_overlay = f'{title}\nНаложение'
        else:
            ax_data.text(0.5, 0.5, f'Неизвестный метод: {method}', ha='center', va='center')
            ax_mask.axis('off')
            ax_overlay.axis('off')
            return

        # Данные
        if data is None:
            ax_data.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        else:
            im_data = ax_data.imshow(data, aspect='auto', cmap=cmap)
            ax_data.set_title(title_data)
            plt.colorbar(im_data, ax=ax_data, fraction=0.046, pad=0.04)

        # Маска
        if mask is None:
            ax_mask.text(0.5, 0.5, 'Нет маски', ha='center', va='center')
        else:
            im_mask = ax_mask.imshow(mask, aspect='auto', cmap='gray', vmin=0, vmax=1)
            ax_mask.set_title(title_mask)
            ax_mask.set_xlabel('1 - данные, 0 - пропуски')
            plt.colorbar(im_mask, ax=ax_mask, fraction=0.046, pad=0.04)

        # Overlay (маска поверх данных)
        if data is None or mask is None:
            ax_overlay.text(0.5, 0.5, 'Нет данных для наложения', ha='center', va='center')
        else:
            # Нормализуем данные для overlay
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            # Создаем RGB-изображение
            overlay = np.zeros((*data.shape, 3))
            overlay[:, :, 0] = data_norm  # R - данные
            overlay[:, :, 1] = data_norm  # G - данные
            overlay[:, :, 2] = data_norm  # B - данные
            # Маска: где 0 - накладываем красный цвет
            mask_bool = mask < 0.5
            if np.any(mask_bool):
                overlay[mask_bool, 0] = 1.0  # R
                overlay[mask_bool, 1] = 0.0  # G
                overlay[mask_bool, 2] = 0.0  # B

            ax_overlay.imshow(overlay, aspect='auto')
            ax_overlay.set_title(title_overlay)
            ax_overlay.set_xlabel('Красный = пропуски')

    def compare_reconstruction(
        self,
        original_series: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (15, 8)
    ) -> None:
        """
        Сравнение оригинального ряда с восстановленными из всех методов.

        Parameters
        ----------
        original_series : np.ndarray, optional
            Оригинальный ряд для сравнения. Если None, используется self.original_series.
        figsize : tuple, default=(15, 8)
            Размер фигуры.
        """
        if original_series is None:
            original_series = self.original_series

        # Восстанавливаем все методы
        recovered = self.from_all()

        fig, axes = plt.subplots(4, 1, figsize=figsize)

        # Оригинал
        axes[0].plot(original_series, color='black', alpha=0.7, label='Оригинал')
        axes[0].set_title('Оригинальный ряд (с пропусками)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Slice
        if recovered['slice'] is not None:
            axes[1].plot(original_series, color='gray', alpha=0.4, label='Оригинал')
            axes[1].plot(recovered['slice'], color='blue', alpha=0.8, label='Slice')
            axes[1].set_title('Восстановление из Slice')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Нет данных', ha='center', va='center')

        # Спектрограмма
        if recovered['spectrogram'] is not None:
            axes[2].plot(original_series, color='gray', alpha=0.4, label='Оригинал')
            axes[2].plot(recovered['spectrogram'], color='green', alpha=0.8, label='Спектрограмма')
            axes[2].set_title('Восстановление из спектрограммы')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Нет данных', ha='center', va='center')

        # Вейвлет
        if recovered['wavelet'] is not None:
            axes[3].plot(original_series, color='gray', alpha=0.4, label='Оригинал')
            axes[3].plot(recovered['wavelet'], color='red', alpha=0.8, label='Вейвлет')
            axes[3].set_title('Восстановление из вейвлета')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        else:
            axes[3].text(0.5, 0.5, 'Нет данных', ha='center', va='center')

        plt.tight_layout()
        plt.show()

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

    def get_data(self) -> Dict[str, Any]:
        """
        Получить все данные (ряды, изображения, маски).
        """
        return {
            'original_series': self.original_series,
            'filled_series': self.filled_series,
            'mask_series': self.mask_series,
            'image_data': self.image_data,
            'image_mask': self.image_mask,
            'image_data_spectrogram': self.image_data_spectrogram,
            'image_mask_spectrogram': self.image_mask_spectrogram,
            'image_data_wavelet': self.image_data_wavelet,
            'image_mask_wavelet': self.image_mask_wavelet
        }

    def reset(self) -> None:
        """
        Сбросить все преобразованные изображения и метаданные.
        """
        self.filled_series = None
        self.image_data = None
        self.image_mask = None
        self.image_data_spectrogram = None
        self.image_mask_spectrogram = None
        self.image_data_wavelet = None
        self.image_mask_wavelet = None
        self.metadata = {
            'slice': {},
            'spectrogram': {},
            'wavelet': {},
            'fill_method': None,
            'fill_params': {}
        }
