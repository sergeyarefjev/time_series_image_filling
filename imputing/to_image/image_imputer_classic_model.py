import numpy as np
from scipy.ndimage import generic_filter

def adaptive_masked_median_filter(
    image: np.ndarray,
    mask: np.ndarray,
    min_size: int = 3,
    max_size: int = 21
) -> np.ndarray:
    """
    Адаптивный медианный фильтр, который динамически увеличивает размер окна,
    если текущее окно полностью состоит из маскированных (NaN) значений.

    Parameters
    ----------
    image : np.ndarray
        Исходное 2D-изображение.
    mask : np.ndarray
        Бинарная маска (1 - валидный пиксель, 0 - пропуск/NaN).
    min_size : int, default=3
        Стартовый размер окна (должен быть нечетным).
    max_size : int, default=21
        Максимально допустимый размер окна для предотвращения чрезмерного размытия.

    Returns
    -------
    np.ndarray
        Отфильтрованное изображение с адаптивно заполненными пропусками.
    """
    h, w = image.shape
    result = image.copy().astype(np.float64)

    # Подготовим матрицу, где пропуски заполнены NaN
    work_img = image.copy().astype(np.float64)
    work_img[mask < 0.5] = np.nan

    # Будем заполнять только те пиксели, которые изначально были пропущены
    missing_y, missing_x = np.where(mask < 0.5)

    for y, x in zip(missing_y, missing_x):
        current_size = min_size
        val = np.nan

        # Итеративно увеличиваем окно, пока не найдем хотя бы одно валидное значение
        while current_size <= max_size:
            half = current_size // 2

            # Вырезаем локальное окно с учетом границ изображения
            y_min, y_max = max(0, y - half), min(h, y + half + 1)
            x_min, x_max = max(0, x - half), min(w, x + half + 1)

            window = work_img[y_min:y_max, x_min:x_max]
            valid_values = window[~np.isnan(window)]

            if len(valid_values) > 0:
                val = np.median(valid_values)
                break # Нашли медиану, выходим из цикла увеличения окна

            current_size += 2 # Увеличиваем размер окна (сохраняя нечетность)

        # Если даже при максимальном окне не нашли валидных точек,
        # используем глобальную медиану или 0.0
        if np.isnan(val):
            global_valid = work_img[~np.isnan(work_img)]
            result[y, x] = np.median(global_valid) if len(global_valid) > 0 else 0.0
        else:
            result[y, x] = val

    return result

def smart_image_inpainting_adaptive_median(
    img_data: np.ndarray,
    img_mask: np.ndarray,
    min_size: int = 3,
    max_size: int = 15
) -> np.ndarray:
    """
    Инпейнтинг на основе адаптивного медианного фильтра.
    """
    img_float = img_data.astype(np.float32)

    if img_float.ndim == 3:
        filled_channels = []
        for c in range(img_float.shape[0]):
            channel = img_float[c]
            filtered = adaptive_masked_median_filter(channel, img_mask, min_size=min_size, max_size=max_size)
            channel_filled = channel.copy()
            channel_filled[img_mask < 0.5] = filtered[img_mask < 0.5]
            filled_channels.append(channel_filled)
        return np.stack(filled_channels, axis=0)
    else:
        filtered = adaptive_masked_median_filter(img_float, img_mask, min_size=min_size, max_size=max_size)
        img_filled = img_float.copy()
        img_filled[img_mask < 0.5] = filtered[img_mask < 0.5]
        return img_filled


class ImageImputer:
    """
    Класс для заполнения пропусков во временных рядах через преобразование в изображения.

    Этапы работы:
    1. Временной ряд преобразуется в 2D-изображение (Slice, Spectrogram или Wavelet)
    2. Проводится инпейнтинг (заполнение областей пропусков) на 2D-сетке
    3. Восстановленное 2D-изображение конвертируется обратно в одномерный ряд

    Поддерживаемые модели заполнения на изображениях:
    1. 'median' - медианный фильтр (scipy.ndimage.median_filter)
    2. 'telea' - метод Александра Телеа (OpenCV INPAINT_TELEA)
    3. 'ns' - метод Навье-Стокса (OpenCV INPAINT_NS)
    """

    def __init__(
        self,
        representation: Literal['slice', 'spectrogram', 'wavelet'] = 'slice',
        model: Literal['median', 'telea', 'ns'] = 'telea',
        representation_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None
    ):
        self.representation = representation
        self.model = model
        self.representation_params = representation_params or {}
        self.model_params = model_params or {}

    def _apply_inpainting(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Применяет выбранный алгоритм заполнения к 2D-изображению.
        """
        cv2_mask = ((1.0 - mask) * 255).astype(np.uint8)
        img_float = img.astype(np.float32)

        if self.model == 'median':
            # Извлекаем параметры для адаптивного маскированного фильтра
            min_size = self.model_params.get('size', 3)
            max_size = self.model_params.get('max_size', 15)

            return smart_image_inpainting_adaptive_median(
                img_float,
                mask,
                min_size=min_size,
                max_size=max_size
            )

        elif self.model in ['telea', 'ns']:
            inpaint_radius = self.model_params.get('inpaint_radius', 3.0)
            flags = cv2.INPAINT_TELEA if self.model == 'telea' else cv2.INPAINT_NS

            if img_float.ndim == 3:
                filled_channels = []
                for c in range(img_float.shape[0]):
                    channel = img_float[c]
                    filled_c = cv2.inpaint(channel, cv2_mask, inpaintRadius=int(inpaint_radius), flags=flags)
                    filled_channels.append(filled_c)
                return np.stack(filled_channels, axis=0)
            else:
                return cv2.inpaint(img_float, cv2_mask, inpaintRadius=int(inpaint_radius), flags=flags)
        else:
            raise ValueError(f"Неизвестная модель инпейнтинга: {self.model}")

    def fit_predict(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Заполняет пропуски в ряду методом инпейнтинга изображений.
        """
        converter = TimeSeries2Image(series)

        if self.representation == 'slice':
            seasonality = self.representation_params.get('seasonality', 24)
            img_data, img_mask = converter.to_slice(seasonality=seasonality, fill_method='linear')
            img_inpainted = self._apply_inpainting(img_data, img_mask)
            return converter.from_slice(image_data=img_inpainted, use_mask=True)

        elif self.representation == 'spectrogram':
            n_fft = self.representation_params.get('n_fft', 128)
            hop_length = self.representation_params.get('hop_length', 64)
            img_data, img_mask = converter.to_spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                fill_method='linear',
                mask_mode='strict'
            )
            img_inpainted = self._apply_inpainting(img_data, img_mask)
            return converter.from_spectrogram(image_data=img_inpainted, use_mask=True)

        elif self.representation == 'wavelet':
            wavelet = self.representation_params.get('wavelet', 'morl')
            scales_range = self.representation_params.get('scales_range', (1, 128))
            num_scales = self.representation_params.get('num_scales', 128)
            img_data, img_mask = converter.to_wavelet(
                wavelet=wavelet,
                scales_range=scales_range,
                num_scales=num_scales,
                fill_method='linear',
                mask_mode='strict'
            )
            img_inpainted = self._apply_inpainting(img_data, img_mask)
            return converter.from_wavelet(image_data=img_inpainted, use_mask=True)
        else:
            raise ValueError(f"Неподдерживаемое представление: {self.representation}")
