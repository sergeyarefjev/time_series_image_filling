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
