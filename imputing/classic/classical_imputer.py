class ClassicalImputer:
    """
    Класс для заполнения пропусков во временных рядах классическими методами.

    Поддерживаемые модели:
    1. 'linear' - линейная интерполяция
    2. 'knn' - K-ближайших соседей
    3. 'lgbm' - LightGBM (градиентный бустинг)
    """

    def __init__(
        self,
        model: Literal['linear', 'knn', 'lgbm'] = 'linear',
        model_params: Optional[Dict[str, Any]] = None,
        lags: Union[int, List[int]] = 1,
        seasonal_lags: Optional[Dict[int, int]] = None,
        use_time_features: bool = True,
        n_iterations: int = 3, # Увеличим число итераций по умолчанию
        boundary_fill: Literal['extrapolate', 'ffill', 'bfill'] = 'extrapolate',
        scaler: bool = True,
        random_state: int = 42
    ):
        self.model = model
        self.model_params = model_params or {}
        self.seasonal_lags = seasonal_lags or {}
        self.use_time_features = use_time_features
        self.n_iterations = max(1, n_iterations)
        self.boundary_fill = boundary_fill
        self.scaler = scaler
        self.random_state = random_state

        if isinstance(lags, int):
            self.lags = list(range(1, lags + 1))
        else:
            self.lags = sorted(lags)

        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

        self._validate_params()

    def _validate_params(self) -> None:
        valid_models = ['linear', 'knn', 'lgbm']
        if self.model not in valid_models:
            raise ValueError(f"Неизвестная модель: {self.model}. Доступные: {valid_models}")

    def _create_features(
        self,
        series: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        n = len(series)
        if mask is None:
            mask = ~np.isnan(series)

        features = pd.DataFrame(index=range(n))

        # 1. Обычные лаги
        for lag in self.lags:
            features[f'lag_{lag}'] = np.roll(series, lag)
            features[f'lag_{lag}'].iloc[:lag] = np.nan

        # 2. Сезонные лаги
        for period, count in self.seasonal_lags.items():
            for i in range(1, count + 1):
                lag = period * i
                features[f'seasonal_lag_{period}_{i}'] = np.roll(series, lag)
                features[f'seasonal_lag_{period}_{i}'].iloc[:lag] = np.nan

        # 3. Тригонометрические признаки
        if self.use_time_features:
            periods = set(self.seasonal_lags.keys())
            if not periods:
                periods = {24, 168}
            for period in periods:
                features[f'sin_{period}'] = np.sin(2 * np.pi * np.arange(n) / period)
                features[f'cos_{period}'] = np.cos(2 * np.pi * np.arange(n) / period)

        target = series.copy()
        valid_for_train = mask.copy()
        for col in features.columns:
            valid_for_train = valid_for_train & ~np.isnan(features[col].values)
        valid_for_train = valid_for_train & ~np.isnan(target)

        self.feature_names_ = list(features.columns)
        return features, target, valid_for_train

    def _fill_boundary_with_interpolation(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        series_filled = series.copy()
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return np.zeros_like(series)

        first_valid, last_valid = valid_indices[0], valid_indices[-1]

        if self.boundary_fill == 'extrapolate':
            if len(valid_indices) >= 2:
                slope, intercept = np.polyfit(valid_indices, series[valid_indices], 1)
                series_filled[:first_valid] = slope * np.arange(first_valid) + intercept
                series_filled[last_valid + 1:] = slope * np.arange(last_valid + 1, len(series)) + intercept
            else:
                val = series[first_valid]
                series_filled[:first_valid] = val
                series_filled[last_valid + 1:] = val
        elif self.boundary_fill == 'ffill':
            series_filled[:first_valid] = series[first_valid]
            series_filled[last_valid + 1:] = series[last_valid]
        elif self.boundary_fill == 'bfill':
            series_filled[:first_valid] = series[first_valid]
            series_filled[last_valid + 1:] = series[last_valid]

        return series_filled

    def _interpolate_linear(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        series_filled = self._fill_boundary_with_interpolation(series, mask)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) >= 2:
            missing_indices = np.where(~mask)[0]
            if len(missing_indices) > 0:
                interp_func = interp1d(
                    valid_indices,
                    series[valid_indices],
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                series_filled[missing_indices] = interp_func(missing_indices)
        return series_filled

    def fit(self, series: np.ndarray, mask: Optional[np.ndarray] = None) -> 'ClassicalImputer':
        if mask is None:
            mask = ~np.isnan(series)
        if self.model == 'linear':
            self.is_fitted_ = True
            return self

        # Инициализируем пропуски линейной интерполяцией для корректного построения признаков при обучении
        series_initialized = self._interpolate_linear(series, mask)

        features, target, valid_mask = self._create_features(series_initialized, mask)
        X_train = features.values[valid_mask]
        y_train = target[valid_mask]

        if len(X_train) == 0:
            raise ValueError("Нет валидных данных для обучения!")

        if self.scaler:
            self.scaler_ = StandardScaler()
            X_train = self.scaler_.fit_transform(X_train)

        if self.model == 'knn':
            params = {'n_neighbors': 5, 'weights': 'distance', 'n_jobs': -1}
            params.update(self.model_params)
            self.model_ = KNeighborsRegressor(**params)
            self.model_.fit(X_train, y_train)
        elif self.model == 'lgbm':
            params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': self.random_state, 'verbose': -1, 'n_jobs': -1}
            params.update(self.model_params)
            self.model_ = lgb.LGBMRegressor(**params)
            self.model_.fit(X_train, y_train)

        self.is_fitted_ = True
        return self

    def _predict_with_model(self, X_all: np.ndarray, missing_indices: np.ndarray) -> np.ndarray:
        X_missing = X_all[missing_indices]
        nan_mask = np.isnan(X_missing).any(axis=1)
        if np.any(nan_mask):
            # Если вдруг остались NaN (например, на краях), заменяем их нулями
            X_missing = np.nan_to_num(X_missing, nan=0.0)

        if self.scaler and self.scaler_ is not None:
            X_missing = self.scaler_.transform(X_missing)

        preds = self.model_.predict(X_missing)
        return preds

    def predict(self, series: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if mask is None:
            mask = ~np.isnan(series)
        result = series.copy()
        if np.all(mask):
            return result
        if self.model == 'linear':
            return self._interpolate_linear(result, mask)

        if not self.is_fitted_ or self.model_ is None:
            raise ValueError("Сначала необходимо вызвать fit()!")

        # 1. Первичная «горячая» инициализация пропусков линейной интерполяцией,
        # чтобы убрать проблемы холодного старта при вычислении признаков (лагов)
        result_init = self._interpolate_linear(result, mask)

        # 2. Итеративно уточняем значения в пропущенных точках
        for _ in range(self.n_iterations):
            features, _, _ = self._create_features(result_init, mask)
            still_missing = np.where(~mask)[0]
            if len(still_missing) == 0:
                break

            preds = self._predict_with_model(features.values, still_missing)
            # Записываем уточненные предсказания обратно в рабочий массив
            result_init[still_missing] = preds

        # Возвращаем исходные известные значения на свои места, а пропуски заполняем модельными
        final_result = series.copy()
        final_result[~mask] = result_init[~mask]
        return final_result

    def fit_predict(self, series: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(series, mask)
        return self.predict(series, mask)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self.model != 'lgbm' or self.model_ is None:
            return None
        df = pd.DataFrame({'feature': self.feature_names_, 'importance': self.model_.feature_importances_})
        return df.sort_values('importance', ascending=False)
