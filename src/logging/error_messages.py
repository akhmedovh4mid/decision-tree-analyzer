class DataCleanerError:
    FILL_CONSTANT_MISSING_VALUE_ERROR = (
        "Для стратегии 'fill_constant' необходимо указать fill_value"
    )
    NON_NUMERIC_COLUMN_IQR_ERROR = "Для удаления выбросов из межквартильного размаха (IQR) столбец '{col}' должен быть числовым."
    MISSING_COLUMNS_NOT_FOUND_ERROR = (
        "Для отсутствующих данных по обработке отсутствуют корректные столбцы."
    )
    NON_NUMERIC_COLUMN_ZSCORE_ERROR = "Для удаления выбросов из Z-критерия столбец '{col}' должен содержать числовые значения."


class EncoderError:
    NO_COLUMNS_TO_ENCODE = "Не найдено колонок для кодирования."
    UNKNOWN_STRATEGY = "Неизвестная стратегия кодирования: {strategy}"
    ORDINAL_CATEGORIES_REQUIRED = (
        "Для ordinal-кодирования необходимо указать категории."
    )
    CATEGORIES_NOT_FOUND = (
        "Не задан список категорий для колонки '{col}' при ordinal-кодировании."
    )
    TARGET_COLUMN_REQUIRED = (
        "Для target-кодирования необходимо указать целевую колонку."
    )
    TARGET_COLUMN_NOT_FOUND = "Целевая колонка '{col}' не найдена в DataFrame."
    ENCODER_NOT_FITTED = "Энкодер для колонки '{col}' не был обучен."


class ScalerError:
    NO_COLUMNS_TO_SCALE = "Не найдено числовых колонок для масштабирования."
    UNKNOWN_STRATEGY = "Неизвестная стратегия масштабирования: {strategy}"
    SCALER_NOT_FITTED = "Скейлер для колонки '{col}' не был обучен."


class SplitterError:
    TARGET_COLUMN_NOT_FOUND = "Целевая колонка '{col}' не найдена в DataFrame."
    SPLIT_FAILED = "Ошибка при разделении данных."
    NOT_SPLIT_YET = "Данные ещё не разделены. Сначала вызовите split()."


class LoaderError:
    FILE_NOT_FOUND = "Файл не найден: {path}"
    UNSUPPORTED_FORMAT = "Неподдерживаемый формат файла: {suffix}"
    LOAD_FAILED = "Ошибка при загрузке файла: {path}"
    NO_DATA_LOADED = "Данные ещё не загружены. Сначала вызовите load()."


class SaverError:
    FILE_EXISTS = (
        "Файл уже существует: {path}. Используйте overwrite=True для перезаписи."
    )
    UNSUPPORTED_FORMAT = "Неподдерживаемый формат файла: {suffix}"
    UNSUPPORTED_REPORT_FORMAT = (
        "Неподдерживаемый формат отчёта: {fmt}. Используйте 'json' или 'txt'."
    )
    UNSUPPORTED_IMAGE_FORMAT = (
        "Неподдерживаемый формат изображения: {suffix}. Используйте png, jpg, pdf."
    )
    SAVE_FAILED = "Ошибка при сохранении файла: {path}"
    MATPLOTLIB_NOT_AVAILABLE = "Matplotlib не установлен. Невозможно сохранить график."


class DatasetError:
    DATASET_EMPTY = "Dataset пуст. Сначала загрузите данные."
    TARGET_NOT_SET = "Целевая колонка не установлена."
    COLUMN_NOT_FOUND = "Колонка '{col}' не найдена в DataFrame."


class ModelError:
    MODEL_NOT_FITTED = "Модель ещё не обучена. Сначала вызовите fit()."
    INVALID_TASK = (
        "Неверный тип задачи: {task}. Используйте 'classification' или 'regression'."
    )
    PREDICTION_FAILED = "Ошибка при предсказании."
    TRAIN_FAILED = "Ошибка при обучении модели."
    EVALUATION_FAILED = "Ошибка при вычислении метрик."
    FEATURE_NAMES_REQUIRED = "Не указаны имена признаков."


class EvaluatorError:
    INVALID_TASK = (
        "Неверный тип задачи: {task}. Используйте 'classification' или 'regression'."
    )
    EVALUATION_FAILED = "Ошибка при вычислении метрик."
    PLOT_FAILED = "Ошибка при построении графика."


class VisualizationError:
    DATAFRAME_EMPTY = "DataFrame пуст. Невозможно построить график."
    COLUMN_NOT_FOUND = "Колонка '{col}' не найдена в DataFrame."
    PLOT_FAILED = "Ошибка при построении графика."
