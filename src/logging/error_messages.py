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
