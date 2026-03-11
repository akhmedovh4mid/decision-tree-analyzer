class DataCleanerLog:
    INIT = "DataCleaner инициализирован. shape=%s"
    COLUMN_NOT_FOUND = "Колонка '%s' отсутствует в DataFrame"
    START_HANDLE_MISSINGS = "Обработка пропущенных значений. strategy=%s columns=%s"
    MISSING_BEFORE = "Количество пропусков до обработки: %s"
    ROWS_DROPPED = "Удалено строк: %s"
    COLUMNS_DROPPED = "Удалены колонки: %s"
    FILLED_CONSTANT = "Пропуски заполнены константой: %s"
    FILLED_MEAN = "Колонка %s заполнена средним значением %s"
    NON_NUMERIC_COLUMN = "Колонка '%s' не числовая. %s пропущен."
    FILLED_MEDIAN = "Колонка %s заполнена медианой %s"
    FILLED_MODE = "Колонка %s заполнена модой %s"
    MODE_NOT_FOUND = "Колонка '%s' не имеет моды (все значения NaN?)"
    APPLY_FFILL = "Применён forward fill"
    APPLY_BFILL = "Применён backward fill"
    MISSING_AFTER = "Количество пропусков после обработки: %s"


class EncoderLog:
    INIT = "Инициализация Encoder. Размер DataFrame: %s"
    START_ENCODE = "Начато кодирование. Стратегия: '%s'. Колонки: %s"
    COLUMN_NOT_FOUND = "Колонка '%s' не найдена в DataFrame"
    COLUMN_MISSING_FOR_TRANSFORM = (
        "Колонка '%s' отсутствует в новом наборе данных при transform"
    )
    ENCODE_COMPLETE = "Кодирование завершено. Стратегия: '%s'"
    FITTED_ONEHOT = "Обучен OneHotEncoder для колонки '%s'"
    FITTED_LABEL = "Обучен label-encoder для колонки '%s'"
    FITTED_ORDINAL = "Обучен ordinal-encoder для колонки '%s'"
    FITTED_FREQUENCY = "Вычислено frequency-кодирование для колонки '%s'"
    FITTED_TARGET = "Вычислено target-кодирование для колонки '%s'"
    TRANSFORM_ONEHOT = (
        "Применено one-hot кодирование к колонке '%s'. Создано %s колонок"
    )
    TRANSFORM_LABEL = "Применено label-кодирование к колонке '%s'"
    TRANSFORM_ORDINAL = "Применено ordinal-кодирование к колонке '%s'"
    TRANSFORM_FREQUENCY = "Применено frequency-кодирование к колонке '%s'"
    TRANSFORM_TARGET = "Применено target-кодирование к колонке '%s'"
    MISSING_BIN_COLS = "Бинарные колонки для обратного преобразования отсутствуют: '%s'"
    NO_BIN_COLS_SAVED = "Не найдены сохранённые one-hot колонки для '%s'"
    UNKNOWN_ENCODER_TYPE = "Неизвестный тип энкодера для колонки '%s': %s"
    INVERSE_NOT_POSSIBLE = (
        "Обратное преобразование невозможно для колонки '%s' "
        "(frequency/target кодирование)"
    )


class ScalerLog:
    INIT = "Scaler инициализирован. shape=%s"
    COLUMN_NOT_FOUND = "Колонки '%s' отсутствуют в DataFrame"
    NON_NUMERIC_COLUMN = "Колонка '%s' не числовая – пропущена"
    START_SCALE = "Масштабирование. Стратегия: '%s'. Колонки: %s"
    FITTED_STANDARD = "StandardScaler обучен для колонки '%s'"
    FITTED_MINMAX = "MinMaxScaler обучен для колонки '%s'"
    FITTED_ROBUST = "RobustScaler обучен для колонки '%s'"
    FITTED_MAXABS = "MaxAbsScaler обучен для колонки '%s'"
    FITTED_QUANTILE = "QuantileTransformer обучен для колонки '%s'"
    FITTED_POWER = "PowerTransformer обучен для колонки '%s'"
    TRANSFORM_APPLIED = "Колонка '%s' масштабирована (стратегия: %s)"
    SCALE_COMPLETE = "Масштабирование завершено. Стратегия: '%s'"
    COLUMN_MISSING_FOR_TRANSFORM = (
        "Колонка '%s' отсутствует в новом наборе данных при transform"
    )
    NON_NUMERIC_COLUMN_TRANSFORM = "Колонка '%s' не числовая – пропущена при transform"
