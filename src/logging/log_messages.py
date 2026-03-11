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


class SplitterLog:
    INIT = "Splitter инициализирован. shape=%s"
    SPLIT_COMPLETE = "Разделение завершено. Обучающая выборка: %s строк, тестовая: %s строк. Целевая колонка: %s"


class LoaderLog:
    INIT = "DataLoader инициализирован."
    START_LOAD = "Загрузка файла: %s, формат: %s"
    LOAD_SUCCESS = "Файл успешно загружен. Размер данных: %s"
    CLEAR = "Данные очищены."


class SaverLog:
    INIT = "DataSaver инициализирован."
    START_SAVE_DATA = "Сохранение данных в файл: %s, формат: %s"
    START_SAVE_MODEL = "Сохранение модели в файл: %s"
    START_SAVE_REPORT = "Сохранение отчёта в файл: %s, формат: %s"
    START_SAVE_FIGURE = "Сохранение графика в файл: %s, формат: %s"
    SAVE_SUCCESS = "Файл успешно сохранён: %s"
    MODEL_SAVED_JOBLIB = "Модель сохранена с помощью joblib"
    MODEL_SAVED_PICKLE = "Модель сохранена с помощью pickle"
    USE_PICKLE_INSTEAD_OF_JOBLIB = "joblib не доступен, используется pickle"
    CLEAR_LAST = "Информация о последнем сохранении сброшена."


class DatasetLog:
    INIT = "Dataset инициализирован."
    DATA_LOADED = "Dataset загружен из файла: %s"
    TARGET_SET = "Установлена целевая колонка: %s"
    DATA_UPDATED = "Dataset обновлён. Новый размер: %s"
    DATASET_CLEARED = "Dataset очищен."


class ModelLog:
    INIT = "DecisionTreeModel инициализирован. Тип задачи: %s"
    START_TRAIN = "Начато обучение дерева решений."
    TRAIN_COMPLETE = "Обучение завершено."
    START_PREDICT = "Выполняется предсказание."
    PREDICT_COMPLETE = "Предсказание завершено."
    START_EVALUATE = "Вычисление метрик модели."
    EVALUATE_COMPLETE = "Метрики вычислены."
    FEATURE_IMPORTANCE = "Вычислена важность признаков."
    TREE_PLOTTED = "Граф дерева решений построен."


class EvaluatorLog:
    INIT = "Evaluator инициализирован. Тип задачи: %s"
    START_EVALUATION = "Начато вычисление метрик."
    EVALUATION_COMPLETE = "Метрики успешно вычислены."
    CONFUSION_MATRIX = "Матрица ошибок построена."
    CROSS_VALIDATION = "Выполнена перекрёстная проверка модели."
    PLOT_CREATED = "График оценки модели построен."


class VisualizationLog:
    HISTOGRAM = "Построена гистограмма для колонки: %s"
    BOXPLOT = "Построен boxplot для колонки: %s"
    CORRELATION = "Построена корреляционная матрица."
    SCATTER = "Построен scatter plot: %s vs %s"
    FEATURE_IMPORTANCE = "Построен график важности признаков."
    CONFUSION_MATRIX = "Построена визуализация confusion matrix."


class TreePlotterLog:
    TREE_PLOT_CREATED = "Граф дерева решений успешно построен."
    TREE_EXPORT_TEXT = "Текстовое представление дерева успешно создано."


class ControllerLog:
    INIT = "Controller инициализирован"
    DATA_LOADED = "Данные загружены. Shape=%s"
    DATA_SAVED = "Данные сохранены в %s"
    TARGET_SET = "Целевая колонка установлена: %s"
    DATA_SCALED = "Данные масштабированы стратегией %s"
    DATA_SPLIT = "Данные разделены на train/test"
    MODEL_TRAINED = "Модель дерева решений обучена"
    PREDICTION_DONE = "Предсказания выполнены"
    METRICS_COMPUTED = "Метрики вычислены"
    FEATURE_IMPORTANCE = "Важность признаков рассчитана"
