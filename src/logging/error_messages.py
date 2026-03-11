class DataCleanerError:
    FILL_CONSTANT_MISSING_VALUE_ERROR = (
        "Для стратегии 'fill_constant' необходимо указать fill_value"
    )
    NON_NUMERIC_COLUMN_IQR_ERROR = "Для удаления выбросов из межквартильного размаха (IQR) столбец '{col}' должен быть числовым."
    MISSING_COLUMNS_NOT_FOUND_ERROR = (
        "Для отсутствующих данных по обработке отсутствуют корректные столбцы."
    )
    NON_NUMERIC_COLUMN_ZSCORE_ERROR = "Для удаления выбросов из Z-критерия столбец '{col}' должен содержать числовые значения."
