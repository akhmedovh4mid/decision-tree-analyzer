import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class AppLogger:
    """
    Централизованная настройка логирования приложения.
    """

    _configured: bool = False
    _log_dir: Path | None = None

    @classmethod
    def configure(
        cls,
        log_dir: str | Path = "logs",
        level: int = logging.INFO,
        console_level: int | None = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        """
        Выполняет глобальную настройку логирования для приложения.

        Args:
            log_dir: Директория, в которой будут храниться файлы логов.
            level: Уровень логирования для root-логгера и файлового обработчика.
            console_level: Уровень логирования для вывода в консоль.
                Если указано ``None``, вывод в консоль отключается.
            max_bytes: Максимальный размер файла лога в байтах,
                после которого выполняется ротация.
            backup_count: Количество архивных файлов логов,
                которые будут сохраняться.

        Returns:
            None
        """

        if cls._configured:
            return

        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = RotatingFileHandler(
            filename=cls._log_dir / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        if console_level is not None:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Возвращает логгер по имени.

        Args:
            name: Имя логгера.

        Returns:
            logging.Logger: Экземпляр логгера.
        """

        if not cls._configured:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            cls._configured = True

        return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """
    Упрощённая функция для получения логгера.

    Args:
        name: Имя логгера.

    Returns:
        logging.Logger: Экземпляр логгера с указанным именем.
    """
    return AppLogger.get_logger(name)
