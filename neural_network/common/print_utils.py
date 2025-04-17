class PrintUtils:
    BOLD = "\033[01m"

    GREEN = "\033[92m"
    YELLOW = "\033[93m"

    RESET = "\033[0m"

    @staticmethod
    def print_warning(message: str) -> None:
        print(f"{PrintUtils.YELLOW}{message}{PrintUtils.RESET}")

    @staticmethod
    def print_info(message: str) -> None:
        print(f"{PrintUtils.BOLD}{message}{PrintUtils.RESET}")

    @staticmethod
    def print_success(message: str) -> None:
        print(f"{PrintUtils.GREEN}{message}{PrintUtils.RESET}")

    @staticmethod
    def print_progress_bar(current, total):
        p: float = current / total
        print(f"Progress: [{'='*int(30*p):<30}] {current:>5} / {total} [{p*100:>6.2f}%]  ", end='\r')

    @staticmethod
    def print_noop(*args, **kwargs):
        pass