class InputValidationError(Exception):
    def __str__(self):
        # Red color escape code for printing
        return f"\033[91m{self.args[0]}\033[0m" # Red text and reset after