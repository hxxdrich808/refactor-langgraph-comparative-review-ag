from rich.console import Console as RichConsole
from rich.text import Text

console = RichConsole()

def print(*args, **kwargs):
    """Wrapper around rich.print to maintain compatibility."""
    # Convert all arguments to strings and join with space
    message = " ".join(str(arg) for arg in args)
    console.print(message, **kwargs)
