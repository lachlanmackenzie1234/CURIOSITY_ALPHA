import os
from typing import NoReturn


def setup_pythonnet() -> bool:
    """Configure Python.NET to use .NET Core on ARM64.

    Note: The imports of 'clr' and 'System' are only available after Python.NET
    is properly initialized at runtime. They will show as unresolved in static analysis.

    Returns:
        bool: True if initialization was successful
    """
    os.environ["PYTHONNET_RUNTIME"] = "coreclr"
    os.environ["PYTHONNET_CORECLR"] = (
        "/usr/local/share/dotnet/shared/Microsoft.NETCore.App/9.0.2/libcoreclr.dylib"
    )

    try:
        # These imports will be available since Python.NET is already initialized
        import clr  # type: ignore
        import System  # type: ignore

        print(f"Python.NET initialized with .NET version: {System.Environment.Version}")
        return True
    except Exception as e:
        print(f"Error initializing Python.NET: {str(e)}")
        return False


if __name__ == "__main__":
    setup_pythonnet()
