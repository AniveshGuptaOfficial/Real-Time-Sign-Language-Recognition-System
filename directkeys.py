import ctypes
import time

# ---------------------------------------------------
# Access Windows API function SendInput
# Used to simulate keyboard input programmatically
# ---------------------------------------------------
SendInput = ctypes.windll.user32.SendInput

# ---------------------------------------------------
# Virtual Key Scan Codes
# 0x4D → Right key (example usage)
# 0x4B → Left key (example usage)
# These can be modified based on application
# ---------------------------------------------------
right_pressed = 0x4D
left_pressed = 0x4B

# ---------------------------------------------------
# Define C Structures required by Windows API
# Using ctypes to mimic Windows INPUT structure
# ---------------------------------------------------

PUL = ctypes.POINTER(ctypes.c_ulong)

# Keyboard Input Structure
class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),        # Virtual Key Code
        ("wScan", ctypes.c_ushort),      # Hardware Scan Code
        ("dwFlags", ctypes.c_ulong),     # Key event flags
        ("time", ctypes.c_ulong),        # Timestamp
        ("dwExtraInfo", PUL)             # Extra information
    ]

# Hardware Input Structure (rarely used here)
class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort)
    ]

# Mouse Input Structure (not used here but defined)
class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

# Union for input types
class Input_I(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),   # Keyboard
        ("mi", MouseInput),   # Mouse
        ("hi", HardwareInput) # Hardware
    ]

# Main Input Structure
class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),  # 1 = Keyboard input
        ("ii", Input_I)
    ]

# ---------------------------------------------------
# Function: PressKey
# Simulates pressing a key
# ---------------------------------------------------
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()

    # 0x0008 → KEYEVENTF_SCANCODE
    ii_.ki = KeyBdInput(
        0,
        hexKeyCode,
        0x0008,
        0,
        ctypes.pointer(extra)
    )

    x = Input(ctypes.c_ulong(1), ii_)

    # Send input to Windows
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


# ---------------------------------------------------
# Function: ReleaseKey
# Simulates releasing a key
# ---------------------------------------------------
def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()

    # 0x0008 → KEYEVENTF_SCANCODE
    # 0x0002 → KEYEVENTF_KEYUP (release flag)
    ii_.ki = KeyBdInput(
        0,
        hexKeyCode,
        0x0008 | 0x0002,
        0,
        ctypes.pointer(extra)
    )

    x = Input(ctypes.c_ulong(1), ii_)

    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


# ---------------------------------------------------
# Test Section
# This runs only if file is executed directly
# ---------------------------------------------------
if __name__ == '__main__':

    # Example: Press and release Ctrl key repeatedly
    while True:
        PressKey(0x11)     # 0x11 = Ctrl key
        time.sleep(1)

        ReleaseKey(0x11)
        time.sleep(1)