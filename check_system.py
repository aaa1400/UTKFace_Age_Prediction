
import ctypes

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

memory = MEMORYSTATUSEX()
memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)

if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
    print("Total RAM:", round(memory.ullTotalPhys / (1024 ** 3), 2), "GB")
    print("Available RAM:", round(memory.ullAvailPhys / (1024 ** 3), 2), "GB")
    print("Memory Usage:", memory.dwMemoryLoad, "%")
else:
    print("Could not retrieve memory information.")