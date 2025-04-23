# main.py

import argparse
import serial
from xmodem import send_file, receive_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XModem File Transfer")
    parser.add_argument("--mode", choices=["send", "receive"], required=True, help="Mode: send or receive")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM3")
    parser.add_argument("--file", required=True, help="File to send or save to")
    parser.add_argument("--crc", action="store_true", help="Use CRC instead of checksum")

    args = parser.parse_args()

    try:
        ser = serial.Serial(
            args.port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=5
        )
    except Exception as e:
        print(f"Error opening serial port {args.port}: {e}")
        exit(1)

    if args.mode == "send":
        send_file(ser, args.file, use_crc=args.crc)
    else:
        receive_file(ser, args.file, use_crc=args.crc)

    ser.close()
