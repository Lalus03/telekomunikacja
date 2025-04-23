# xmodem.py

import time

SOH = 0x01
EOT = 0x04
ACK = 0x06
NAK = 0x15
CAN = 0x18
CRC = 0x43

PACKET_SIZE = 128

def calc_crc(data):
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
        crc &= 0xFFFF
    return crc

def calc_checksum(data):
    return sum(data) % 256

def send_file(ser, file_path, use_crc=True):
    handshake = CRC if use_crc else NAK
    mode_name = "CRC" if use_crc else "checksum"
    print(f"[TX] Waiting for receiver ({mode_name}) handshake...")

    # czekaj na NAK (0x15) lub 'C' (0x43)
    start = time.time()
    while True:
        if ser.in_waiting:
            ch = ser.read(1)
            print(f"[TX] Handshake recv: {ch.hex()}")
            if ch == bytes([handshake]):
                print(f"[TX] Receiver switched to {mode_name} mode.")
                break
        if time.time() - start > 10:
            print("[TX] No handshake, aborting.")
            return
        time.sleep(0.2)

    block = 1
    with open(file_path, "rb") as f:
        while True:
            data = f.read(PACKET_SIZE)
            if not data:
                break
            if len(data) < PACKET_SIZE:
                data += b'\0' * (PACKET_SIZE - len(data))
            pkt = bytearray([SOH, block % 256, 0xFF - (block % 256)]) + data
            if use_crc:
                c = calc_crc(data)
                pkt += bytes([c >> 8, c & 0xFF])
            else:
                chk = calc_checksum(data)
                pkt += bytes([chk])

            while True:
                print(f"[TX] Sending block {block}")
                ser.write(pkt)
                resp = ser.read(1)
                print(f"[TX] Response: {resp.hex() if resp else 'timeout'}")
                if resp == bytes([ACK]):
                    break
                if resp == bytes([NAK]):
                    print(f"[TX] NAK, resending block {block}")
                    continue
                if resp == bytes([CAN]):
                    print("[TX] Cancel from receiver.")
                    return
                print("[TX] Unknown resp, retrying...")
                time.sleep(0.5)

            block += 1

    print("[TX] Sending EOT")
    ser.write(bytes([EOT]))
    while True:
        r = ser.read(1)
        print(f"[TX] EOT resp: {r.hex() if r else 'timeout'}")
        if r == bytes([ACK]):
            print("[TX] Transfer complete.")
            break

def receive_file(ser, file_path, use_crc=True):
    handshake = CRC if use_crc else NAK
    mode_name = "CRC" if use_crc else "checksum"
    print(f"[RX] Starting in {mode_name} mode, sending handshake...")

    # wyślij handshakujący znak kilka razy
    for _ in range(10):
        ser.write(bytes([handshake]))
        print(f"[RX] Sent handshake: {handshake:#02x}")
        time.sleep(1)
        if ser.in_waiting:
            break

    with open(file_path, "wb") as f:
        block = 1
        while True:
            hdr = ser.read(1)
            if not hdr:
                continue
            if hdr == bytes([SOH]):
                blk = ser.read(1)[0]
                inv = ser.read(1)[0]
                if (blk + inv) & 0xFF != 0xFF:
                    print("[RX] Bad block #, NAK")
                    ser.write(bytes([NAK]))
                    continue
                data = ser.read(PACKET_SIZE)
                if use_crc:
                    rec = int.from_bytes(ser.read(2), "big")
                    calc = calc_crc(data)
                    if rec != calc:
                        print(f"[RX] CRC err {rec:04x}!={calc:04x}, NAK")
                        ser.write(bytes([NAK])); continue
                else:
                    rec = ser.read(1)[0]
                    calc = calc_checksum(data)
                    if rec != calc:
                        print(f"[RX] Chk err {rec:02x}!={calc:02x}, NAK")
                        ser.write(bytes([NAK])); continue

                if blk == block % 256:
                    f.write(data); block += 1; ser.write(bytes([ACK]))
                    print(f"[RX] Block {blk} OK, ACK")
                else:
                    print(f"[RX] Duplicate/out-of-order blk {blk}, ACK/NAK")
                    ser.write(bytes([ACK]))

            elif hdr == bytes([EOT]):
                ser.write(bytes([ACK]))
                print("[RX] EOT, done.")
                break
            else:
                print(f"[RX] Ignoring hdr {hdr.hex()}")
    print("[RX] Reception complete.")
