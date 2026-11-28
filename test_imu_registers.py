import struct
import unittest
from unittest.mock import patch

import imu


class FieldRegisterTests(unittest.TestCase):
    def test_split_and_bundled_notifications_keep_distinct_streams(self):
        motion = bytes.fromhex("55 61") + bytes(18)
        magnetic = bytes.fromhex("55 71 3a 00") + struct.pack("<3h", -300, 20, 1) + bytes(10)
        quaternion = bytes.fromhex("55 71 51 00") + struct.pack("<4h", 16384, -8192, 0, 0) + bytes(8)
        chip = bytes.fromhex("55 71 72 00 06 00") + bytes(14)
        stream = motion + chip + magnetic + quaternion + motion
        with patch.dict(imu.CONFIG, device_address=None):
            state = imu.start_imu(load_zero=False, record_session=True,
                                  register_addresses=imu.FIELD_REGISTERS, record_ble=True)
        imu.receive_data(state, stream[:23])
        imu.receive_data(state, stream[23:])
        self.assertEqual(len(state["session_samples"]), 2)
        first, second = state["ble_notifications"]
        self.assertEqual(first[1:], (1, 23, 1, 0, 0, 3))
        self.assertEqual(second[1:], (2, 77, 1, 3, 0, 0))
        for notification_id, sample in enumerate(state["session_samples"], 1):
            self.assertEqual(sample["ble_notification_id"], notification_id)
            self.assertEqual(sample["raw_hex"], motion.hex())
            self.assertEqual(sample["timing_epoch"], 0)
        self.assertNotIn("raw_hex", state["samples"][0])
        chip_reply, mag, quat = state["register_samples"]
        self.assertEqual(chip_reply["mag_chip_id"], 6)
        self.assertAlmostEqual(mag["mag_x_ut"], -2.0)
        self.assertNotIn("mag_x_ut", imu.decode_registers(magnetic))
        self.assertEqual((mag["mag_x_raw"], mag["mag_y_raw"], mag["mag_z_raw"]), (-300, 20, 1))
        self.assertEqual((quat["q0"], quat["q1"]), (0.5, -0.25))
        self.assertEqual(quat["raw_hex"], quaternion.hex())
        self.assertLessEqual(mag["receive_monotonic"], quat["receive_monotonic"])

    def test_status_and_environment_registers(self):
        prefix = bytes.fromhex("55 71 41 00")
        row = imu.decode_registers(prefix + struct.pack("<4Hii", 1, 0xffff, 0, 42, 100123, -275))
        self.assertEqual([row[f"d{i}_status_raw"] for i in range(4)], [1, 65535, 0, 42])
        self.assertEqual((row["pressure_pa"], row["barometric_height_m"]), (100123, -2.75))
        absent = imu.decode_registers(prefix + bytes(16))
        self.assertNotIn("pressure_pa", absent)
        self.assertNotIn("barometric_height_m", absent)
        temp = imu.decode_registers(bytes.fromhex("55 71 40 00") + struct.pack("<h", -350) + bytes(14))
        battery = imu.decode_registers(bytes.fromhex("55 71 64 00 78 01") + bytes(14))
        self.assertEqual(temp["temperature_c"], -3.5)
        self.assertEqual(battery["battery_v"], 3.76)
