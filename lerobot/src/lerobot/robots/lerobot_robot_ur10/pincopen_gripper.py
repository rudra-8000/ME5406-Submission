"""
PincOpen Gripper Controller — Dynamixel XM430-W350-T via U2D2 (Protocol 2.0)

Designed for integration with the LeRobot UR10 robot plugin.
Exposes a normalized [0.0, 1.0] interface:
    0.0 = fully open  (open_angle  = 285.0°)
    1.0 = fully closed (close_angle = 166.0°)
"""

import time
import dynamixel_sdk as dxl

# ── Control Table Addresses ────────────────────────────────────────────────
ADDR_OPERATING_MODE   = 11
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_CURRENT     = 102
ADDR_PROFILE_ACCEL    = 108
ADDR_PROFILE_VEL      = 112
ADDR_GOAL_POSITION    = 116
ADDR_MOVING           = 122
ADDR_PRESENT_POSITION = 132

# ── Motor Constants ────────────────────────────────────────────────────────
TORQUE_ENABLE          = 1
TORQUE_DISABLE         = 0
CURRENT_BASED_POS_MODE = 5
POS_RESOLUTION         = 4096      # ticks/rev, 0.088 deg/tick
MAX_PROFILE_VEL        = 200       # ~45.8 rpm
MAX_CURRENT_UNITS      = 1193      # default Current Limit; 1 unit ≈ 2.69 mA
POSITION_TOLERANCE     = 15        # ticks
MOTION_TIMEOUT_S       = 10.0
MOVING_ARM_TIMEOUT_S   = 0.25
POLL_INTERVAL_S        = 0.02
PROTOCOL_VER           = 2.0


# ── Helpers ────────────────────────────────────────────────────────────────

def _deg_to_ticks(deg: float) -> int:
    return int(round(deg * POS_RESOLUTION / 360.0))

def _ticks_to_deg(ticks: int) -> float:
    return ticks * 360.0 / POS_RESOLUTION

def _check_comm(result, error, ph, label: str) -> bool:
    if result != dxl.COMM_SUCCESS:
        print(f"[ERROR] {label}: {ph.getTxRxResult(result)}")
        return False
    if error != 0:
        print(f"[WARN]  {label} hw error: {ph.getRxPacketError(error)}")
    return True

def _wait_motion(port_handler, packet_handler, dxl_id) -> bool:
    """Two-phase wait: arm (MOVING→1) then complete (MOVING→0)."""
    arm_deadline = time.time() + MOVING_ARM_TIMEOUT_S
    while time.time() < arm_deadline:
        moving, res, err = packet_handler.read1ByteTxRx(port_handler, dxl_id, ADDR_MOVING)
        if not _check_comm(res, err, packet_handler, "Read MOVING (arm)"):
            return False
        if moving == 1:
            break
        time.sleep(POLL_INTERVAL_S)

    deadline = time.time() + MOTION_TIMEOUT_S
    while time.time() < deadline:
        moving, res, err = packet_handler.read1ByteTxRx(port_handler, dxl_id, ADDR_MOVING)
        if not _check_comm(res, err, packet_handler, "Read MOVING (complete)"):
            return False
        if moving == 0:
            return True
        time.sleep(POLL_INTERVAL_S)

    print(f"[ERROR] Motion timeout ({MOTION_TIMEOUT_S}s).")
    return False


# ── Main Class ─────────────────────────────────────────────────────────────

class GripperController:
    """
    Persistent-connection controller for the PincOpen gripper.

    Normalized interface for LeRobot:
        0.0 → fully open  (open_angle)
        1.0 → fully closed (close_angle)
    """

    def __init__(
        self,
        port: str         = "/dev/ttyUSB0",
        baud: int         = 1_000_000,
        dxl_id: int       = 0,
        open_angle: float  = 285.0,
        close_angle: float = 166.0,
        default_speed: float  = 1.0,
<<<<<<< HEAD
        default_torque: float = 0.2,
=======
        default_torque: float = 0.4,
>>>>>>> 86760d5fb4b2b916c91d8e3736abf60f4663a89b
    ):
        self.port          = port
        self.baud          = baud
        self.dxl_id        = dxl_id
        self.open_angle    = open_angle
        self.close_angle   = close_angle
        self.default_speed  = max(0.01, min(1.0, default_speed))
        self.default_torque = max(0.01, min(1.0, default_torque))
        self._connected    = False

        self.port_handler   = dxl.PortHandler(port)
        self.packet_handler = dxl.PacketHandler(PROTOCOL_VER)
        self._connect()

    # ── Init ───────────────────────────────────────────────────────────────

    def _connect(self):
        if not self.port_handler.openPort():
            raise IOError(f"Cannot open gripper port: {self.port}")
        if not self.port_handler.setBaudRate(self.baud):
            raise IOError(f"Cannot set baud rate: {self.baud}")

        # ── Clear any latched hardware error from previous session ────────
        hw_err, res, _ = self.packet_handler.read1ByteTxRx(
            self.port_handler, self.dxl_id, 70)  # Addr 70 = Hardware Error Status
        if res == dxl.COMM_SUCCESS and hw_err != 0:
            print(f"[Gripper] Hardware error detected (0x{hw_err:02x}), rebooting motor...")
            self.packet_handler.reboot(self.port_handler, self.dxl_id)
            time.sleep(1.0)  # XM430 needs ~1s to come back after reboot
            print("[Gripper] Reboot complete.")

        for addr, val, nbytes, label in [
            (ADDR_TORQUE_ENABLE,  TORQUE_DISABLE,         1, "torque off"),
            (ADDR_OPERATING_MODE, CURRENT_BASED_POS_MODE, 1, "current-based pos mode"),
            (ADDR_TORQUE_ENABLE,  TORQUE_ENABLE,          1, "torque on"),
        ]:
            
    # def _connect(self):
    #     if not self.port_handler.openPort():
    #         raise IOError(f"Cannot open gripper port: {self.port}")
    #     if not self.port_handler.setBaudRate(self.baud):
    #         raise IOError(f"Cannot set baud rate: {self.baud}")

    #     for addr, val, nbytes, label in [
    #         (ADDR_TORQUE_ENABLE,  TORQUE_DISABLE,         1, "torque off"),
    #         (ADDR_OPERATING_MODE, CURRENT_BASED_POS_MODE, 1, "current-based pos mode"),
    #         (ADDR_TORQUE_ENABLE,  TORQUE_ENABLE,          1, "torque on"),
    #     ]:
            fn = (self.packet_handler.write1ByteTxRx if nbytes == 1
                  else self.packet_handler.write4ByteTxRx)
            res, err = fn(self.port_handler, self.dxl_id, addr, val)
            if not _check_comm(res, err, self.packet_handler, label):
                raise RuntimeError(f"Gripper init failed at: {label}")

        self._connected = True
        print(f"[Gripper] Connected on {self.port} | ID={self.dxl_id} | mode=CurrentBasedPos")

    def reboot_if_error(self) -> bool:
        """
        Check hardware error register (addr 70) and reboot+reinit if latched.
        Call this at the start of every episode reset.
        Returns True if a reboot was performed.
        """
        hw_err, res, _ = self.packet_handler.read1ByteTxRx(
            self.port_handler, self.dxl_id, 70)
        if res != dxl.COMM_SUCCESS or hw_err == 0:
            return False

        print(f"[Gripper] Hardware error 0x{hw_err:02x} — rebooting...")
        self.packet_handler.reboot(self.port_handler, self.dxl_id)
        time.sleep(1.5)

        for addr, val, nbytes, label in [
            (ADDR_TORQUE_ENABLE,  TORQUE_DISABLE,         1, "torque off"),
            (ADDR_OPERATING_MODE, CURRENT_BASED_POS_MODE, 1, "current-based pos mode"),
            (ADDR_TORQUE_ENABLE,  TORQUE_ENABLE,          1, "torque on"),
        ]:
            fn = (self.packet_handler.write1ByteTxRx if nbytes == 1
                else self.packet_handler.write4ByteTxRx)
            res, err = fn(self.port_handler, self.dxl_id, addr, val)
            _check_comm(res, err, self.packet_handler, label)

        print("[Gripper] Reboot + reinit complete.")
        return True
    # ── Internal writes ────────────────────────────────────────────────────

    def _w4(self, addr, val, label): 
        res, err = self.packet_handler.write4ByteTxRx(self.port_handler, self.dxl_id, addr, val)
        return _check_comm(res, err, self.packet_handler, label)

    def _w2(self, addr, val, label): 
        res, err = self.packet_handler.write2ByteTxRx(self.port_handler, self.dxl_id, addr, val)
        return _check_comm(res, err, self.packet_handler, label)

    def _w1(self, addr, val, label): 
        res, err = self.packet_handler.write1ByteTxRx(self.port_handler, self.dxl_id, addr, val)
        return _check_comm(res, err, self.packet_handler, label)

    # ── Core move ──────────────────────────────────────────────────────────

    def move(self, angle: float, speed: float = None, torque: float = None) -> int:
        """Move to angle (degrees). Returns 1=success, 0=failure."""
        if not self._connected:
            print("[ERROR] Gripper not connected.")
            return 0

        speed  = self.default_speed  if speed  is None else max(0.01, min(1.0, speed))
        torque = self.default_torque if torque is None else max(0.01, min(1.0, torque))

        lo    = min(self.open_angle, self.close_angle)
        hi    = max(self.open_angle, self.close_angle)
        angle = max(lo, min(hi, angle))

        goal_ticks   = _deg_to_ticks(angle)
        profile_vel  = max(1, int(round(speed  * MAX_PROFILE_VEL)))
        goal_current = max(1, min(MAX_CURRENT_UNITS, int(round(torque * MAX_CURRENT_UNITS))))

        try:
            if not self._w4(ADDR_PROFILE_VEL,  profile_vel,  "profile vel"):  return 0
            if not self._w2(ADDR_GOAL_CURRENT,  goal_current, "goal current"): return 0
            if not self._w4(ADDR_GOAL_POSITION, goal_ticks,   "goal pos"):     return 0

            if not _wait_motion(self.port_handler, self.packet_handler, self.dxl_id):
                return 0

            present, res, err = self.packet_handler.read4ByteTxRx(
                self.port_handler, self.dxl_id, ADDR_PRESENT_POSITION)
            if not _check_comm(res, err, self.packet_handler, "read pos"):
                return 0

            present    = present & 0xFFFFFFFF
            tick_error = abs(present - goal_ticks)

            if tick_error > POSITION_TOLERANCE:
                # In current-based mode, stopping before goal = valid contact
                print(f"[Gripper] Torque-limited stop at {_ticks_to_deg(present):.1f}° "
                      f"(goal {angle:.1f}°, Δ={tick_error} ticks)")
                return 1

            print(f"[Gripper] Reached {_ticks_to_deg(present):.1f}° (Δ {tick_error} ticks)")
            return 1

        except Exception as e:
            print(f"[Gripper EXCEPTION] {e}")
            return 0

    def move_non_blocking(
        self, angle: float, speed: float = None, torque: float = None
    ) -> int:
        """Send goal command without waiting for motion completion (GELLO-compatible)."""
        if not self._connected:
            print("[ERROR] Gripper not connected.")
            return 0

        speed = self.default_speed if speed is None else max(0.01, min(1.0, speed))
        torque = self.default_torque if torque is None else max(0.01, min(1.0, torque))

        lo = min(self.open_angle, self.close_angle)
        hi = max(self.open_angle, self.close_angle)
        angle = max(lo, min(hi, angle))

        goal_ticks = _deg_to_ticks(angle)
        profile_vel = max(1, int(round(speed * MAX_PROFILE_VEL)))
        goal_current = max(1, min(MAX_CURRENT_UNITS, int(round(torque * MAX_CURRENT_UNITS))))

        try:
            if not self._w4(ADDR_PROFILE_VEL, profile_vel, "profile vel"):
                return 0
            if not self._w2(ADDR_GOAL_CURRENT, goal_current, "goal current"):
                return 0
            if not self._w4(ADDR_GOAL_POSITION, goal_ticks, "goal pos"):
                return 0
            return 1
        except Exception as e:
            print(f"[Gripper EXCEPTION] {e}")
            return 0

    # ── Normalized interface (used by ur10.py) ─────────────────────────────

    def get_pos_normalized(self) -> float:
        """
        Read current position as a normalized value in [0.0, 1.0].
        0.0 = fully open, 1.0 = fully closed.
        Returns -1.0 on communication error.
        """
        ticks, res, err = self.packet_handler.read4ByteTxRx(
            self.port_handler, self.dxl_id, ADDR_PRESENT_POSITION)
        if not _check_comm(res, err, self.packet_handler, "read pos normalized"):
            return -1.0
        ticks  = ticks & 0xFFFFFFFF
        deg    = _ticks_to_deg(ticks)
        # Clamp and normalize: open_angle→0.0, close_angle→1.0
        lo, hi = min(self.open_angle, self.close_angle), max(self.open_angle, self.close_angle)
        deg    = max(lo, min(hi, deg))
        # open_angle > close_angle for this gripper (285 open, 166 closed)
        # so invert the direction:
        return (self.open_angle - deg) / (self.open_angle - self.close_angle)

    def set_pos_normalized(self, value: float, speed: float = None, torque: float = None) -> int:
        """
        Move to a normalized position [0.0, 1.0].
        0.0 = fully open, 1.0 = fully closed.
        """
        value = max(0.0, min(1.0, value))
        # Invert: 0.0→open_angle, 1.0→close_angle
        angle = self.open_angle - value * (self.open_angle - self.close_angle)
        return self.move(angle, speed=speed, torque=torque)

    def set_pos_normalized_non_blocking(
        self, value: float, speed: float = None, torque: float = None
    ) -> int:
        """Normalized [0,1] command without blocking on motion completion."""
        value = max(0.0, min(1.0, value))
        angle = self.open_angle - value * (self.open_angle - self.close_angle)
        return self.move_non_blocking(angle, speed=speed, torque=torque)


    def set_pos_normalized_async(self, value: float, speed: float = None, torque: float = None) -> bool:
        """
        Non-blocking move — writes goal position and returns immediately.
        Use this during teleoperation. Never calls _wait_motion().
        """
        if not self._connected:
            return False

        value  = max(0.0, min(1.0, value))
        angle  = self.open_angle - value * (self.open_angle - self.close_angle)
        speed  = self.default_speed  if speed  is None else max(0.01, min(1.0, speed))
        torque = self.default_torque if torque is None else max(0.01, min(1.0, torque))

        lo, hi = min(self.open_angle, self.close_angle), max(self.open_angle, self.close_angle)
        angle  = max(lo, min(hi, angle))

        goal_ticks   = _deg_to_ticks(angle)
        profile_vel  = max(1, int(round(speed * MAX_PROFILE_VEL)))
        goal_current = max(1, min(MAX_CURRENT_UNITS, int(round(torque * MAX_CURRENT_UNITS))))

        self._w4(ADDR_PROFILE_VEL,  profile_vel,  "profile vel")
        self._w2(ADDR_GOAL_CURRENT, goal_current, "goal current")
        self._w4(ADDR_GOAL_POSITION, goal_ticks,  "goal pos")
        return True
    # ── Convenience ────────────────────────────────────────────────────────

    def open(self, speed=None, torque=None):
        return self.move(self.open_angle, speed, torque)

    def close(self, speed=None, torque=None):
        return self.move(self.close_angle, speed, torque)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def disconnect(self):
        if self._connected:
            self._w1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque off (shutdown)")
            self.port_handler.closePort()
            self._connected = False
            print("[Gripper] Disconnected.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disconnect()
