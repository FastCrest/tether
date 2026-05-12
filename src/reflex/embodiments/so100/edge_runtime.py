# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Minimal arm wrapper for the bench wizard.

Replaces `arm_controller.py` for sandbox use. Has only what the calibration
wizard needs:
  - open serial + run setup_arm (PID + torque + homing offsets)
  - move to a safe centred home pose (no calibration required)
  - raw motor read/write helpers

Does NOT depend on calibration/model.py / kinematic IK / CAL_RANGES — those are downstream
artefacts produced by the wizard, not prerequisites for it.

Usage:
    from reflex.embodiments.so100.edge_runtime import SOArmHardware
    arm = SOArmHardware()
    arm.home()
    pose = arm.read_pose()
    ...
    arm.close()
"""
import time

# scservo_sdk is the SO-ARM 100 motor SDK. Hardware-only; not pip-installable
# in CI/test envs without the arm hardware. Guard the import so that pure-Python
# constants (PAN_ID / LIFT_ID / ...) and class definitions below remain
# importable. Calibration code can introspect motor-id constants without
# pulling the SDK; construction of `SOArmHardware()` raises a clear error
# when the SDK is missing.
try:
    import scservo_sdk as scs
    _SCS_AVAILABLE = True
except ImportError:
    scs = None  # type: ignore[assignment]
    _SCS_AVAILABLE = False

from reflex.embodiments.so100.calibration.arm_utils import setup_arm


class _SmsStsCompat:
    """Adapt scservo_sdk.sms_sts to the PacketHandler-style calls used here.

    This installed SDK binds the serial port into the packet object and exposes
    methods as read2ByteTxRx(id, addr), while the sandbox code was written
    against a PacketHandler-like shape: read2ByteTxRx(port, id, addr).
    """

    def __init__(self, port_h):
        self._pkt = scs.sms_sts(port_h)

    def read1ByteTxRx(self, _port_h, scs_id, address):
        return self._pkt.read1ByteTxRx(scs_id, address)

    def read2ByteTxRx(self, _port_h, scs_id, address):
        return self._pkt.read2ByteTxRx(scs_id, address)

    def write1ByteTxRx(self, _port_h, scs_id, address, data):
        return self._pkt.write1ByteTxRx(scs_id, address, data)

    def write2ByteTxRx(self, _port_h, scs_id, address, data):
        return self._pkt.write2ByteTxRx(scs_id, address, data)


def make_packet_handler(port_h):
    if hasattr(scs, "PacketHandler"):
        return scs.PacketHandler(0)
    return _SmsStsCompat(port_h)

# Motor IDs
PAN_ID = 1
LIFT_ID = 2
ELBOW_ID = 3
WRIST_FLEX_ID = 4
WRIST_ROLL_ID = 5
GRIPPER_ID = 6

ALL_IDS = [PAN_ID, LIFT_ID, ELBOW_ID, WRIST_FLEX_ID, WRIST_ROLL_ID, GRIPPER_ID]
NAMES = {PAN_ID: "pan", LIFT_ID: "lift", ELBOW_ID: "elbow",
         WRIST_FLEX_ID: "wrist_flex", WRIST_ROLL_ID: "wrist_roll",
         GRIPPER_ID: "gripper"}
IDS_BY_NAME = {name: mid for mid, name in NAMES.items()}

# SCServo register addresses (from feetech docs)
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
ADDR_TORQUE_ENABLE = 40

HOME_POSE = {
    "pan": 2082,
    "lift": 2089,
    "wrist_flex": 1500,
}
HOME_MOVE_STEPS = 24
HOME_MOVE_STEP_DELAY_S = 0.06
HOME_WRIST_LEAD_S = 0.02

# Note: NO hardcoded "home pose" here. Raw motor positions only make
# physical sense after `lerobot calibrate` has been run (which writes
# arm-specific homing offsets into so_follower/None.json). On a fresh
# arm with miscalibrated offsets, hardcoded positions could drive the
# arm into the table. The wizard avoids this by NOT issuing any goal
# commands during setup — `setup_arm()` pins goal=current so torque-on
# leaves the arm exactly where the user physically placed it.
#
# `home()` below is provided for downstream collect/eval scripts that
# can rely on lerobot calibration having been completed first. The
# wizard itself doesn't call it.


class SOArmHardware:
    def __init__(self, port="/dev/ttyACM0", baud=1_000_000):
        if not _SCS_AVAILABLE:
            raise ImportError(
                "SOArmHardware requires the scservo_sdk Python package, which "
                "is the Pi-side motor SDK for SO-ARM 100. Install via the "
                "`[so100]` extra: `pip install 'reflex-vla[so100]'`. "
                "(scservo_sdk is hardware-bundled; you typically only need it "
                "on the machine that's wired to the arm.)"
            )
        self.port_h = scs.PortHandler(port)
        if not self.port_h.openPort():
            raise RuntimeError(f"Could not open serial port {port}")
        self.port_h.setBaudRate(baud)
        self.ph = make_packet_handler(self.port_h)
        # PID + torque + homing offsets
        setup_arm(self.ph, self.port_h)

    # -- Raw motor I/O ---------------------------------------------------------

    def write_goal(self, mid, pos):
        self.ph.write2ByteTxRx(self.port_h, mid, ADDR_GOAL_POSITION, int(pos))

    def read_pos(self, mid):
        val, _, _ = self.ph.read2ByteTxRx(self.port_h, mid,
                                          ADDR_PRESENT_POSITION)
        return val

    def set_torque(self, enabled, mids=None):
        if mids is None:
            mids = ALL_IDS
        for mid in mids:
            self.ph.write1ByteTxRx(self.port_h, mid, ADDR_TORQUE_ENABLE,
                                   1 if enabled else 0)

    def read_pose(self):
        return {NAMES[mid]: self.read_pos(mid) for mid in ALL_IDS}

    # -- Compound -------------------------------------------------------------

    def move_pose_smooth(self, target, *, steps=HOME_MOVE_STEPS,
                         step_delay_s=HOME_MOVE_STEP_DELAY_S,
                         wrist_lead_s=HOME_WRIST_LEAD_S):
        cur = self.read_pose()
        names = [name for name in ("pan", "lift", "wrist_flex")
                 if name in target]
        for step in range(1, steps + 1):
            t = step / steps
            goals = {
                name: int(round(cur[name] + (target[name] - cur[name]) * t))
                for name in names
            }
            if "wrist_flex" in goals:
                self.write_goal(WRIST_FLEX_ID, goals["wrist_flex"])
                time.sleep(wrist_lead_s)
            for name in ("pan", "lift"):
                if name in goals:
                    self.write_goal(IDS_BY_NAME[name], goals[name])
            time.sleep(step_delay_s)

    def home(self, settle_s=0.4):
        """Move to a centred pose: pan/lift near mid, wrist_flex raised.

        REQUIRES lerobot calibration to be sound — raw motor units
        (2082, 2089, 1500) only mean "safe physical pose" when the arm's
        homing offsets are correctly written. Do not call from the
        bench wizard before homing offsets have been verified.
        """
        self.move_pose_smooth(HOME_POSE)
        time.sleep(settle_s)

    def close(self):
        self.port_h.closePort()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
