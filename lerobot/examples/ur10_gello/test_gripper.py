# run this standalone on the robot PC
from dynamixel_sdk import *

DEVICE = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO51RF-if00-port0"
BAUD   = 1_000_000   # or 1000000 depending on your gripper config
DXL_ID = 0       # check your gripper's ID, often 1 or 9

port = PortHandler(DEVICE)
packet = PacketHandler(2.0)  # protocol 2.0
port.openPort()
port.setBaudRate(BAUD)

val, result, error = packet.read1ByteTxRx(port, DXL_ID, 70)  # addr 70 = Hardware Error Status
print(f"Hardware Error Status: 0b{val:08b}")
print(result)
# Bit 0 = Input Voltage Error
# Bit 2 = Overheating Error
# Bit 3 = Motor Encoder Error
# Bit 5 = Overload Error   ← almost certainly this one
# Reboot clears the hardware error latch
packet.reboot(port, DXL_ID)
time.sleep(0.5)

# Then re-enable torque
TORQUE_ENABLE_ADDR = 64
packet.write1ByteTxRx(port, DXL_ID, TORQUE_ENABLE_ADDR, 1)
port.closePort()