import pyrealsense2 as rs

# Create a context object. This manages all connected RealSense devices
context = rs.context()

# Get a list of all connected devices
connected_devices = context.query_devices()

# Print the number of connected devices
print(f"Number of connected RealSense devices: {len(connected_devices)}")

# Iterate over each device and print details
for i, device in enumerate(connected_devices):
    print(f"\nDevice {i + 1}")
    print(f"  Name: {device.get_info(rs.camera_info.name)}")
    print(f"  Serial Number: {device.get_info(rs.camera_info.serial_number)}")
    print(f"  Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")
    print(f"  USB Type Descriptor: {device.get_info(rs.camera_info.usb_type_descriptor)}")
    print(f"  Product ID: {device.get_info(rs.camera_info.product_id)}")
    print(f"  Camera Locked: {device.get_info(rs.camera_info.camera_locked)}")
    print(dir(device))
