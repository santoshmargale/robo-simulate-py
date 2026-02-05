# Sample Robot Control Program (INTENTIONALLY UNSAFE)

# PID Control Values
Kp = 10.0   # Very high proportional gain
Ki = 0.5
Kd = 0.1

# Motion Speed (Too High)
joint_speed = 3.5   # Safe limit in tutor app is 1.5

def move_joint(joint_name, angle):
    print(f"Moving {joint_name} to {angle} degrees at high speed!")

def close_gripper():
    print("Gripper closing immediately!")

# Infinite loop jogging motion
while True:
    move_joint("Joint1", 120)   # Sudden large angle jump
    move_joint("Joint2", -130)  # Another large jump
    close_gripper()             # No delay before closing
