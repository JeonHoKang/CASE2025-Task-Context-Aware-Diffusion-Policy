from pyRobotiqGripper import RobotiqGripper
import time



class ControlRobotiq:
    
    def __init__(self) -> None:
        self.gripper = RobotiqGripper(portname="/dev/ttyUSB1")
        self.gripper.activate()
        
        time.sleep(1)
        self.gripper.goTo(0)
    def send_gripper_command(self, position):
        position = position*255/0.8   
        self.gripper.goTo(position)

    def get_gripper_current_pose(self) -> float:
        current_position = self.gripper.getPosition()
        # print(current_position)
        current_position = current_position*0.8/255
        return current_position



def main():
    gripper = ControlRobotiq()
    gripper.send_gripper_command(0.0)
    # print(gripper.get_gripper_current_pose())
if __name__ == "__main__":
    main()