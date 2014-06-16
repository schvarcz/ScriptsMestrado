from morse.builder import *

#
# "cat" robot
#
cat = ATRV()
cat.translate(x=-6.0, z=0.2)

motion = MotionVW()
cat.append(motion)

pose = Pose()
cat.append(pose)

semanticL = VideoCamera()
semanticL.translate(x=0.2, y=0.3, z=0.9)
cat.append(semanticL)
    
semanticR = VideoCamera()
semanticR.translate(x=0.2, y=-0.3, z=0.9)
cat.append(semanticR)

motion.add_stream('ros')
semanticL.add_stream('ros')
semanticR.add_stream('ros')
pose.add_stream('ros')
cat.add_stream('ros')


keyboard = Keyboard()
keyboard.properties(Speed=3.0)
cat.append(keyboard)

#
# Environment
#
#env = Environment('land-1/buildings_2')
#env = Environment('indoors-1/indoor-1')
env = Environment('outdoors')
#env = Environment('./paris/pari.blend')
env.place_camera([10.0, -10.0, 10.0])
env.aim_camera([1.0470, 0, 0.7854])
env.select_display_camera(semanticL)
env.select_display_camera(semanticR)
