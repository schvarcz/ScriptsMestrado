import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,cm, imread
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

plt.ion()
f = plt.figure()
plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
f.canvas.get_tk_widget().update()

def plotVisualSLAM(robotPath, img, currentFrame, maxFrame, plotLimits = None, saveName = None):
    global f
    
    if plotLimits != None:
        (mx,nx), (my,ny), (mz,nz) = plotLimits
    f.clear()
    gs = GridSpec(3,2)
    ax1 = plt.subplot(gs[:-1,1])
    ax2 = plt.subplot(gs[0:,0])#, projection='3d')
    ax3 = plt.subplot(gs[-1,1])
    try:
        ax1.imshow(img)
        ax1.set_title("Frame {0}".format(currentFrame))
        ax2.clear()
        ax2.plot(robotPath[0],robotPath[1])#,robotPath[2],"b")
        ax2.set_xlabel("X Axis (m)")
        ax2.set_ylabel("Y Axis (m)")
        #ax2.set_zlabel("Z Axis (m)")
        if plotLimits != None:
            ax2.set_xlim(mx,nx)
            ax2.set_ylim(my,ny)
            #ax2.set_zlim(mz,nz)

        ax3.clear()
        ax3.plot(robotPath[4])
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Angle")
        ax3.set_xlim(0,maxFrame)
        ax3.set_ylim(-360,360)
        f.canvas.draw()
        if saveName!= None:
            f.savefig(saveName)
        f.canvas.get_tk_widget().update()
    except:
        pass

