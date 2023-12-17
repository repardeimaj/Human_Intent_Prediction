#! /usr/bin/env/ python3

#****************************************************************************#

#                             visualizer.py                                  #

#    file for easily vizualizing the graph structures and sequences          #

#****************************************************************************#


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML

import matplotlib.animation as animation



names = ["Pelvis", "Spine_Naval", "Spine_Chest", "Neck", "Clavicle_left", "Shoulder_left", "Elbow_left", "Wrist_left", "Hand_left", "Handtip_left", "thumb_left", "Clavicle_right", "Shoulder_right", "Elbow_right", "Wrist_right", "Hand_right", "Handtip_right", "Thumb_right", "Hip_left", "Knee_left", "Ankle_left", "Foot_left", "Hip_right", "Knee_right", "Ankle_right", "Foot_right", "Head", "Nose", "Eye_Left", "Ear_Left", "Eye_Right", "Ear_Right"]


#Generate a networx graph (bidirected) given joint data
def generateGraph(x,y,z):
    #useful for displaying the human 
    #and getting the adjacency matrix

    G = nx.Graph()
    for i in range(32):
        G.add_node(i, x=x[i], y=y[i],z=z[i])

    G.add_edge(0,1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 5)
    G.add_edge(5, 6)
    G.add_edge(6,7)
    G.add_edge(7, 8)
    G.add_edge(8, 9)
    G.add_edge(8, 10)
    G.add_edge(3, 26)
    G.add_edge(0, 18)
    G.add_edge(18, 19)
    G.add_edge(19, 20)
    G.add_edge(20, 21)
    G.add_edge(3, 11)
    G.add_edge(11, 12)
    G.add_edge(12, 13)
    G.add_edge(13, 14)
    G.add_edge(14, 15)
    G.add_edge(15, 16)
    G.add_edge(15, 17)
    G.add_edge(0, 22)
    G.add_edge(22,23)
    G.add_edge(23,24)
    G.add_edge(24,25)
    G.add_edge(27,26)
    G.add_edge(28,26)
    G.add_edge(29,26)
    G.add_edge(30,26)
    G.add_edge(31,26)

    return G

#get adjacency matrix from networx graph
def adj(G):
    return nx.to_numpy_array(G,dtype=np.float32)

#vizualize a single frame of graph data
def visualize_frame(data):

    x = [data[i][0] for i in range(len(data))]
    z = [- data[i][1] for i in range(len(data))]
    y = [data[i][2] for i in range(len(data))]



    G = generateGraph(x,y,z)

    pos = {node: (G.nodes[node]['x'], G.nodes[node]['z']) for node in G.nodes()}
    nx.draw(G, pos=pos, with_labels=False,node_size = 29)

    plt.gca().set_aspect('equal')


    plt.show()

#convert the trainig data to graph structure
def convert_frame(data):

    x = data[0]
    y = data[1]
    z = data[2]

    G = generateGraph(x,y,z)

    pos = {node: (G.nodes[node]['x'], G.nodes[node]['z']) for node in G.nodes()}
    nx.draw(G, pos=pos, with_labels=False,node_size = 29)
    return G, pos

#Generate mp4 of a time sequence of graphs
def visualize_sequence(sequence, title, speed):
    x = 25
    labels = ["crouching", "follow", "meet", "lifting","idle"]

    anim = None 

    fig, ax = plt.subplots()

    #uncomment for displaying prediction labels and porbibilities over the video
    # Add a text object for displaying probabilities
    #prob_text = fig.text(0.025, 0.8, "", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    def update(frame):
        ax.clear()
        G,pos = convert_frame(sequence[frame])
        nx.draw(G, pos=pos, with_labels=False, node_size=29)
        ax.set_aspect('equal')

    
        # prob_text.set_text("Probabilities:\nCrouching: {}\nFollow: {}\nMeet: {}\nLifting: {}\nIdle: {}".format(
        #     prediction[frame][0],
        #     prediction[frame][1],
        #     prediction[frame][2],
        #     prediction[frame][3],
        #     prediction[frame][4]
        # ))
        # ax.set_title('Frame: {}'.format(frame) + " \n" + "Predicted: " + labels[np.argmax(prediction[frame])] + "\n Actual: " + labels[label_sequence1[frame]])

        ax.set_title('Frame: {}, {}'.format(frame,title) )

        #  # Add a separate subplot for the probability text
        # prob_ax = fig.add_axes([0.7, 0.7, 0.2, 0.2])  # Adjust the position and size as needed
        # prob_ax.axis('off')
        # prob_ax.text(0, 0.5, prob_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        #Add the probability text in a fixed position on the screen
        # ax.annotate(prob_text, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10,
                    # bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    anim = animation.FuncAnimation(fig, update, frames=(len(sequence) - 1), interval=50 / speed)

    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save("/home/paternaincomputer/HIP_ws/" + title + '.mp4', writer = FFwriter)

        