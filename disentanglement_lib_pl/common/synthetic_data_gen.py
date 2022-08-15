#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import argparse

import matplotlib.pyplot as plt
import os
import numpy as np

class FlowDataGenerator:

    def __init__(self, data_root):
        
        self.flow_path = os.path.join(data_root, "flow")
        self.check_dir()

    def generate_data(self):
        
        # variable name changes from orig. to this
        # r -> ball_radius
        # h_raw -> water_level_height
        # hole -> hole_position
        # hole_size
        # hole -> hole_level
        # deep -> hole_level_scaled

        HOLE_SIZE = 0.4
        plt.rcParams['figure.figsize'] = (1.0, 1.0)
        
        for ball_radius in range(5, 35):
            for water_level_height in range(10, 40):
                for hole_level in range(6, 15):
                    
                    ax = plt.gca()
                    ball_radius_scaled = ball_radius / 30.0
                    water_level_height_scaled = pow(ball_radius_scaled, 3) + water_level_height / 10.0 
                    hole_level_scaled = hole_level / 3.0
            
                    # Draw water and red ball
                    rect = plt.Rectangle((3., 0.), 5, 5 + water_level_height_scaled, color='lightskyblue')
                    ax.add_artist(rect)
                    ball = plt.Circle((5.5, ball_radius_scaled + 0.5), ball_radius_scaled, color = 'firebrick')
                    
                    # Draw left and rigth 'boundary' of the Cup
                    left = plt.Polygon(([3, 0],[3, 19]), color = 'black', linewidth = 2)
                    # because of the hole rigth bounrdary will be split into two wih a gap for hole
                    right_1 = plt.Polygon(([8, 0],[8, hole_level_scaled]), color = 'black', linewidth = 2)
                    right_2 = plt.Polygon(([8, hole_level_scaled + HOLE_SIZE],[8, 19]), color = 'black', linewidth = 2)
                    ax.add_artist(left)
                    ax.add_artist(right_1)
                    ax.add_artist(right_2)
                    ax.add_artist(ball)
            
                    # Water line
                    y = np.linspace(hole_level_scaled, 0.5)
                    # Noise N(0.01, 1) - but only positives
                    # TODO: this is probably buggy -- commented for now
                    # epsilon = 0.01 * np.max(np.abs(np.random.randn(1)),1)
                    #x_noisy = np.sqrt( 2 * (0.98 + epsilon) * water_level_height_scaled * (hole_level_scaled - y) ) + 8
                    # TODO: Toricelli's Law 
                    # -- Acc to Wikipedia '2' should be out of sqrt 
                    # -- No idea where 0.98 factor came from
                    x_true = np.sqrt ( 2 * 0.98 * water_level_height_scaled * (hole_level_scaled - 0.5))
                    plt.plot(x_true, y, color='lightskyblue', linewidth = 2)
            
                    # Ground
                    x = np.linspace(0, 20, num=50)
                    y = np.zeros(50) + 0.2
                    plt.plot(x,y,color='black',linewidth = 2)
                    
                    ax.set_xlim((0, 20))
                    ax.set_ylim((0, 20))
                    plt.axis('off')

                    plt.savefig(
                        os.path.join(self.flow_path, f'flow_r{ball_radius}_h{water_level_height}_x{x_true*10}_o{hole_level}.jpg'), 
                        dpi=96
                    )

                    plt.clf()
    
    def check_dir(self):

        if not os.path.exists(self.flow_path): 
            os.makedirs(self.flow_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Flow and Pendulum data gen')

    parser.add_argument('--dataset', help='Datase', required=True, choices=["flow", "pendulum"])
    parser.add_argument('--data_root', help='Data root', required=True)

    args = parser.parse_args()

    if args.dataset == "flow":
        flow_gen = FlowDataGenerator(args.data_root)
        flow_gen.generate_data()
