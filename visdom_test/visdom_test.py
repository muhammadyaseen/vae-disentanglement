import numpy
import visdom
import numpy as np
import matplotlib.pyplot as plt
import time

class VisdomDataGather(object):

    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[], 
                    metric1=[], 
                    metric2=[],
                    images=[]
                )

    def insert(self, **kwargs):
        
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

viz_name = "Demo Visualization"
viz_port = 7089
visdom_instance = visdom.Visdom(port=viz_port) #, log_to_filename="visdom.log")
vis_graph_1, vis_graph_2 = None, None

# vis_graph_1 = visdom_instance.line(
#                     X=x_,
#                     Y=y_,
#                     env='Scalar_Metrics',
#                     opts=dict(
#                         width=400,
#                         height=400,
#                         xlabel='iteration',
#                         title='metric 1 tracking')
#                 )

visdom_instance.replay_log("visdom.log")

exit(0)

for x in range(1000):

    # simulate heavy computation
    time.sleep(1)

    # metric 1 display
    y_ = np.array([np.random.randn()])
    x_ = np.array([x])

    print(x_.shape, y_.shape)

    if vis_graph_1 is None:
        vis_graph_1 = visdom_instance.line(
                    X=x_,
                    Y=y_,
                    env='Scalar_Metrics',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title='metric 1 tracking')
                )
    else:
        vis_graph_1 = visdom_instance.line(
                    X=x_,
                    Y=y_,
                    env='Scalar_Metrics',
                    update='append',
                    win=vis_graph_1,
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title='metric 1 tracking')
                )