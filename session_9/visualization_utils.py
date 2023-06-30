import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter


class FeatureVisualization:
    def __init__(self, model):
        self.model = model
        self.model_weights = []
        self.conv_layers = []
        self.model_children = list(self.model.children())
        self.feature_maps = {}

    def get_conv_layers(self):
        for children in self.model_children:
            if type(children) == nn.Sequential:
                for child in children:
                    if type(child) == nn.Conv2d:
                        counter += 1
                        self.model_weights.append(child.weight)
                        self.conv_layers.append(child)
        print(f"Total convolution layers: {counter}")
        print("conv_layers")

    def _extract_processed_features(self, layer_names, layer_outputs):
        for cnt, layer_name, feature_map in enumerate(zip(layer_names,layer_outputs)):
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map,0)
            gray_scale = gray_scale / feature_map.shape[0]
            layer_id = f"{layer_name}_num_{cnt}"
            self.feature_maps[layer_id] = gray_scale.data.cpu().numpy()

    def compute_features(self):
        #counter to keep count of the conv layers
        counter = 0
        # get all the model children as list
        layer_outputs = []
        layer_names = []
        for layer in self.conv_layers[0:]:
            image = layer(image)
            layer_outputs.append(image)
            layer_names.append(str(layer))
        print(len(layer_outputs))
        self._extract_processed_features(layer_names, layer_outputs)

    def plot_feature_weights(self):
        # visualize the first conv layer filters
        plt.figure(figsize=(5, 4))
        first_layer_weights = self.model_weights[0].cpu()
        for i, filter in enumerate(first_layer_weights):
            plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
            plt.imshow(filter[0, :, :].detach(), cmap='gray')
            plt.axis('off')
        plt.show()

    def plot_feature_maps(self):
        fig = plt.figure(figsize=(6, 10))
        cnt = 0
        for layer, feature in self.feature_maps.items():
            a = fig.add_subplot(5, 4, cnt+1)
            imgplot = plt.imshow(feature[cnt])
            a.axis("off")
            a.set_title(layer.split('(')[0], fontsize=10)
        plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')

    def visualize_model_features(self):
        # extract and plot feature weights
        self.get_conv_layers()
        self.plot_feature_weights()
        # extract feature maps
        self.compute_features()
        self.plot_feature_maps()


def build_confusion_matrix(model, data_loader):
    writer = SummaryWriter('doc')
    images,labels = next(data_loader)
    writer.add_graph(model, images)
    writer.close()

