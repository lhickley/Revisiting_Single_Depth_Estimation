import argparse
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb
import cv2

from torchvision import transforms

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

def load_image(image_path):
    # Load image, then convert it to RGB and normalize it to [0, 1]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

transform = transforms.Compose([
    transforms.ToTensor(),
    ])

def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model)#.cuda()
    model.load_state_dict(torch.load('./pretrained_model/model_senet', map_location=torch.device('cpu')))
    model.eval()

    #nyu2_loader = loaddata.readNyu2('/Users/lewishickley/Downloads/InitialTestImage.jpg')
  
    test(model)


def test(model):
    img = load_image('/Users/lewishickley/Downloads/InitialTestImage.jpg')
    img = transform(img)
    img = img.unsqueeze(0).float()
    with torch.no_grad():
        output = model(img)

    output = output.cpu().numpy().squeeze()

    plt.imshow(output, cmap='inferno')
    plt.colorbar()
    plt.show()
    '''
    #for i, image in enumerate(nyu2_loader):     
        image = torch.autograd.Variable(image, volatile=True)
        output = model(image)

        plt.imshow(output, cmap='inferno')
        plt.colorbar()
        plt.show()
        
        #matplotlib.image.imsave('data/demo/out.png', out.view(out.size(2),out.size(3)).data.cpu().numpy())
        '''

if __name__ == '__main__':
    main()
