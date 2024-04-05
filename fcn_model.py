import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # Encoder
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Decoder
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
    # Pass input x through the feature blocks
     x1 = self.features_block1(x)  # Output after first block
     x2 = self.features_block2(x1)  # Output after second block
     x3 = self.features_block3(x2)  # Output after third block, for skip connection
     x4 = self.features_block4(x3)  # Output after fourth block, for skip connection
     x5 = self.features_block5(x4)  # Final feature extraction output
    
    # Apply the modified classifier to the last feature map
     score = self.classifier(x5)  # This is the score from the deepest layer
    
    # Upsample the score map by 2 using the first upscore layer
     upscore2 = self.upscore2(score)
    
    # Use the output from the fourth pooling layer (x4) for skip connection,
    # apply 1x1 convolution to get it to the same number of channels as the score map
     score_pool4 = self.score_pool4(x4)
    # The size of upscore2 might not match score_pool4 exactly due to
    # the convolution arithmetic, so we crop to match the sizes
     upscore2 = upscore2[:, :, 1:1+score_pool4.size()[2], 1:1+score_pool4.size()[3]]
    # Add the skip connection from pool4
     score_pool4 = score_pool4 + upscore2
    
    # Upsample the result of the skip connection
     upscore_pool4 = self.upscore_pool4(score_pool4)
    
    # Use the output from the third pooling layer (x3) for another skip connection,
    # similarly apply 1x1 convolution to match the number of channels
     score_pool3 = self.score_pool3(x3)
    # Again, ensure the sizes match for the addition by cropping
     upscore_pool4 = upscore_pool4[:, :, 1:1+score_pool3.size()[2], 1:1+score_pool3.size()[3]]
    # Add the skip connection from pool3
     score_pool3 = score_pool3 + upscore_pool4
    
    # Final upsampling to the input image size
     upscore_final = self.upscore_final(score_pool3)
    # We may need to crop the final upsampled output too, depending on your input size
    # Here's an example crop to match the input size (assuming your input height and width are divisible by 8)
     upscore_final = upscore_final[:, :, 4:4+x.size()[2], 4:4+x.size()[3]]
    
     return upscore_final

