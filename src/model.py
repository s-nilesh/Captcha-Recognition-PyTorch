import torch
from torch import nn
from torch.nn import functional as F

class TextModel(nn.Module):
    def __init__(self, num_chars):
        super(TextModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 256, kernel_size=(3,3), padding=(1,1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv_2 = nn.Conv2d(256, 128, kernel_size=(3,3), padding=(1,1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv_3 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=(1,1))
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(2,2))

        self.linear_1 = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.5)

        self.lstm = nn.LSTM(64, 32, bidirectional=True, num_layers=2, dropout=0.5)
        self.output = nn.Linear(64, num_chars+1)

        
    def forward(self, images, targets=None):
        bs, c, h, w = images.size()
        # print(bs, c, h, w)

        # conv1 layer
        x = F.relu(self.conv_1(images))
        # print(x.size())
        x = self.max_pool_1(x)
        # print(x.size())

        # conv2 layer
        x = F.relu(self.conv_2(x))
        # print(x.size())
        x = self.max_pool_2(x)
        # print(x.size())

        # conv3 layer
        x = F.relu(self.conv_3(x))
        # print(x.size())
        # x = self.max_pool_3(x)      #1,32,9,18
        # print(x.size())     

        # bring the width to second position, because we want to see the width of the image when we apply RNN model
        x = x.permute(0, 3, 1, 2)   #1,18,32,9
        # print(x.size())             # 1152 values for 37 time steps

        # change the view of the image
        x = x.view(bs, x.size(1), -1)
        # print(x.size())     

        x = self.linear_1(x)       # now we have 37 time steps and for each time steps we have 64 values
        x = self.drop_1(x)        
        # print(x.size())     

        x, _ = self.lstm(x)
        # print(x.size())     

        x = self.output(x)
        # print(x.size())            # now we have 103 outputs for 37 time stamps

        # use CTC loss -> Connectionist Temporal Classification loss
        x = x.permute(1,0,2)     # to implement CTC loss
        if targets is not None:      # using loss function which makes sense with sequences
            log_softmax_values = F.log_softmax(x, 2)    # axis = 2 , where we have all our classes
            input_lengths = torch.full(
                size = (bs, ), 
                fill_value= log_softmax_values.size(0),
                dtype=torch.int32
            )
            print(input_lengths)
            target_lengths = torch.full(
                size = (bs, ), 
                fill_value= targets.size(1),
                dtype=torch.int32
            )
            print(target_lengths)
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_lengths, target_lengths 
            )
            return x, loss

        return x, None

if __name__ == '__main__':
    tm = TextModel(102)
    img = torch.rand(5, 3, 75, 150)      # initializing 5 random images, batch of 5 images
    target = torch.randint(1, 103, (5,3))
    x, loss = tm(img, target)