import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device("cuda")
prediction = torch.rand(2,2,3)
# for i in range(prediction.shape[1]):
#     print(prediction[0:2][i][:])
print("prediction", prediction + prediction)
# print("cat",torch.cat((prediction[0][1][:], prediction[1][1][:]),0))







# p = torch.add(prediction,prediction)
# p2 = prediction + prediction

# print(p)
# print(p2)
# print(prediction.shape[0])
# label = torch.rand(2,3)
# pos_weight = torch.full((2,3), 2.2, dtype=torch.float32)
# criterion  = nn.BCEWithLogitsLoss(reduction='sum', pos_weight = pos_weight)
# loss_bce = criterion(prediction, label)
# print(loss_bce)




# output = torch.randn(5,3)
# y = torch.randn(5,3)
# print(y)
# # weight = torch.tensor([0.1, 0.9])
# # a = y.data.view(-1).float()
# pos_weight = torch.cuda.DoubleTensor([5.00/3.00])
# print(5.00/3.00)
# # for i in range(4):
# #     print(weight[i])
# weight_ = weight[y.data.view(-1).long()].view_as(y)
# criterion = nn.BCELoss(reduce=False)
# loss = criterion(output, y)
# loss_class_weighted = loss * weight_
# loss_class_weighted = loss_class_weighted.mean()
