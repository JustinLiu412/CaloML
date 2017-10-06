from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import os
from torch import cat

execfile("loader.py")

OutPath='best/'
OutPath=os.getcwd()+'/'+OutPath

def objective(params):
# define the model
    depth, width = params
    learning_rate=0.001
    decay_rate=0
    class Net2(nn.Module):
        def __init__(self):
            super(Net2, self).__init__()
            self.fc1 = nn.Linear(25 * 25 * 25 + 5 * 5 * 60, width)
            self.fc5 = nn.Linear(width, width)
            self.fc4 = nn.Linear(width, 2)

        def forward(self, x1, x2):
            x1 = x1.view(-1, 25 * 25 * 25)
            x2 = x2.view(-1, 5 * 5 * 60)
        
            x = cat((x1,x2), 1)

            x = F.relu(self.fc1(x))
            for _ in range(depth-1):
                x = F.relu(self.fc5(x))
            x = self.fc4(x)

            return x


    from torch import load

    #net = nn.DataParallel(Net2())
    net = Net2()

# load previous model
    #net.load_state_dict(load("../Downsampled_GammaPi0_1_merged_nn_outputs/savedmodel_32-6_lr-0.0002_dr-0_10-15"))

    net.cuda()

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay_rate)


    loss_history = []


    epoch_num=20


# main process for training
    prev_val_loss = 0.0
    stag_break_count = 0
    over_break_count = 0
    prev_epoch_end_val_loss = 0
    epoch_end_val_loss = 0
    for epoch in range(epoch_num):
        running_loss = 0.0
        break_flag = False
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            ECAL, HCAL, labels = Variable(inputs[0].cuda()), Variable(inputs[1].cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = net(ECAL, HCAL)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 20 == 19:
                print('[%d, %5d] loss: %.10f' %
                        (epoch + 1, i + 1, running_loss)),


                val_loss = 0.0
                for _, data in enumerate(val_loader, 0):
                    inputs, labels = data
                    ECAL, HCAL, labels = Variable(inputs[0].cuda()), Variable(inputs[1].cuda()), Variable(labels.cuda())
                    outputs = net(ECAL, HCAL)
                    loss = criterion(outputs, labels)
                    val_loss += loss.data[0]

                print('    val loss: %.10f' %
                        (val_loss)),


                loss_history.append([epoch + 1, i + 1, running_loss, val_loss])

                relative_error = (val_loss-prev_val_loss)/float(val_loss),
                print('    relative error: %.10f' %
                        (relative_error))
                if(relative_error>0.05):
                    over_break_count+=1
                    if(over_break_count>3):
                        break_flag=True
                        break
                else:
                    over_break_count=0
                
                print('    over break count: %d' %
                        (over_break_count))

                if(i+1==200):
                    epoch_end_val_loss = val_loss
                    epoch_end_relative_error = (epoch_end_val_loss-prev_epoch_end_val_loss)/float(epoch_end_val_loss)
                    print('    epoch_end_relative_error: %.10f' %
                            (epoch_end_relative_error)),
                    if(epoch_end_relative_error > -0.01 and epoch!=0):
                        stag_break_count+=1
                        if(stag_break_count>0):
                            break_flag=True
                            break
                    else:
                        stag_break_count=0

                    print('    stag_break_count: %d' %
                            (stag_break_count))

                    prev_epoch_end_val_loss = epoch_end_val_loss
                prev_val_loss = val_loss
                running_loss = 0.0
        if(break_flag):
            break;

    loss_history=np.array(loss_history)
    with h5py.File(OutPath+"loss_history-depth_"+str(depth)+"-width_"+str(width)+".h5", 'w') as loss_file:
        loss_file.create_dataset("loss", data=loss_history)

    from torch import save
    save(net.state_dict(), OutPath+"savedmodel_depth_"+str(depth)+"-width_"+str(width))

    print('Finished Training')

# Analysis
    from torch import max

    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        ECAL, HCAL, labels = Variable(images[0].cuda()), Variable(images[1].cuda()), labels.cuda()
        outputs = net(ECAL, HCAL)
        _, predicted = max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on test images: %f %%' % (
            100 * float(correct) / total))


    train_ele_gen = iter(train_ele_loader)
    test_ele_gen = iter(test_ele_loader)

    train_chpi_gen = iter(train_chpi_loader)
    test_chpi_gen = iter(test_chpi_loader)

    from torch import Tensor

    outputs0=Tensor().cuda()
    outputs1=Tensor().cuda()
    outputs2=Tensor().cuda()
    outputs3=Tensor().cuda()


#  separate outputs for training/testing signal/backroung events. 
    for data in train_ele_gen:
        images, labels = data
        outputs0 = cat((outputs0, net(Variable(images[0].cuda()),Variable(images[1].cuda())).data))

    for data in test_ele_gen:
        images, labels = data
        outputs1 = cat((outputs1, net(Variable(images[0].cuda()),Variable(images[1].cuda())).data))

    for data in train_chpi_gen:
        images, labels = data
        outputs2 = cat((outputs2, net(Variable(images[0].cuda()),Variable(images[1].cuda())).data))

    for data in test_chpi_gen:
        images, labels = data
        outputs3 = cat((outputs3, net(Variable(images[0].cuda()),Variable(images[1].cuda())).data))

    with h5py.File(OutPath+"out_depth_"+str(depth)+"-width_"+str(width)+"_0.h5", 'w') as o1, h5py.File(OutPath+"out_depth_"+str(depth)+"-width_"+str(width)+"_1.h5", 'w') as o2, h5py.File(OutPath+"out_depth_"+str(depth)+"-width_"+str(width)+"_2.h5", 'w') as o3, h5py.File(OutPath+"out_depth_"+str(depth)+"-width_"+str(width)+"_3.h5", 'w') as o4:
        o1.create_dataset("output", data=outputs0.cpu().numpy())
        o2.create_dataset("output", data=outputs1.cpu().numpy())
        o3.create_dataset("output", data=outputs2.cpu().numpy())
        o4.create_dataset("output", data=outputs3.cpu().numpy())

    return (1-(float(correct) / total))*100.0

if __name__ == '__main__':
    space=[(2,8),(64, 192)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skopt import gp_minimize
    from skopt.plots import plot_convergence
    from skopt.plots import plot_evaluations
    from skopt.plots import plot_objective

    res_gp = gp_minimize(objective, space, n_calls=20)
    "Best score=%.4f" % res_gp.fun
    print("""Best parameters:
    - depth=%d
    - width=%d""" % (res_gp.x[0], res_gp.x[1])) 

    plot_convergence(res_gp)
    plt.savefig(OutPath+'plot_convergence.pdf', format='pdf')

    plt.set_cmap("viridis")
    _ = plot_evaluations(res_gp)
    plt.savefig(OutPath+'plot_evaluations.pdf', format='pdf')

    _ = plot_objective(res_gp)
    plt.savefig(OutPath+'plot_objective.pdf', format='pdf')
