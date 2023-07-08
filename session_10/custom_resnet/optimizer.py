from torch_lr_finder import LRFinder
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR



class Optimization:
    def __init__(self, model, device, train_loader, criterion, num_epochs=10):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.optimizer = None
        self.scheduler = None
        self.define_optimizer()
    
    def __call__(self):
        self.define_optimizer()
        self.define_scheduler()
        self.define_one_cycle_lr_scheduler()

    def define_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.03, weight_decay=1e-4)


    def define_scheduler(self, max_lr):
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr = max_lr,
            steps_per_epoch=len(self.train_loader),
            epochs = self.num_epochs,
            pct_start = 5/self.num_epochs,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )


def find_best_lr(model, train_loader, optimizer, criterion, device):
    lr_finder = LRFinder(model, optimizer, criterion, device) 
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode='exp')
    lr_finder.plot()
    lr_finder.reset()
    return lr_finder.history
