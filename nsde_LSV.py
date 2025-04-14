import sys
import os
sys.path.append(os.path.dirname('__file__'))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import time
from random import randrange
import copy
import argparse
import random

from IPython.display import Image
from networks import *
from geomloss import SamplesLoss


class Net_LSV(nn.Module):
    """
    Calibration of LV model: dS_t = S_t*r*dt + L(t,S_t,theta)dW_t to vanilla prices at different maturities
    """

    def __init__(self, dim, timegrid, strikes_call,  n_layers, vNetWidth, device, rate, maturities, n_maturities, activation="relu"):
        
        super(Net_LSV, self).__init__()
        self.dim = dim
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        self.maturities = maturities
        self.rate = rate
        self.activation = activation
         
        
        
        # Neural SDE for LSV model
        self.diffusion = Net_timegrid(dim=dim+2, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, n_maturities=n_maturities, activation = activation, activation_output="softplus")
        self.v0 = torch.nn.Parameter(torch.rand(1)-3)
        self.driftV = Net_timegrid(dim=dim, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, n_maturities=n_maturities)
        self.diffusionV = Net_timegrid(dim=dim, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, n_maturities=n_maturities, activation = activation, activation_output="softplus")
        self.rho = torch.nn.Parameter(2*torch.rand(1)-1)
        
        # Control Variates
        self.control_variate_vanilla = Net_timegrid(dim=dim+1, nOut=len(strikes_call)*n_maturities, n_layers=3, vNetWidth=30, n_maturities=n_maturities, activation= activation)
        self.control_variate_exotics = Net_timegrid(dim=dim*len(self.timegrid)+1+1, nOut=1, n_layers = 3, vNetWidth = 20, n_maturities=n_maturities, activation= activation)
        

    def forward(self, S0, z, MC_samples, ind_T, period_length=30): 
        """this is to be used for evaluation so that everything fits into memory

        """
        #S_old = torch.repeat_interleave(S0, MC_samples, dim=0)
        ones = torch.ones(MC_samples, 1, device=self.device)
        path = torch.zeros(MC_samples, len(self.timegrid), device=self.device)
        S_old = ones * S0
        path[:,0] = S_old.squeeze(1)
        V_old = ones * torch.sigmoid(self.v0)*0.5
        rho = torch.tanh(self.rho)
        
        cv_vanilla = torch.zeros(S_old.shape[0], len(self.strikes_call)*len(self.maturities), device=self.device)
        price_vanilla_cv = torch.zeros(len(self.maturities), len(self.strikes_call), device=self.device)
        var_price_vanilla_cv = torch.zeros_like(price_vanilla_cv)

        cv_exotics = torch.zeros(S_old.shape[0], 1, device=self.device)

        exotic_option_price = torch.zeros_like(S_old)
        running_max = S_old
        
        # Solve for S_t (Euler)   
        for i in range(1, ind_T+1):
            idx = (i-1)//period_length # assume maturities are evenly distributed
            t = torch.ones_like(S_old) * self.timegrid[i-1]
            h = self.timegrid[i]-self.timegrid[i-1]    
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(MC_samples,1)
            zz = torch.randn_like(dW)
            dB = rho * dW + torch.sqrt(1-rho**2)*torch.sqrt(h)*zz

            current_time = ones*self.timegrid[i-1]
            diffusion = self.diffusion.forward_idx(idx, torch.cat([t,S_old, V_old],1))
            S_new = S_old + self.rate*S_old*h/(1+self.rate*S_old.detach()*torch.sqrt(h)) + S_old*diffusion* dW/(1+S_old.detach()*diffusion.detach()*torch.sqrt(h))
            V_new = V_old + self.driftV.forward_idx(idx,V_old)*h + self.diffusionV.forward_idx(idx, V_old)*dB
            
            cv_vanilla += torch.exp(-self.rate * self.timegrid[i-1]) * S_old.detach() * diffusion.detach() * self.control_variate_vanilla.forward_idx(idx,torch.cat([t,S_old.detach()],1)) * dW.repeat(1,len(self.strikes_call)*len(self.maturities))
            cv_exotics += torch.exp(-self.rate * self.timegrid[i-1]) * S_old.detach() * diffusion.detach() * self.control_variate_exotics.forward_idx(idx,torch.cat([t,path, V_old.detach()],1)) * dW 
            
            S_old = S_new
            V_old = torch.clamp(V_new,0)
            path[:,i] = S_old.detach().squeeze(1)
            
            running_max = torch.max(running_max, S_old)

            if i in self.maturities:
                ind_maturity = self.maturities.index(i)
                for idx, strike in enumerate(self.strikes_call):
                    cv = cv_vanilla.view(-1,len(self.maturities), len(self.strikes_call))
                    price_vanilla = torch.exp(-self.rate*self.timegrid[i])*torch.clamp(S_old-strike,0).squeeze(1)-cv[:,ind_maturity,idx]
                    price_vanilla_cv[ind_maturity,idx] = price_vanilla.mean()#torch.exp(-rate/n_steps)*price.mean()
                    var_price_vanilla_cv[ind_maturity,idx] = price_vanilla.var()

        exotic_option_price = running_max - S_old
        error = torch.exp(-self.rate*self.timegrid[ind_T])*exotic_option_price.detach() - torch.mean(torch.exp(-self.rate*self.timegrid[ind_T])*exotic_option_price.detach()) - cv_exotics.detach()
        exotic_option_price = torch.exp(-self.rate*self.timegrid[ind_T])*exotic_option_price  - cv_exotics
        
        return price_vanilla_cv, var_price_vanilla_cv, exotic_option_price, exotic_option_price.mean(), exotic_option_price.var(), error

#censé etre mieux pour relu et silu 
def init_weights_lu(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')  
        if m.bias is not None:
            m.bias.data.zero_()
#censé etre mieux pour tanh
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.5)



def train_nsde(model, z_test, config,loss_calibration='MSE',init='xavier'): #METTRE loss_calibration='Wass' pour loss de Wasserstein (juste la calibration pas le hedging) et init='kaiming' pour init a kiaming, sinon parametre par defaut sont comme avant les changements 
        
    
    if loss_calibration=='Wass':
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05)
    else:
        loss_fn = nn.MSELoss() 
    
    n_maturities = len(maturities)
    model = model.to(device)
    if init=='kaiming':
        model.apply(init_weights_lu)
    else:
        model.apply(init_weights)
    params_SDE = list(model.diffusion.parameters())+list(model.driftV.parameters()) + list(model.diffusionV.parameters()) + [model.rho, model.v0]
    

    n_epochs = config["n_epochs"]
    T = config["maturities"][-1]
    # we take the target data that we are interested in
    target_mat_T = torch.tensor(config["target_data"][:len(config["maturities"]),:len(config["strikes_call"])], device=device).float()
    
    optimizer_SDE = torch.optim.Adam(params_SDE,lr=0.001)
    optimizer_CV = torch.optim.Adam(list(model.control_variate_vanilla.parameters()) + list(model.control_variate_exotics.parameters()),lr=0.001)
    scheduler_SDE = torch.optim.lr_scheduler.MultiStepLR(optimizer_SDE, milestones=[500,800], gamma=0.2)
    
    loss_val_best = 10
    itercount=0
    
    errors_hedge_2=torch.zeros(n_epochs)
    errors_hedge_inf=torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        
        # We alternate Neural SDE optimisation and Hedging strategy optimisation
        requires_grad_CV = (epoch+1) % 2 == 0
        requires_grad_SDE = not requires_grad_CV
        if requires_grad_CV:
            model.control_variate_vanilla.unfreeze() 
            model.control_variate_exotics.unfreeze() 
            model.diffusion.freeze()
            model.driftV.freeze()
            model.diffusionV.freeze()
            model.v0.requires_grad_(False)
            model.rho.requires_grad_(False)
        else:
            model.diffusion.unfreeze()
            model.driftV.unfreeze()
            model.diffusionV.unfreeze()
            model.v0.requires_grad_(True)
            model.rho.requires_grad_(True)
            model.control_variate_vanilla.freeze()
            model.control_variate_exotics.freeze()
        
        
        
        batch_size = config["batch_size"]
        
        # we go through an epoch, i.e. 20*batch size paths
        for i in range(0,20*batch_size, batch_size):
            batch_z = torch.randn(batch_size, config["n_steps"], device=device) # just me being paranoid to be sure that we have independent samples in the batch. Sampling from an antithetic dataset does not make sense to me 

            optimizer_SDE.zero_grad()
            optimizer_CV.zero_grad()
            
            init_time = time.time()
            pred, var, _, exotic_option_price, exotic_option_var, _ = model(S0, batch_z, batch_size,T, period_length=16)
            time_forward = time.time() - init_time

            itercount += 1
            if requires_grad_CV:
                loss = var.sum() + exotic_option_var
                init_time = time.time()
                loss.backward()
                time_backward = time.time() - init_time
                nn.utils.clip_grad_norm_(list(model.control_variate_vanilla.parameters()) + list(model.control_variate_exotics.parameters()), 3)
                optimizer_CV.step()
            else:
                Wass = loss_fn(pred, target_mat_T)
                loss = Wass
                init_time = time.time()
                loss.backward()
                nn.utils.clip_grad_norm_(params_SDE, 5)
                time_backward = time.time() - init_time
                optimizer_SDE.step()
        
        scheduler_SDE.step() 
        
        #evaluate and print RMSE validation error at the start of each epoch
        with torch.no_grad():
            pred, _, exotic_option_price, exotic_price_mean, exotic_price_var, error = model(S0, z_test, z_test.shape[0], T, period_length=16)
        
        
        # Exotic option price hedging strategy error
        error_hedge = error
        error_hedge_2 = torch.mean(error_hedge**2)
        errors_hedge_2[epoch] = error_hedge_2
        error_hedge_inf = torch.max(torch.abs(error_hedge))
        errors_hedge_inf[epoch] = error_hedge_inf
        with open("error_hedge.txt","a") as f:
            f.write("{},{:.4f},{:.4f},{:.4f}\n".format(epoch,error_hedge_2, error_hedge_inf,exotic_price_var.item()))
        if (epoch+1)%100 == 0:
            torch.save(error_hedge, "error_hedge.pth.tar")
        
        # Evaluation Error of calibration to vanilla option prices
        if loss_calibration=='Wass':
            loss_val= loss_fn(pred, target_mat_T)
        else:
            loss_val= torch.sqrt(loss_fn(pred,target_mat_T))

        with open("log_eval.txt","a") as f:
            f.write('{},{:.4e}\n'.format(epoch, loss_val.item()))
        
        # save checkpooint
        if loss_val < loss_val_best:
             model_best = model
             loss_val_best=loss_val
             type_bound = "no"#"lower" if args.lower_bound else "upper"
             filename = "Neural_SDE_exp{}_{}bound_maturity{}_AugmentedLagrangian.pth.tar".format(args.experiment,type_bound,T)
             checkpoint = {"state_dict":model.state_dict(),
                     "exotic_price_mean": exotic_price_mean,
                     "exotic_price_var":exotic_price_var,
                     "T":T,
                     "pred":pred,
                     "target_mat_T": target_mat_T}

             torch.save(checkpoint, filename)

        if loss_val.item() < 2e-5:
            break

    n=range(1,n_epochs+1)
    print('\n Loss for calibration of NSDE coefficient:', loss_calibration, '\n Activation fonction: ' ,model.activation)
    plt.figure(figsize=(3, 5))  
    plt.plot(n, errors_hedge_2.cpu().numpy(), label="Error Hedge 2")
    plt.plot(n, errors_hedge_inf.cpu().numpy(), label="Error Hedge Inf")
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot2.png", dpi=300)
    plt.show()

    return model_best


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--vNetWidth', type=int, default=50)
    parser.add_argument('--experiment', type=int, default=0)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device='cuda:{}'.format(args.device)
        torch.cuda.set_device(args.device)
    else:
        device="cpu"

    # Load market prices and set training target
    data = torch.load("Call_prices_59.pt")

    # Set up training - Strike values, time discretisation and maturities
    strikes_call = np.arange(0.8,1.21, 0.02)
    print(strikes_call)
    n_steps=96
    timegrid = torch.linspace(0,1,n_steps+1).to(device) 
    maturities = range(16, 33, 16)
    n_maturities = len(maturities)
    
    # Neural SDE
    S0 = 1
    rate = 0.025 # risk-free rate
    activation = 'silu' #DEFINIR: relu, tanh ou silu
    model = Net_LSV(dim=1, timegrid=timegrid, strikes_call=strikes_call, n_layers=args.n_layers, vNetWidth=args.vNetWidth, device=device, n_maturities=n_maturities, maturities=maturities, rate=rate, activation=activation)
    model.to(device)
    model.apply(init_weights)
    
    # Monte Carlo test data
    MC_samples_test=200000
    z_test = torch.randn(MC_samples_test, n_steps, device=device)
    z_test = torch.cat([z_test, -z_test], 0) # We will use antithetic Brownian paths for testing
    
    # Logging file
    with open("error_hedge.txt","w") as f:
        f.write("epoch,error_hedge_2,error_hedge_inf\n")

    CONFIG = {"batch_size":40000,
            "n_epochs":20,
            "maturities":maturities,
            "n_maturities":n_maturities,
            "strikes_call":strikes_call,
            "timegrid":timegrid,
            "n_steps":n_steps,
            "target_data":data}
    

    model = train_nsde(model, z_test, CONFIG)
    model = train_nsde(model, z_test, CONFIG,loss_calibration='Wass') #loss_calibration='Wass' sinon par defaut MSE #init='kaiming' sinon par defaut xavier


