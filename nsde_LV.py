import sys
import os
sys.path.append(os.path.dirname('__file__'))

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

from networks import *


class Net_LV(nn.Module):
    """
    Calibration of LV model: dS_t = S_t*r*dt + L(t,S_t,theta)dW_t to vanilla prices at different maturities
    """

    def __init__(self, dim, timegrid, strikes_call,  n_layers, vNetWidth, device, rate, maturities, n_maturities):
        
        super(Net_LV, self).__init__()
        self.dim = dim
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        self.maturities = maturities
        self.rate = rate
        
        
        # Leverage function
        self.diffusion = Net_timegrid(dim=dim+1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth, n_maturities=n_maturities, activation_output="softplus")
        
        # Control Variates
        self.control_variate_vanilla = Net_timegrid(dim=dim+1, nOut=len(strikes_call)*n_maturities, n_layers=3, vNetWidth=30, n_maturities=n_maturities)
        self.control_variate_exotics = Net_timegrid(dim=dim*len(self.timegrid)+1, nOut=1, n_layers = 3, vNetWidth = 20, n_maturities=n_maturities)
        

    def forward(self, S0, z, MC_samples, ind_T, period_length=30): 
        """this is to be used for evaluation so that everything fits into memory

        """
        #S_old = torch.repeat_interleave(S0, MC_samples, dim=0)
        ones = torch.ones(MC_samples, 1, device=self.device)
        path = torch.zeros(MC_samples, len(self.timegrid), device=self.device)
        S_old = ones * S0
        path[:,0] = S_old.squeeze(1)
        
        cv_vanilla = torch.zeros(S_old.shape[0], len(self.strikes_call)*len(self.maturities), device=self.device)
        price_vanilla_cv = torch.zeros(len(self.maturities), len(self.strikes_call), device=self.device)
        var_price_vanilla_cv = torch.zeros_like(price_vanilla_cv)

        cv_exotics = torch.zeros(S_old.shape[0], 1, device=self.device)

        exotic_option_price = torch.zeros_like(S_old)

        # we keep a running_max for Lookback option
        running_max = S_old
        
        # Solve for S_t (Euler)   
        for i in range(1, ind_T+1):
            idx = (i-1)//period_length # assume maturities are evenly distributed
            t = torch.ones_like(S_old) * self.timegrid[i-1]
            h = self.timegrid[i]-self.timegrid[i-1]    
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(MC_samples,1)

            current_time = ones*self.timegrid[i-1]
            diffusion = self.diffusion.forward_idx(idx, torch.cat([t,S_old],1))
            S_new = S_old + self.rate*S_old*h/(1+self.rate*S_old.detach()*torch.sqrt(h)) + S_old*diffusion* dW/(1+S_old.detach()*diffusion.detach()*torch.sqrt(h))
            
            cv_vanilla += torch.exp(-self.rate * self.timegrid[i-1]) * S_old.detach() * diffusion.detach() * self.control_variate_vanilla.forward_idx(idx,torch.cat([t,S_old.detach()],1)) * dW.repeat(1,len(self.strikes_call)*len(self.maturities))
            cv_exotics += torch.exp(-self.rate * self.timegrid[i-1]) * S_old.detach() * diffusion.detach() * self.control_variate_exotics.forward_idx(idx,torch.cat([t,path],1)) * dW 
            S_old = S_new
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



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.5)


def train_nsde(model, z_test, config):
    
    loss_fn = nn.MSELoss() 
    
    n_maturities = len(maturities)
    model = model.to(device)
    model.apply(init_weights)
    params_SDE = list(model.diffusion.parameters())#+list(model.driftV.parameters()) + list(model.diffusionV.parameters()) + [model.rho, model.v0]
    
    n_epochs = config["n_epochs"]
    T = config["maturities"][-1]
    
    # we take the target data that we are interested in.
    target_mat_T = torch.tensor(config["target_data"][:len(config["maturities"]),:len(config["strikes_call"])], device=device).float()
    
    optimizer_SDE = torch.optim.Adam(params_SDE,lr=0.001)
    optimizer_CV = torch.optim.Adam(list(model.control_variate_vanilla.parameters()) + list(model.control_variate_exotics.parameters()),lr=0.001)
    scheduler_SDE = torch.optim.lr_scheduler.MultiStepLR(optimizer_SDE, milestones=[500,800], gamma=0.2)
    
    loss_val_best = 10
    itercount=0
    
    for epoch in range(n_epochs):
        
        # We alternate Neural SDE optimisation and Hedging strategy optimisation
        requires_grad_CV = (epoch+1) % 2 == 0
        requires_grad_SDE = not requires_grad_CV
        if requires_grad_CV:
            model.control_variate_vanilla.unfreeze() 
            model.control_variate_exotics.unfreeze() 
            model.diffusion.freeze()
        else:
            model.diffusion.unfreeze()
            model.control_variate_vanilla.freeze()
            model.control_variate_exotics.freeze()
        
        
        print('epoch:', epoch)
        
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
                print('iteration {}, sum_variance={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount, loss.item(), time_forward, time_backward))
                nn.utils.clip_grad_norm_(list(model.control_variate_vanilla.parameters()) + list(model.control_variate_exotics.parameters()), 3)
                optimizer_CV.step()
            else:
                MSE = loss_fn(pred, target_mat_T)
                loss = MSE
                init_time = time.time()
                loss.backward()
                nn.utils.clip_grad_norm_(params_SDE, 5)
                time_backward = time.time() - init_time
                print('iteration {}, loss={:4.2e}, exotic price={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount, loss.item(), exotic_option_price, time_forward, time_backward))
                optimizer_SDE.step()
        
        scheduler_SDE.step() 
        
        #evaluate and print RMSE validation error at the start of each epoch
        with torch.no_grad():
            pred, _, exotic_option_price, exotic_price_mean, exotic_price_var, error = model(S0, z_test, z_test.shape[0], T, period_length=16)
            print("pred:",pred)
            print("target", target_mat_T)
        
        # Exotic option price hedging strategy error
        error_hedge = error
        error_hedge_2 = torch.mean(error_hedge**2)
        error_hedge_inf = torch.max(torch.abs(error_hedge))
        with open("error_hedge.txt","a") as f:
            f.write("{},{:.4f},{:.4f},{:.4f}\n".format(epoch,error_hedge_2, error_hedge_inf,exotic_price_var.item()))
        if (epoch+1)%100 == 0:
            torch.save(error_hedge, "error_hedge.pth.tar")
        
        # Evaluation Error of calibration to vanilla option prices
        MSE = loss_fn(pred, target_mat_T)
        loss_val=torch.sqrt(MSE)
        print('epoch={}, loss={:.4f}'.format(epoch, loss_val.item()))
        with open("log_train.txt","a") as f:
            f.write('epoch={}, loss={:.4f}\n'.format(epoch, loss_val.item()))
        
        # save checkpooint
        if loss_val < loss_val_best:
             model_best = model
             loss_val_best=loss_val
             print('loss_val_best', loss_val_best)
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
    maturities = range(16, 65, 16)
    n_maturities = len(maturities)
    
    # Neural SDE
    S0 = 1
    rate = 0.025 # risk-free rate
    model = Net_LV(dim=1, timegrid=timegrid, strikes_call=strikes_call, n_layers=args.n_layers, vNetWidth=args.vNetWidth, device=device, n_maturities=n_maturities, maturities=maturities, rate=rate)
    model.apply(init_weights)
    
    # Monte Carlo test data
    MC_samples_test=200000
    z_test = torch.randn(MC_samples_test, n_steps, device=device)
    z_test = torch.cat([z_test, -z_test], 0) # We will use antithetic Brownian paths for testing
    
    # Logging file
    with open("error_hedge.txt","w") as f:
        f.write("epoch,error_hedge_2,error_hedge_inf\n")

    CONFIG = {"batch_size":40000,
            "n_epochs":1000,
            "maturities":maturities,
            "n_maturities":n_maturities,
            "strikes_call":strikes_call,
            "timegrid":timegrid,
            "n_steps":n_steps,
            "target_data":data}
    
    model = train_nsde(model, z_test, CONFIG)

