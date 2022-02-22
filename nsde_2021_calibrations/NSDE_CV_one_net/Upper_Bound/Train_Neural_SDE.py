import torch
torch.manual_seed(901) # fixed for reproducibility
import torch.nn as nn
import numpy as np
import math
import os
import time
from random import randrange
import copy
import argparse
from random import randint
from py_vollib.black_scholes.implied_volatility import implied_volatility 
from py_vollib.black_scholes import black_scholes as prc 



class Net_timestep(nn.Module):
    def __init__(self, dim, nOut, n_layers, vNetWidth, activation = "relu", activation_output="id", batchnorm=False):
        super(Net_timestep, self).__init__()
        self.dim = dim
        self.nOut = nOut
        self.batchnorm=batchnorm
        
        if activation=="relu":
            self.activation = nn.ReLU()
        elif activation=="tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("unknown activation function {}".format(activation))

        if activation_output == "id":
            self.activation_output = nn.Identity()
        elif activation_output == "softplus":
            self.activation_output = nn.Softplus()
        else:
            raise ValueError("unknown output activation function {}".format(activation_output))
        
        self.i_h = self.hiddenLayerT1(dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for l in range(n_layers-1)])
        self.h_o = self.outputLayer(vNetWidth, nOut)
        
    
    def hiddenLayerT1(self, nIn, nOut):
        if self.batchnorm:
            layer = nn.Sequential(nn.Linear(nIn,nOut,bias=True),
                                 # nn.BatchNorm1d(nOut, momentum=0.1),  
                                  self.activation)   
        else:
            layer = nn.Sequential(nn.Linear(nIn,nOut,bias=True),
                                  self.activation)   
        return layer
    
    # no constant bias in the output layer (bias=False)
    def outputLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut,bias=False), self.activation_output)
        return layer
    
    def forward(self, S):
        h = self.i_h(S)
        for l in range(len(self.h_h)):
            h = self.h_h[l](h)
        output = self.h_o(h)
        return output
    

class Net_timegrid(nn.Module):
    def __init__(self, dim, nOut, n_layers, vNetWidth, n_maturities, activation="relu", activation_output="id"):
        super().__init__()
        self.dim = dim
        self.nOut = nOut
        self.net_t = nn.ModuleList([Net_timestep(dim, nOut, n_layers, vNetWidth, activation=activation, activation_output=activation_output) for idx in range(n_maturities)])
        
    def forward_idx(self, idnet, x):
        y = self.net_t[idnet](x)
        return y

    def freeze(self):
        for p in self.net_t.parameters():
            p.requires_grad=False

    def unfreeze(self, *args):
        if not args:
            for p in self.net_t.parameters():
                p.requires_grad=True
        else:
            # unfreeze the parameters between [last_T,T]
            self.freeze()
            for idx in args:
                for p in self.net_t[idx].parameters():
                    p.requires_grad=True
           
class Net_LV(nn.Module):
    #Calibration of neural SDE model
    def __init__(self, timegrid, strikes_call, device, n_maturities):
        
        super(Net_LV, self).__init__()
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        # initialise neural network for diffusion and control variate outputs 
        self.master =  Net_timegrid(dim=3, nOut=1+1*len(strikes_call), n_layers=3, vNetWidth=100, n_maturities=n_maturities, activation_output="softplus")

        # stochastic vol parameters
        self.v0 = torch.nn.Parameter(torch.rand(1)*0.1)
        self.V = Net_timegrid(dim=1, nOut=2, n_layers=2, vNetWidth=20, n_maturities=n_maturities)
        self.rho = torch.nn.Parameter(2*torch.rand(1)-1)
        self.exotic_hedge = Net_timegrid(dim=2, nOut=1, n_layers=3, vNetWidth=200, n_maturities=n_maturities)


    def forward(self, S0, rate, z_1, MC_samples, ind_T, period_length,cal_exotic_var): 
        # initialisation
        ones = torch.ones(MC_samples, 1, device=self.device)
        zeros = torch.zeros(MC_samples,1, device=self.device)
        S1_old = ones * S0
        cv_van_1 = torch.zeros(S1_old.shape[0], len(self.strikes_call), device=self.device)
        cv_exotic = torch.zeros(S1_old.shape[0], 1 , device=self.device)
        pv_1cv = torch.zeros(len(self.strikes_call), device=self.device)
        var_pv_1cv = torch.zeros_like(pv_1cv)
        var_pv_1_cv_detach = torch.zeros_like(pv_1cv)
        var_prc_vanilla_1cv_out_detach = torch.Tensor().to(device)
        exotic_prc_no_cv = torch.zeros(1, device=self.device)
        exotic_prc_cv = torch.zeros(1, device=self.device)
        exotic_prc_var_cv = torch.zeros(1, device=self.device)
        exotic_prc_var_no_cv = torch.zeros(1, device=self.device)
        prc_vanilla_1cv_out = torch.Tensor().to(device)
        var_prc_vanilla_1cv_out = torch.Tensor().to(device)
        martingale_test = torch.zeros(1, device=self.device)
        put_atm = torch.zeros(1, device=self.device)
        call_atm = torch.zeros(1, device=self.device)
        V_old = ones * torch.sigmoid(self.v0)*0.5
        rho = torch.tanh(self.rho)
        running_max = S1_old # initialisation of running_max

        str_rho = 'rho.txt'
        str_V0 = 'V0.txt'
        save_file(torch.tanh(self.rho).detach().cpu(), str_rho)
        save_file((torch.sigmoid(self.v0)*0.5).detach().cpu(), str_V0)


        for i in range(1, ind_T+1):
            idx = 0 # all maturities calibrated simultaneously 
            idx_net = (i-1)//period_length # assume maturities are evenly distributed, i.e. 0, 16, 32, ..., 96
            t = torch.ones_like(S1_old) * self.timegrid[i-1]
            h = self.timegrid[i]-self.timegrid[i-1]   
            dW_1 = (torch.sqrt(h) * z_1[:,i-1]).reshape(MC_samples,1)
            zz = torch.randn_like(dW_1)
            dB = rho * dW_1 + torch.sqrt(1-rho**2)*torch.sqrt(h)*zz
            current_time = ones*self.timegrid[i-1]
            S1exp_old=torch.log(S1_old)
            if cal_exotic_var==0:
                diffusion = self.master.forward_idx(idx_net, torch.cat([t,S1exp_old,V_old],1))
                volatility = self.V.forward_idx(idx_net,V_old)
            if cal_exotic_var==1:
               with torch.no_grad():
                  diffusion = self.master.forward_idx(idx_net, torch.cat([t,S1exp_old,V_old],1))
                  volatility = self.V.forward_idx(idx_net,V_old)
               # exotic_hedge = self.exotic_hedge.forward_idx(idx_net, torch.cat([t,S1exp_old.detach(), V_old.detach()],1)) # if exotic hedge depends on inst. vol.
               exotic_hedge = self.exotic_hedge.forward_idx(idx_net, torch.cat([t,S1exp_old.detach()],1))
            V_new = V_old + volatility[:,0].reshape(MC_samples,1)*h + torch.nn.functional.softplus(volatility[:,1]).reshape(MC_samples,1)*dB
            diffusion_const = diffusion.detach()
            # Drift normalisations are constants in Tamed Euler hence gradient detached
            drift_1 =  (rate-0.5*(diffusion[:,0]**2).reshape(MC_samples,1))
            drift_1c = (1+torch.abs(drift_1.detach())*torch.sqrt(h))

            # Those are used in grad calculation 
            diff_1 = diffusion[:,0].reshape(MC_samples,1) * dW_1
            # Normalisation constants in diffusion Tamed Euler scheme
            diff_1_c = (1+torch.abs(diffusion_const[:,0].reshape(MC_samples,1))*torch.sqrt(h))
            
            # Tamed Euler step
            S1exp_new = S1exp_old + drift_1*h/drift_1c + diff_1/diff_1_c 
            S1_new=torch.exp(S1exp_new)
            # Those are used for control variate training       
            dS_1 = torch.exp(-rate*self.timegrid[i])*S1_new - torch.exp(-rate*self.timegrid[i-1])*S1_old
            # Forward pass through CV neural network
            cv_fwd = diffusion  
            discount_T = ((i-1)//16 + 1)*16
            dis_fact = torch.exp(-rate*self.timegrid[discount_T])
            dis_CV = torch.exp(-rate*self.timegrid[i])
            # Repeat tensors of asset prcice changes to be applied for all options 
            repeat_1 = dS_1.detach().repeat(1,len(self.strikes_call)) 
            
            # Evaluate stoch. integrals corresponding to control variates using discounted asset prices as martingales
            cv_van_1 += cv_fwd[:,1:1+len(self.strikes_call)] * repeat_1
            if cal_exotic_var==1:
               cv_exotic +=  exotic_hedge.reshape(MC_samples,1) * dS_1.detach()
 
                        
            # Update values of asset prc processes for next Tamed Euler step
            S1_old = S1_new

            # Evaluate exotic payoff 
            if cal_exotic_var==0: 
                 running_max = torch.max(running_max, S1_new)
            if cal_exotic_var==1: 
                 running_max = torch.max(running_max, S1_new).detach()
            V_old = torch.clamp(V_new,0)
        
           # Evaluate vanillas for maturity corresponding to vanillas     
            if int(i) in maturities: 

        # Apply control variate and calculate variance of estimates (used to train CV neural network)         
               for idx, strike in enumerate(self.strikes_call):
                  pv_1_cv = dis_fact*torch.clamp(S1_old-strike,0).squeeze(1).detach()-cv_van_1[:,idx]
                  pv_1_cv_detach = dis_fact*torch.clamp(S1_old-strike,0).squeeze(1)-cv_van_1[:,idx].detach()
                  pv_1cv[idx] = pv_1_cv_detach.mean() # gradient of cv not applied when evaluating price with control variate
                  var_pv_1cv[idx] = pv_1_cv.var() # gradient of cv applied
                  var_pv_1_cv_detach[idx] = pv_1_cv_detach.var() # gradient of cv not applied in variance evaluation 
               prc_vanilla_1cv_out = torch.cat([prc_vanilla_1cv_out,pv_1cv.T],0) 
               var_prc_vanilla_1cv_out = torch.cat([var_prc_vanilla_1cv_out,var_pv_1cv.T],0)
               var_prc_vanilla_1cv_out_detach = torch.cat([var_prc_vanilla_1cv_out_detach,var_pv_1_cv_detach.T],0)
            if int(i)==96:
                  martingale_test = dis_fact*S1_old.detach()
                  martingale_test = martingale_test.mean()
                  put_atm = dis_fact*torch.clamp(S0-S1_old.detach(),0).squeeze(1)
                  put_atm = put_atm.mean().detach()
                  call_atm = dis_fact*torch.clamp(S1_old.detach()-S0,0).squeeze(1)
                  call_atm = call_atm.mean()
                  put_call_parity_error = call_atm-put_atm - S0 + dis_fact*S0
                  payoff = running_max - S1_old
                  payoff_cv = payoff.detach()
                  exotic_prc_cv = dis_fact*payoff_cv.squeeze(1).reshape(MC_samples,1)-cv_exotic.reshape(MC_samples,1)
                  exotic_prc_no_cv = dis_fact*payoff.squeeze(1)
                  exotic_prc_var_cv = exotic_prc_cv.var()
                  exotic_prc_var_no_cv = exotic_prc_no_cv.var()
                  exotic_prc_no_cv = exotic_prc_no_cv.mean() 
                  exotic_prc_cv = exotic_prc_cv.mean()      
        
        return prc_vanilla_1cv_out, var_prc_vanilla_1cv_out,var_prc_vanilla_1cv_out_detach, exotic_prc_no_cv, exotic_prc_cv, exotic_prc_var_no_cv, exotic_prc_var_cv, martingale_test, put_atm, call_atm, put_call_parity_error      
        
def save_file(loss, filename):
    with open(filename, "a") as f:
        f.write("{}\n".format(loss.item()))
    return 0

def train_models(args):
    
    loss_fn = nn.MSELoss() 
    maturities = range(16, 97, 16)
    n_maturities = len(maturities)
    model = Net_LV(timegrid=timegrid, strikes_call=strikes_call, device=device, n_maturities=n_maturities)
    model = model.to(device)
    itercount = 0
    itercount_2 = 0
    itercount_3 = 0
    itercount_4 = 0
    n_epochs = 260
    batch_size = 50000
    batch_size_hedge = 50000
         
    for idx, T in enumerate(maturities):

       if idx==0:
          T=96 # simulate paths for the 6 month horizon
       else:
          model_best=model # model is trained to all maturities simultaneously
          break
   
       base_learing_rate = 0.0005 # set maximum learning_rate so that model trains (if too high learning rate is chosen model parameters get stuck in some local minimum)

       params_SDE = list(model.master.parameters()) + list(model.V.parameters()) + [model.rho, model.v0]
       params_hedge = list(model.exotic_hedge.parameters())

       optimizer_SDE = torch.optim.Adam(params_SDE,lr=base_learing_rate)
       optimizer_SDE_2 = torch.optim.Adam(params_SDE,lr=base_learing_rate)
       optimizer_SDE_4 = torch.optim.Adam(params_hedge,lr=100*base_learing_rate)

       scheduler_SDE = torch.optim.lr_scheduler.MultiStepLR(optimizer_SDE, milestones=[50], gamma=0.15)
       scheduler_SDE_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_SDE_2, milestones=[50], gamma=0.15)
       scheduler_SDE_4 = torch.optim.lr_scheduler.MultiStepLR(optimizer_SDE_4, milestones=[50], gamma=0.15)

       loss_val_best = 10
       best_IV_error = 10
       best_hedge_error = 10
       exotic_hedge_error = torch.tensor(10, device=device).float()
       target_mat_T = torch.tensor(data, device=device).float()
       
       MC_samples_gen=1000000 # this is for validation
       MC_samples_var=10000
       # unfreeze price and volatility networks for all maturities
       model.master.unfreeze()
       model.V.unfreeze()
        
       # freeze cv
       model.exotic_hedge.freeze()

       batch_steps= 96
       LAMBDA = args.LAMBDA
       LAMBDA_2 = args.LAMBDA
       c = args.c
       c_2 = args.c
       
       for epoch in range(n_epochs):
            # evaluate model at initialisation and save model statistics at initialisation
            if epoch==0:

                # calculate target IVs
               # van_target_1 = target_mat_T[6:21].cpu().numpy()
                K = strikes_call
                iv_target_1 = np.zeros_like(K)
                iv_target_1_out = torch.Tensor().to(device)
                with torch.no_grad():
                  try:
                   for idxt, t in enumerate(maturities):
                        van_target_1 = target_mat_T[idxt,::].cpu().numpy() 
                        for idxx, ( target, k) in enumerate(zip(van_target_1, K)):
                            iv_target_1[idxx] = implied_volatility(target,  S=S0, K=k, r=0.0, t=t/(2*96), flag="c") 
                        iv_target_1_out = torch.cat([iv_target_1_out,torch.from_numpy(iv_target_1).float().to(device).T],0)  
                  except:
                   pass
                print('iv_target_1',iv_target_1_out)
                
                # calculate vega weights 
                vega_target_1 = np.ones_like(K)
                vega_out = torch.Tensor().to(device)
                constant_shift=0.01
                
                with torch.no_grad():
                       
                    try:
                        for idxt, t in enumerate(maturities):
                            van_target_1 = target_mat_T[idxt,::].cpu().numpy() 
                            for idxx, (target, k) in enumerate(zip(van_target_1, K)):
                                iv_target_11 = implied_volatility(target, S=S0, K=k, r=0.0, t=t/(2*96), flag="c") 
                                iv_1 = iv_target_11+constant_shift
                                vega_target_1[idxx] = constant_shift/(prc(flag='c',S=S0,K=k,t=t/(2*96),r=0.0,sigma=iv_1)-target)
                            vega_out = torch.cat([vega_out,torch.from_numpy(vega_target_1).float().to(device).T],0)                    
                    except:
                        pass
                vega_target_1=torch.tensor(vega_target_1).to(device)
                print('vega_out', vega_out)
                vega=torch.reshape(vega_out,(6,21))
                #normalise the weights by the number of options 6x21=126
                vega=(21*6)/torch.sum(vega)*vega
                print('vegas of target options:',vega)
                
                with torch.no_grad():
                    pv_1cv, _,_,exotic_prc_no_cv,exotic_prc_cv,_, _,martingale_test, put_atm, call_atm, put_call_parity_error = model(S0, rate, z_1, MC_samples_gen, T, period_length=16,cal_exotic_var=1)
                    _, var_pv_1cv_val,_,_,_,exotic_prc_var_no_cv_val, exotic_prc_var_cv_val,_,_,_,_ = model(S0, rate, z_1_var,MC_samples_var, T, period_length=16,cal_exotic_var=1)
                 
                    print("pred:",pv_1cv)
                    print("target:", target_mat_T )
                pred=torch.reshape(pv_1cv,(6,21))
                loss_val = torch.sqrt(loss_fn(pred,target_mat_T))
                loss_val_rel  = torch.sqrt(loss_fn(torch.div(pred,target_mat_T), torch.ones_like(pred)))
                loss_vega = torch.sqrt(loss_fn(torch.mul(pred,vega),torch.mul(target_mat_T,vega)))
                print('validation {}, loss={:.10f}'.format(itercount, loss_val.item()))
                print('validation_rel {}, loss={:.10f}'.format(itercount, loss_val_rel.item()))
                print('validation_vega {}, loss={:.10f}'.format(itercount, loss_vega.item()))   
                strloss='loss_'+str(T)+'.txt'
                strloss_rel='loss_rel'+str(T)+'.txt'
                strloss_vega='loss_vega'+str(T)+'.txt'
                strloss_martingale='martingale_test_'+str(T)+'.txt'
                strloss_call_atm='call_atm_'+str(T)+'.txt'
                strloss_put_atm='put_atm_'+str(T)+'.txt'
                strloss_put_call_parity_error = 'put_call_parity_error'+str(T)+'.txt'
                strvan_1_var = 'variance_van_1_'+str(T)+'.txt'
                strexotic_var_cv = 'variance_exotic_cv'+str(T)+'.txt'
                strexotic_var_no_cv = 'variance_exotic_no_cv'+str(T)+'.txt'
                strexotic_prc_cv = 'prc_exotic_cv'+str(T)+'.txt'
                strexotic_prc_no_cv = 'prc_exotic_no_cv'+str(T)+'.txt'
                print('Variance_V1:', torch.sum(var_pv_1cv_val))
                print('Variance of exotic option with cv:', exotic_prc_var_cv_val)
                print('Variance of exotic option without cv:', exotic_prc_var_no_cv_val)
                print('Price of exotic option with cv:', exotic_prc_cv)
                print('Price of exotic option with cv:', exotic_prc_no_cv)
                save_file(loss_val.cpu(), strloss)
                save_file(loss_val_rel.cpu(), strloss_rel)
                save_file(loss_vega.cpu(), strloss_vega)
                save_file(torch.sum(var_pv_1cv_val).cpu(), strvan_1_var)
                save_file(martingale_test.cpu(), strloss_martingale) 
                save_file(call_atm.cpu(), strloss_call_atm)
                save_file(put_atm.cpu(), strloss_put_atm) 
                save_file(put_call_parity_error.cpu(), strloss_put_call_parity_error) 
                save_file(exotic_prc_var_cv_val.cpu(), strexotic_var_cv)
                save_file(exotic_prc_cv.cpu(), strexotic_prc_cv)
                save_file(exotic_prc_var_no_cv_val.cpu(), strexotic_var_no_cv)
                save_file(exotic_prc_no_cv.cpu(), strexotic_prc_no_cv)
                print('Martingale test at initialisation:', martingale_test)
                print('Call price at initialisation:', call_atm)
                print('Put price at initialisation:', put_atm)
                print('Put call parity error at initialisation:', put_call_parity_error)
                        
            print('Epoch, batch size:', epoch, batch_size)
            for i in range(0,20*batch_size, batch_size):
                # simulate paths over entire training horizon 
                if epoch<250:
                   batch_x1 = torch.randn(batch_size, batch_steps, device=device)
                else:
                   batch_x1 = torch.randn(batch_size_hedge, batch_steps, device=device)
                timestart=time.time()
                optimizer_SDE.zero_grad()
                optimizer_SDE_2.zero_grad()
                optimizer_SDE_4.zero_grad()

                 
                init_time = time.time()
                if epoch<250 :
                   pv_1cv, var_pv_1cv,var_pv_1cv_detach, exotic_prc_no_cv,exotic_prc_cv,_, _, martingale_test, put_atm, call_atm, put_call_parity_error = model(S0, rate, batch_x1, batch_size,T, period_length=16,cal_exotic_var=0)
                if epoch>=250:
                   model.master.freeze()
                   model.exotic_hedge.unfreeze()
                   _, _,_, _,exotic_prc, exotic_prc_var_no_cv,exotic_prc_var_cv,_, _, _, _ = model(S0, rate, batch_x1, batch_size_hedge,T, period_length=16,cal_exotic_var=1)
                time_forward = time.time() - init_time
                
                pred = torch.reshape(pv_1cv,(6,21))
                sum_variances = torch.sum(var_pv_1cv)
                sum_variances_detach = torch.sum(var_pv_1cv_detach)
                if epoch%2==0 and epoch<250: 
                    MSE = loss_fn(torch.mul(pred,vega),torch.mul(target_mat_T,vega))
                    loss= sum_variances + LAMBDA * MSE + c/2 * MSE**2
                    init_time = time.time()
                    itercount += 1
                    loss.backward()
                    if itercount%20==0:
                       LAMBDA += c*MSE.detach() if LAMBDA<1e6 else LAMBDA
                       c = 2*c if c<1e10 else c
                    nn.utils.clip_grad_norm_(params_SDE, 1)
                    time_backward = time.time() - init_time
                    if (itercount % 20 == 0):
                        print('training statistics')
                        print('iteration {}, sum_variance_van={:.4f}, variance_year={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount, sum_variances,var_pv_1cv[0].item(), time_forward, time_backward))
                    optimizer_SDE.step()
                    scheduler_SDE.step()

                if epoch%2==1 and epoch<250:
                    MSE = loss_fn(torch.mul(pred,vega),torch.mul(target_mat_T,vega))
                    loss= -0.5*exotic_prc_no_cv + LAMBDA_2 * MSE + c_2/2 * MSE**2  # we use exotic price without control variate in optimising payoff bound 
                    init_time = time.time()
                    itercount_2 +=1
                    loss.backward()
                    if itercount_2%20==0:
                       LAMBDA_2 += c_2*MSE.detach() if LAMBDA_2<1e6 else LAMBDA_2
                       c_2 = 2*c_2 if c_2<1e10 else c_2
                    time_backward = time.time() - init_time
                    nn.utils.clip_grad_norm_(params_SDE, 1)
                    if (itercount_2 % 20 == 0):
                        print('training statistics')
                        print('iteration {}, sum_variance_van={:.4f}, variance_year={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount, sum_variances,var_pv_1cv[0].item(), time_forward, time_backward))
                    optimizer_SDE_2.step()
                    scheduler_SDE_2.step()

                if epoch>=250:
                    loss= exotic_prc_var_cv
                    init_time = time.time()
                    itercount_4 +=1
                    loss.backward()
                    time_backward = time.time() - init_time
                    if (itercount_4 % 20 == 0):
                        print('training exotic control variate')
                    optimizer_SDE_4.step()
                    scheduler_SDE_4.step()                
         
            with torch.no_grad():
                    pv_1cv, _,_,exotic_prc_no_cv, exotic_prc_cv,_, _,martingale_test, put_atm, call_atm, put_call_parity_error = model(S0, rate, z_1, MC_samples_gen, T, period_length=16,cal_exotic_var=1)
                    _, var_pv_1cv_val,_,_,_, exotic_prc_var_no_cv_val,exotic_prc_var_cv_val, _,_,_,_ = model(S0, rate, z_1_var,  MC_samples_var, T, period_length=16,cal_exotic_var=1)
            print("pred_1:",pv_1cv)
            print("target:", target_mat_T )
            pred=torch.reshape(pv_1cv,(6,21))
            print('pred', pred)
            loss_val = torch.sqrt(loss_fn(pred,target_mat_T))
            loss_vega = torch.sqrt(loss_fn(torch.mul(pred,vega),torch.mul(target_mat_T,vega)))
            loss_val_rel  = torch.sqrt(loss_fn(torch.div(pred,target_mat_T), torch.ones_like(pred)))
            print('validation {}, loss={:.10f}'.format(itercount, loss_val.item()))
            print('validation_rel {}, loss={:.10f}'.format(itercount, loss_val_rel.item()))
            print('validation_vega {}, loss={:.10f}'.format(itercount, loss_vega.item())) 
            exotic_hedge_error_change =torch.abs(1-exotic_hedge_error/exotic_prc_var_cv_val)  
            exotic_hedge_error = exotic_prc_var_cv_val
            strloss='loss_'+str(T)+'.txt'
            strvan_1_var = 'variance_van_1_'+str(T)+'.txt'
            strexotic_var_cv = 'variance_exotic_cv'+str(T)+'.txt'
            strexotic_var_no_cv = 'variance_exotic_no_cv'+str(T)+'.txt'
            strexotic_prc_cv = 'prc_exotic_cv'+str(T)+'.txt'
            strexotic_prc_no_cv = 'prc_exotic_no_cv'+str(T)+'.txt'
            strloss_rel='loss_rel'+str(T)+'.txt'
            strloss_vega='loss_vega'+str(T)+'.txt'
            strloss_martingale='martingale_test_'+str(T)+'.txt'
            strloss_call_atm='call_atm_'+str(T)+'.txt'
            strloss_put_atm='put_atm_'+str(T)+'.txt'
            strloss_put_call_parity_error = 'put_call_parity_error'+str(T)+'.txt'
            strIVerror='IV_error_'+str(T)+'.txt'
            strIVerror_infinity='IV_error_infinity'+str(T)+'.txt'
            print('Variance_V1:', torch.sum(var_pv_1cv_val))
            print('Variance_exotic with cv:', exotic_prc_var_cv_val)
            print('Variance_exotic without cv:', exotic_prc_var_no_cv_val)
            print('Price_exotic with cv:', exotic_prc_cv)
            print('Price_exotic without cv:', exotic_prc_no_cv)
            save_file(loss_val.cpu(), strloss)
            save_file(loss_val_rel.cpu(), strloss_rel)
            save_file(loss_vega.cpu(), strloss_vega)
            save_file(torch.sum(var_pv_1cv_val).cpu(), strvan_1_var)
            save_file(martingale_test.cpu(), strloss_martingale) 
            save_file(call_atm.cpu(), strloss_call_atm)
            save_file(put_atm.cpu(), strloss_put_atm) 
            save_file(put_call_parity_error.cpu(), strloss_put_call_parity_error)
            save_file(exotic_prc_var_cv_val.cpu(), strexotic_var_cv)
            save_file(exotic_prc_cv.cpu(), strexotic_prc_cv)
            save_file(exotic_prc_var_no_cv_val.cpu(), strexotic_var_no_cv)
            save_file(exotic_prc_no_cv.cpu(), strexotic_prc_no_cv)
            print('Martingale test:', martingale_test)
            print('Call price:', call_atm)
            print('Put price:', put_atm)
            print('Put call parity error:', put_call_parity_error)

            if torch.isnan(torch.sum(var_pv_1cv_val)):
                    checkpoint_str= 'Neural_SDE_maturity'+str(96)+'.pth.tar'
                    checkpoint=torch.load(checkpoint_str)
                    model.load_state_dict(checkpoint['state_dict'])
                    model = model.to(device) 

            iv_1 = np.ones_like(K)
            iv_1_out = torch.Tensor().to(device)
            iv_1_error = np.max(np.absolute(iv_target_1-iv_1))
           
            with torch.no_grad():
                   for idxt, t in enumerate(maturities):
                        van_pred_11 = pred[idxt,::].cpu().numpy()
                        for idxx, (pred_1, k) in enumerate(zip(van_pred_11, K)):
                          try:
                            iv_1[idxx] = implied_volatility(pred_1,  S=S0, K=k, r=0.0, t=t/(2*96), flag="c")
                          except:
                            pass
                        iv_1_out = torch.cat([iv_1_out,torch.from_numpy(iv_1).float().to(device).T],0)
            iv_max_error= loss_fn(iv_1_out,iv_target_1_out).cpu().numpy() #np.max(np.absolute(iv_1_out.cpu().numpy()-iv_target_1_out.cpu().numpy()))
            iv_infinity_error = np.max(np.absolute(iv_1_out.cpu().numpy()-iv_target_1_out.cpu().numpy()))
            
            print('iv_out', iv_1_out)
            print('iv_target', iv_target_1_out)
            print('iv_max_error', iv_max_error)
            print('iv_infinity_error', iv_infinity_error)
            save_file(iv_max_error, strIVerror)
            save_file(iv_infinity_error, strIVerror_infinity)
            
            if iv_max_error < best_IV_error and epoch<250:
                 model_best=model
                 best_IV_error=iv_max_error
                 print('current_loss', loss_val)
                 print('IV best error',iv_max_error)
                 filename = "Neural_SDE_maturity{}.pth.tar".format(T)
                 checkpoint = {"state_dict":model.state_dict(),
                         "T":T,
                         "pred":pv_1cv,
                         "target_mat_T": target_mat_T}
                 torch.save(checkpoint, filename)
                
            if epoch==249:
                  checkpoint_str= 'Neural_SDE_maturity'+str(96)+'.pth.tar'
                  checkpoint=torch.load(checkpoint_str)
                  model.load_state_dict(checkpoint['state_dict'])
                  model = model.to(device) 
                        
            if epoch >= 250 and exotic_hedge_error< best_hedge_error: # train exotic control variate for remaining epochs and return best model
                 model_best=model
                 best_hedge_error=exotic_hedge_error
                 print('current_loss', loss_val)
                 print('IV best error',iv_max_error)
                 print('best_hedge_error', best_hedge_error)
                 filename = "Neural_SDE_maturity{}.pth.tar".format(T)
                 checkpoint = {"state_dict":model.state_dict(),
                         "T":T,
                         "pred":pv_1cv,
                         "MSE": loss_val,
                         "iv_max_error": iv_max_error,
                         "MSE_rel": loss_val_rel,      
                         "Exotic_prc": exotic_prc,
                         "target_mat_T": target_mat_T}
                 torch.save(checkpoint, filename)
            if epoch>= 250 and exotic_hedge_error_change < 0.001:
                  break #if hedging strategy cannot be improved stop training

    return model_best   

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=7)
    parser.add_argument('--LAMBDA', type=int, default=5000)
    parser.add_argument('--c', type=int, default=10000)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device='cuda:{}'.format(args.device)
        torch.cuda.set_device(args.device)
    else:
        device="cpu"
       
  
    data = np.load("call_prices_mat_6_21_uniform.npy")/1000
    print('data',data) 
    strikes_call = np.load("strikes_mat_6_21_uniform.npy")/1000
    print('strikes_call', strikes_call)
    # Set up training
    losses=[]
    losses_val=[]
    S0 = 3.221 # initial asset price 
    rate = 0.0
    maturities = range(16, 97, 16)
    n_steps = 96
    timegrid = torch.linspace(0,0.5,n_steps+1)
    np.random.seed(901) # fix seed for reproducibility
    MC_samples_gen=1000000 # this is generated once and used to validate trained model after each epoch
    MC_samples_var=10000
    z_1 = np.random.normal(size=(MC_samples_gen, n_steps))
    z_1_var = np.random.normal(size=(MC_samples_var, n_steps))

    #  pass to torch
    z_1 = torch.tensor(z_1).to(device=device).float()
    z_1_var = torch.tensor(z_1_var).to(device=device).float()
 
    train_models(args)    