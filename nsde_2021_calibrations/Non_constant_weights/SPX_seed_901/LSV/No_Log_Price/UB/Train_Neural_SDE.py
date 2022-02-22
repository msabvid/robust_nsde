import sys
import os
sys.path.append(os.path.dirname('__file__'))

import torch
import torch.nn as nn
import numpy as np
import math
import time
import argparse


from py_vollib.black_scholes.implied_volatility import implied_volatility 
from py_vollib.black_scholes import black_scholes as price_black_scholes 

from networks import *
         
class Net_LSV(nn.Module):
    """
    Calibration of LSV model to vanilla prices at different maturities
    """
    def __init__(self, timegrid, strikes_call, device, n_maturities):
        
        super(Net_LSV, self).__init__()
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        
        # initialise price diffusion neural network (different neural network for each maturity)
        self.S_vol =  Net_timegrid(dim=3, nOut=1, n_layers=3, vNetWidth=100, n_maturities=n_maturities, activation_output="softplus")
        
        # initialise vanilla hedging strategy neural networks 
        """
        network for each maturity is used to hedge only options for that maturity, for example network corresponding to final maturity
        is used to simulate hedging strategy (from time 0) for vanilla options at the final maturity
        """
        self.vanilla_hedge = Net_timegrid(dim=2, nOut=len(strikes_call), n_layers=2, vNetWidth=20, n_maturities=n_maturities, activation_output="softplus")

        # initialise stochastic volatility drift and diffusion neural networks
        self.V_drift = Net_timegrid(dim=1, nOut=1, n_layers=2, vNetWidth=20, n_maturities=n_maturities)
        self.V_vol = Net_timegrid(dim=1, nOut=1, n_layers=2, vNetWidth=20, n_maturities=n_maturities, activation_output="softplus")
        
        # initialise stochastic volatility correlation and initial value parameters
        self.v0 = torch.nn.Parameter(torch.rand(1)*0.1)
        self.rho = torch.nn.Parameter(2*torch.rand(1)-1)
        
        # initialise exotic hedging strategy neural networks 
        """
        network for each maturity is used only to simulate hedging strategy for the timesteps in between previous and current maturity
        so that "n_maturities" networks are used to simulate hedging strategy for exotic option with maturity at the end of considered time horizon
        """
        self.exotic_hedge = Net_timegrid(dim=2, nOut=1, n_layers=2, vNetWidth=20, n_maturities=n_maturities)


    def forward(self, S0, rate, z, MC_samples, ind_T, period_length,n_maturities): 
        """
        pv_h:=price vanilla hedged pv_var_h:=variance of hedged vanilla price
        pe_h:=price exotic hedged; pe_var_h:=variance of hedged exotic price
        pe_u:=price exotic unhedged; pe_var_u:=variance of unhedged exotic price
        cv_vanilla_fwd-stores output of vanilla hedging strategies at each timestep
        """
        ones = torch.ones(MC_samples, 1, device=self.device)
        S_old = ones * S0
        cv_vanilla = torch.zeros(S_old.shape[0], len(self.strikes_call),n_maturities, device=self.device)
        cv_exotic = torch.zeros(S_old.shape[0], 1 , device=self.device)
        pv_h = torch.zeros(len(self.strikes_call), device=self.device)
        pv_var_h = torch.zeros_like(pv_h)
        pe_u = torch.zeros(1, device=self.device)
        pe_h = torch.zeros(1, device=self.device)
        pe_var_h = torch.zeros(1, device=self.device)
        pe_var_u = torch.zeros(1, device=self.device)
        cv_vanilla_fwd = torch.zeros(S_old.shape[0], len(self.strikes_call),n_maturities, device=self.device)
        pv_h_out = torch.Tensor().to(device)  # used to store model vanilla prices for all maturities
        pv_var_h_out = torch.Tensor().to(device) # used to store variances of vanilla options for all maturities
        martingale_test = torch.zeros(1, device=self.device)
        put_atm = torch.zeros(1, device=self.device)
        call_atm = torch.zeros(1, device=self.device)
        V_old = ones * torch.sigmoid(self.v0)*0.5
        rho = torch.tanh(self.rho)
        running_max = S_old # initialisation of running_max for calculation of lookback price
        
        with open("rho_and_v0.txt","a") as f:
            f.write("{:.4f},{:.4f}\n".format(rho.item(),V_old[0,0].item()))

        for i in range(1, ind_T+1):
            idx_net = (i-1)//period_length # assume maturities are evenly distributed, i.e. 0, 16, 32, ..., 96
            t = torch.ones_like(S_old) * self.timegrid[i-1]
            h = self.timegrid[i]-self.timegrid[i-1]   
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(MC_samples,1)
            zz = torch.randn_like(dW)
            dB = rho * dW + torch.sqrt(1-rho**2)*torch.sqrt(h)*zz
            
            price_diffusion = S_old*self.S_vol.forward_idx(idx_net, torch.cat([t,S_old,V_old],1))
            
            # Evaluate vanilla hedging strategies at particular timestep
            for mat in range(n_maturities):
                if mat>=idx_net:
                    cv_vanilla_fwd[:,:,mat] = self.vanilla_hedge.forward_idx(mat, torch.cat([t,S_old.detach()],1))
           
            exotic_hedge = self.exotic_hedge.forward_idx(idx_net, torch.cat([t,S_old.detach()],1))
            V_new = V_old + self.V_drift.forward_idx(idx_net,V_old).reshape(MC_samples,1)*h + self.V_vol.forward_idx(idx_net,V_old).reshape(MC_samples,1)*dB
            price_diffusion_const = price_diffusion.detach()

            drift = rate*S_old.reshape(MC_samples,1)
            diff =  price_diffusion.reshape(MC_samples,1) * dW

            # Drift normalisations in tamed Euler scheme have gradient detached
            drift_c = (1+torch.abs(drift.detach())*torch.sqrt(h))
            diff_c = (1+torch.abs(price_diffusion_const.reshape(MC_samples,1))*torch.sqrt(h))
           
            # Tamed Euler step
            S_new = S_old + drift*h/drift_c + diff/diff_c 
            
            # Discounted price change between timesteps; used for control variate training       
            dS = torch.exp(-rate*self.timegrid[i])*S_new - torch.exp(-rate*self.timegrid[i-1])*S_old
           
            # Evaluate stoch. integrals (w.r.t. discounted price process) corresponding to control variates
            for mat in range(n_maturities):
                 if mat>=idx_net:
                     cv_vanilla[:,:,mat] += cv_vanilla_fwd[:,:,mat] * dS.detach().repeat(1,len(self.strikes_call))  
            cv_exotic +=  exotic_hedge.reshape(MC_samples,1) * dS.detach()
            
            # Evaluate exotic payoff 
            running_max = torch.max(running_max, S_new) 
            
            # Update values of asset prc processes for next Tamed Euler step
            S_old = S_new
            V_old = torch.clamp(V_new,0)
                  
            # Evaluate vanilla option prices and variances of price estimate if timestep corresponds to maturity  
            if int(i) in maturities: 
               dis_fact = torch.exp(-rate*self.timegrid[(idx_net + 1)*period_length])
               for idx, strike in enumerate(self.strikes_call):
                  pv_h_temp = dis_fact*torch.clamp(S_old-strike,0).squeeze(1).detach()-cv_vanilla[:,idx,idx_net]
                  pv_h_detach_cv = dis_fact*torch.clamp(S_old-strike,0).squeeze(1)-cv_vanilla[:,idx,idx_net].detach()
                  pv_h[idx] = pv_h_detach_cv.mean() # gradient of cv not applied when evaluating price of vanilla options 
                  pv_var_h[idx] = pv_h_temp.var() # gradient of price not applied when evaluating variance of vanilla options
               pv_h_out = torch.cat([pv_h_out,pv_h.T],0) 
               pv_var_h_out = torch.cat([pv_var_h_out,pv_var_h.T],0)
                
            # Evaluate exotic option price and variance of exotic price estimates at final maturity (and martingale test)
            if int(i)==n_maturities*period_length:  
                  martingale_test = dis_fact*S_old.detach()
                  martingale_test = martingale_test.mean()/S0
                  put_atm = dis_fact*torch.clamp(S0-S_old.detach(),0).squeeze(1)
                  put_atm = put_atm.mean().detach()
                  call_atm = dis_fact*torch.clamp(S_old.detach()-S0,0).squeeze(1)
                  call_atm = call_atm.mean()
                  put_call_parity_error = call_atm-put_atm - S0 + dis_fact*S0
                  exotic_payoff = running_max - S_old
                  exotic_payoff_detach = exotic_payoff.detach() # detach dependence on price for estimating variance of exotic
                  pe_h_temp = dis_fact*exotic_payoff_detach.squeeze(1).reshape(MC_samples,1)-cv_exotic.reshape(MC_samples,1)
                  pe_u = dis_fact*exotic_payoff.squeeze(1)
                  pe_var_h = pe_h_temp.var()
                  pe_var_u = pe_u.var()
                  pe_u = pe_u.mean() 
                  pe_h = dis_fact*exotic_payoff.squeeze(1).reshape(MC_samples,1)-cv_exotic.detach().reshape(MC_samples,1)
                  pe_h = pe_h.mean()

        return pv_h_out, pv_var_h_out, pe_u, pe_h, pe_var_u, pe_var_h, martingale_test, put_atm, call_atm, put_call_parity_error
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.5)

def train_nsde(model,z_val,z_val_var,z_test,z_test_var,config):
    
    itercount = 0
    loss_fn = nn.MSELoss() 
    
    maturities = config["maturities"]
    n_maturities = config["n_maturities"]
    model = model.to(device)
    n_epochs = config["n_epochs"]
    batch_size = config["batch_size"]
    batch_size_hedge = config["batch_size_hedge"]
    batch_steps=config["maturities"][-1]
    T=config["maturities"][-1]
    learning_rate =config["learning_rate"]
    K = config["strikes_call"]
    rate = config["interest_rate"]
    S0 = config["initial_price"]
    seed = config["seed"]
    n_strikes = len(K)

    params_SDE = list(model.S_vol.parameters()) + [model.rho, model.v0] + list(model.V_drift.parameters())+list(model.V_vol.parameters())
    params_vanilla_hedge = list(model.vanilla_hedge.parameters())
    params_exotic_hedge = list(model.exotic_hedge.parameters())

    optimizer_vanilla_hedge = torch.optim.Adam(params_vanilla_hedge,lr=100*learning_rate)
    optimizer_SDE = torch.optim.Adam(params_SDE,lr=learning_rate)
    optimizer_exotic_hedge = torch.optim.Adam(params_exotic_hedge,lr=100*learning_rate)

    best_IV_mean_error = 10
    target_mat_T = torch.tensor(config["target_data"][:len(config["maturities"]),:len(config["strikes_call"])], device=device).float()
       
    LAMBDA = args.LAMBDA
    c = args.c
        
    # Number of paths used for validation
    MC_samples_price = args.MC_samples_price
    MC_samples_var = args.MC_samples_var
    
    if args.no_bound:
        type_bound = "no_bound"
        opt_constant = 0
    elif args.lower_bound:
        type_bound = "lower_bound"
        opt_constant = 1
    else:
        type_bound = "upper_bound"
        opt_constant = -1
        
    for epoch in range(n_epochs):
        
        # evaluate model at initialisation and save error, exotic price and other statistics at initialisation
        if epoch==0:

            # calculate IV of target options
            iv_target = np.zeros_like(K)
            iv_target_out = torch.Tensor().to(device)
            try:
                for idxt, t in enumerate(maturities):
                    van_target = target_mat_T[idxt,::].cpu().numpy() 
                    for idxx, ( target, k) in enumerate(zip(van_target, K)):
                        iv_target[idxx] = implied_volatility(target,  S=S0, K=k, r=rate, t=t/(2*T), flag="c") 
                    iv_target_out = torch.cat([iv_target_out,torch.from_numpy(iv_target).float().to(device).T],0)  
            except:
                pass
            print('Target Implied Volatility Surface',iv_target_out)
                
            # for each of the target options calculate inverse of the vega (used as calibration weighting scheme) 
            inverse_vega_target = np.ones_like(K)
            inverse_vega_out = torch.Tensor().to(device)
            constant_shift=0.01 # vega of each option is calculated by shifting corresponding IV by 1% 
                                    
            try:
                for idxt, t in enumerate(maturities):
                    van_target = target_mat_T[idxt,::].cpu().numpy() 
                    for idxx, (target, k) in enumerate(zip(van_target, K)):
                        iv_target_temp = implied_volatility(target, S=S0, K=k, r=rate, t=t/(2*T), flag="c") 
                        iv = iv_target_temp+constant_shift
                        inverse_vega_target[idxx] = constant_shift/(price_black_scholes(flag='c',S=S0,K=k,t=t/(2*T),r=rate,sigma=iv)-target)
                    inverse_vega_out = torch.cat([inverse_vega_out,torch.from_numpy(inverse_vega_target).float().to(device).T],0)                    
            except:
                    pass

            inverse_vega=torch.reshape(inverse_vega_out,(n_maturities,n_strikes))
            
            #normalise the weights 
            inverse_vega=(n_strikes*n_maturities)*inverse_vega/torch.sum(inverse_vega)
            
            print('Weighting scheme for target options:',inverse_vega)
                
            with torch.no_grad():
                pv_h, _,pe_u,pe_h,_, _,martingale_test, put_atm, call_atm, put_call_parity_error = model(S0, rate, z_val, MC_samples_price, batch_steps, period_length=period_length,n_maturities=n_maturities)
                _, var_pv_h,_,_,pe_var_u, pe_var_h,_,_,_,_ = model(S0, rate, z_val_var,MC_samples_var, batch_steps, period_length=period_length,n_maturities=n_maturities)                 
            pred=torch.reshape(pv_h,(n_maturities,n_strikes))
            print("Model Option Prices at Initialisation:",pred)
            print("Target Option Call Prices:", target_mat_T )
            loss_val = torch.sqrt(loss_fn(pred,target_mat_T))
            loss_val_rel  = torch.sqrt(loss_fn(torch.div(pred,target_mat_T), torch.ones_like(pred)))
            loss_vega = torch.sqrt(loss_fn(torch.mul(pred,inverse_vega),torch.mul(target_mat_T,inverse_vega)))
            
            with open("Price_Call_and_Price_Put_6_Month_ATM.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(call_atm.item(),put_atm.item() ))
            
            with open("Price_Exotic_Hedged_and_Price_Exotic_Unhedged.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(pe_h.item(),pe_u.item() ))
            
            with open("Sum_Variances__Vanilla_Options_Variance_Exotic_Hedged_Variance_Exotic_Unhedged.txt","a") as f:
                f.write("{:.4f},{:.4f},{:.4f}\n".format(torch.sum(var_pv_h).item(),pe_var_h.item(),pe_var_u.item() ))
            
            with open("Loss_MSE_Loss_REL_Loss_Vega_Weighted.txt","a") as f:
                f.write("{:.4f},{:.4f},{:.4f}\n".format(loss_val.item(),loss_val_rel.item(),loss_vega.item() ))
            
            with open("Put_Call_Parity_Error_and_Martingale_Test.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(put_call_parity_error.item(),martingale_test.item() ))
            
            print('Validation Mean Square Error At Initialisation {}, loss={:.10f}'.format(itercount, loss_val.item()))
            print('Validation Relative Error At Initialisation {}, loss={:.10f}'.format(itercount, loss_val_rel.item()))
            print('Validation Iverse Vega Weighted MSE At Initialisation {}, loss={:.10f}'.format(itercount, loss_vega.item()))   
            print('Sum of Variances of Vanilla Call Options At Initialisation:', torch.sum(var_pv_h))
            print('Variance of Hedged Exotic Option At Initialisation:', pe_var_h)
            print('Variance of Unhedged Exotic Option At Initialisation:', pe_var_u)
            print('Price of Hedged Exotic Option At Initialisation:', pe_h)
            print('Price of Unhedged Exotic Option At Initialisation:', pe_u)
            print('Martingale Test At Initialisation:', martingale_test)
            print('ATM 6-Month Call Price At Initialisation:', call_atm)
            print('ATM 6-Month Put Price At initialisation:', put_atm)
            print('ATM 6-Month Put-Call Parity Error At Initialisation:', put_call_parity_error)

                        
        print('Epoch, batch size:', epoch, batch_size)
        for i in range(0,20*batch_size, batch_size):
            # simulate paths over entire training horizon 

            optimizer_SDE.zero_grad()
            optimizer_vanilla_hedge.zero_grad()
            optimizer_exotic_hedge.zero_grad()
                
            # Start training CV networks from 1st epoch, at initial (0th) epoch neural networks corresponding to SDE parameters are trained  
            
            # Train Vanilla Hedging Strategy 
            if epoch%10==1: 
                model.vanilla_hedge.unfreeze()
                model.S_vol.freeze()
                model.V_drift.freeze()
                model.V_vol.freeze()
                model.exotic_hedge.freeze()
                init_time = time.time()
                batch_x1 = torch.randn(batch_size_hedge, batch_steps, device=device)
                _, var_pv_h, _,_,_, _, _, _, _, _ = model(S0, rate, batch_x1, batch_size_hedge,T, period_length=period_length,n_maturities=n_maturities)
                time_forward = time.time() - init_time
                loss = var_pv_h.sum()
                init_time = time.time()
                itercount += 1
                loss.backward()
                time_backward = time.time() - init_time
                if (itercount % 20 == 0):
                    print('Training Vanilla Hedging Strategy')
                    print('iteration {}, sum_variance_vanilla={:.4f},time_forward={:.4f}, time_backward={:.4f}'.format(itercount, loss.item(), time_forward, time_backward))
                optimizer_vanilla_hedge.step()
            
            # Train Neural Networks Corresponding to SDE Parameters 
            if epoch%10!=1 and epoch%10!=2: 
                model.S_vol.unfreeze()
                model.V_drift.unfreeze()
                model.V_vol.unfreeze()
                model.vanilla_hedge.freeze()
                model.exotic_hedge.freeze()
                init_time = time.time()
                batch_x1 = torch.randn(batch_size, batch_steps, device=device)
                pv_h, _, _,pe_h,_, _, _, put_atm, call_atm, put_call_parity_error = model(S0, rate, batch_x1, batch_size,T, period_length=period_length,n_maturities=n_maturities)
                time_forward = time.time() - init_time
                pred = torch.reshape(pv_h,(n_maturities,n_strikes))
                MSE = loss_fn(torch.mul(pred,inverse_vega),torch.mul(target_mat_T,inverse_vega))
                loss= opt_constant*pe_h + LAMBDA * MSE + c/2 * MSE**2  
                init_time = time.time()
                itercount +=1
                loss.backward()
                if itercount%30==0:
                    LAMBDA += c*MSE.detach() if LAMBDA<1e6 else LAMBDA
                    c = 2*c if c<1e10 else c
                time_backward = time.time() - init_time
                nn.utils.clip_grad_norm_(params_SDE, 5)
                if (itercount % 20 == 0):
                    print('Training SDE Parameters')
                    print('iteration {}, augmented_Lagrangian_loss={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount,loss.item(), time_forward, time_backward))
                optimizer_SDE.step()
            
            # Train Neural Network Corresponding to Exotic Hedging Strategy
            if epoch%10==2: 
                model.S_vol.freeze()
                model.V_drift.freeze()
                model.V_vol.freeze()
                model.vanilla_hedge.freeze()
                model.exotic_hedge.unfreeze()
                init_time = time.time()
                batch_x1 = torch.randn(batch_size_hedge, batch_steps, device=device)
                _, _, _,_, _,pe_var_h,_, _, _, _ = model(S0, rate, batch_x1, batch_size_hedge,T, period_length=period_length,n_maturities=n_maturities)
                time_forward = time.time() - init_time
                loss= pe_var_h
                init_time = t
                itercount +=1
                loss.backward()
                time_backward = time.time() - init_time
                if (itercount % 20 == 0):
                    print('Training Exotic Hedging Strategy')
                    print('iteration {}, variance_exotic_option={:.4f}, time_backward={:.4f}'.format(itercount, loss.item(), time_forward, time_backward))
                optimizer_exotic_hedge.step()                                    
         
        with torch.no_grad():
            pv_h_val, _,pe_u_val, pe_h_val,_, _,martingale_test, put_atm, call_atm, put_call_parity_error = model(S0, rate, z_val, MC_samples_price, T, period_length=period_length,n_maturities=n_maturities)
            _, var_pv_h_val,_,_, pe_var_u_val,pe_var_h_val, _,_,_,_ = model(S0, rate, z_val_var,  MC_samples_var, T, period_length=period_length,n_maturities=n_maturities)
        
        
        pred=torch.reshape(pv_h_val,(n_maturities,n_strikes))
        print("Model Option Prices:",pred)
        print("Target Option Call Prices:", target_mat_T )
        loss_val = torch.sqrt(loss_fn(pred,target_mat_T))
        loss_val_rel  = torch.sqrt(loss_fn(torch.div(pred,target_mat_T), torch.ones_like(pred)))
        loss_vega = torch.sqrt(loss_fn(torch.mul(pred,inverse_vega),torch.mul(target_mat_T,inverse_vega)))
            
        with open("Price_Call_and_Price_Put_6_Month_ATM.txt","a") as f:
            f.write("{:.4f},{:.4f}\n".format(call_atm.item(),put_atm.item() ))
            
        with open("Price_Exotic_Hedged_and_Price_Exotic_Unhedged.txt","a") as f:
            f.write("{:.4f},{:.4f}\n".format(pe_h_val.item(),pe_u_val.item() ))
            
        with open("Sum_Variances__Vanilla_Options_Variance_Exotic_Hedged_Variance_Exotic_Unhedged.txt","a") as f:
            f.write("{:.4f},{:.4f},{:.4f}\n".format(torch.sum(var_pv_h_val).item(),pe_var_h_val.item(),pe_var_u_val.item() ))
            
        with open("Loss_MSE_Loss_REL_Loss_Vega_Weighted.txt","a") as f:
            f.write("{:.4f},{:.4f},{:.4f}\n".format(loss_val.item(),loss_val_rel.item(),loss_vega.item() ))
            
        with open("Put_Call_Parity_Error_and_Martingale_Test.txt","a") as f:
            f.write("{:.4f},{:.4f}\n".format(put_call_parity_error.item(),martingale_test.item() ))
            
        print('Validation Mean Square Error {}, loss={:.10f}'.format(itercount, loss_val.item()))
        print('Validation Relative Error {}, loss={:.10f}'.format(itercount, loss_val_rel.item()))
        print('Validation Iverse Vega Weighted MSE {}, loss={:.10f}'.format(itercount, loss_vega.item()))   
        print('Sum of Variances of Vanilla Call Options:', torch.sum(var_pv_h_val))
        print('Variance of Hedged Exotic Option:', pe_var_h_val)
        print('Variance of Unhedged Exotic Option:', pe_var_u_val)
        print('Price of Hedged Exotic Option:', pe_h_val)
        print('Price of Unhedged Exotic Option:', pe_u_val)
        print('Martingale Test:', martingale_test)
        print('ATM 6-Month Call Price:', call_atm)
        print('ATM 6-Month Put Price:', put_atm)
        print('ATM 6-Month Put-Call Parity Error:', put_call_parity_error)

        iv = np.ones_like(K)
        iv_out = torch.Tensor().to(device)         
        with torch.no_grad():
                for idxt, t in enumerate(maturities):
                    van_pred_temp = pred[idxt,::].cpu().numpy()
                    for idxx, (pred_temp, k) in enumerate(zip(van_pred_temp, K)):
                        try:
                            iv[idxx] = implied_volatility(pred_temp,  S=S0, K=k, r=rate, t=t/(2*T), flag="c")
                        except:
                            pass
                    iv_out = torch.cat([iv_out,torch.from_numpy(iv).float().to(device).T],0)
        inverse_vega += (n_strikes*n_maturities)*torch.clamp(torch.abs(iv_out-iv_target_out).reshape(n_maturities,n_strikes), 0.001,0.5)
        inverse_vega = (n_strikes*n_maturities)*inverse_vega/torch.sum(inverse_vega)
        print('Vega weights', inverse_vega)
                                      
        iv_mean_error= loss_fn(iv_out,iv_target_out).cpu().numpy() 
        iv_infinity_error = np.max(np.absolute(iv_out.cpu().numpy()-iv_target_out.cpu().numpy()))
        with open("IV_Mean_Infinity_Errors.txt","a") as f:
             f.write("{:.4f},{:.4f}\n".format(iv_mean_error.item(),iv_infinity_error.item())) 
            
        print('Implied Volatility Model', iv_out)
        print('Implied Volatility Target', iv_target_out)
        print('Implied Volatility Mean Square Error', iv_mean_error)
        print('Implied Volatility Infinity_error', iv_infinity_error)
            
        if iv_mean_error < best_IV_mean_error:
            model_best=model
            best_IV_mean_error=iv_mean_error
            print('current_loss', loss_val)
            print('IV best error',iv_mean_error)
            filename = "Neural_SDE_validation_{}_maturity_{}_LSV_{}.pth.tar".format(type_bound,T,seed)
            checkpoint = {"state_dict":model.state_dict(),
                         "T":T,
                         "pred":pred,
                         "MSE": loss_val,
                         "exotic_hedge_error": pe_var_h_val,
                         "iv_MSE_error": iv_mean_error,
                         "iv_infinity_error": iv_infinity_error,
                         "MSE_rel": loss_val_rel,
                         "Exotic_prc": pe_h_val,
                         "target_mat_T": target_mat_T}
            torch.save(checkpoint, filename)
                
        if  epoch==(n_epochs-1):
            checkpoint_str= "Neural_SDE_validation_{}_maturity_{}_LSV_{}.pth.tar".format(type_bound,T,seed)

            checkpoint=torch.load(checkpoint_str)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device) 
            
            filename = "Neural_SDE_test_{}_maturity_{}_LSV_{}.pth.tar".format(type_bound,T,seed)

            with torch.no_grad():
                pv_h_test,_,pe_u_test, pe_h_test,_, _,martingale_test, put_atm, call_atm, put_call_parity_error = model(S0, rate, z_test, MC_samples_price, T, period_length=period_length,n_maturities=n_maturities)
                _, var_pv_h_test,_,_, pe_var_u_test,pe_var_h_test, _,_,_,_ = model(S0, rate, z_test_var,  MC_samples_var, T, period_length=period_length,n_maturities=n_maturities)
            
            pred=torch.reshape(pv_h_test,(n_maturities,n_strikes))
            print("Model Option Prices:",pred)
            print("Target Option Call Prices:", target_mat_T )
        
            loss_test = torch.sqrt(loss_fn(pred,target_mat_T))
            loss_test_rel  = torch.sqrt(loss_fn(torch.div(pred,target_mat_T), torch.ones_like(pred)))
            loss_vega = torch.sqrt(loss_fn(torch.mul(pred,inverse_vega),torch.mul(target_mat_T,inverse_vega)))
            
            with open("Price_Call_and_Price_Put_6_Month_ATM.txt","a") as f:
                 f.write("{:.4f},{:.4f}\n".format(call_atm.item(),put_atm.item() ))
            
            with open("Price_Exotic_Hedged_and_Price_Exotic_Unhedged.txt","a") as f:
                 f.write("{:.4f},{:.4f}\n".format(pe_h_test.item(),pe_u_test.item() ))
            
            with open("Sum_Variances__Vanilla_Options_Variance_Exotic_Hedged_Variance_Exotic_Unhedged.txt","a") as f:
                 f.write("{:.4f},{:.4f},{:.4f}\n".format(torch.sum(var_pv_h_test).item(),pe_var_h_test.item(),pe_var_u_test.item() ))
            
            with open("Loss_MSE_Loss_REL_Loss_Vega_Weighted.txt","a") as f:
                 f.write("{:.4f},{:.4f},{:.4f}\n".format(loss_test.item(),loss_test_rel.item(),loss_vega.item() ))
            
            with open("Put_Call_Parity_Error_and_Martingale_Test.txt","a") as f:
                 f.write("{:.4f},{:.4f}\n".format(put_call_parity_error.item(),martingale_test.item() ))
            
            print('Test Mean Square Error {}, loss={:.10f}'.format(itercount, loss_test.item()))
            print('Validation Relative Error {}, loss={:.10f}'.format(itercount, loss_test_rel.item()))
            print('Validation Iverse Vega Weighted MSE {}, loss={:.10f}'.format(itercount, loss_vega.item()))   
            print('Sum of Variances of Vanilla Call Options:', torch.sum(var_pv_h_test))
            print('Variance of Hedged Exotic Option:', pe_var_h_test)
            print('Variance of Unhedged Exotic Option:', pe_var_u_test)
            print('Price of Hedged Exotic Option:', pe_h_test)
            print('Price of Unhedged Exotic Option:', pe_u_test)
            print('Martingale Test:', martingale_test)
            print('ATM 6-Month Call Price:', call_atm)
            print('ATM 6-Month Put Price:', put_atm)
            print('ATM 6-Month Put-Call Parity Error:', put_call_parity_error)
                     
            iv = np.ones_like(K)
            iv_out = torch.Tensor().to(device)
            
            with torch.no_grad():
                for idxt, t in enumerate(maturities):
                    van_pred_temp = pred[idxt,::].cpu().numpy()
                    for idxx, (pred_temp, k) in enumerate(zip(van_pred_temp, K)):
                        try:
                            iv[idxx] = implied_volatility(pred_temp,  S=S0, K=k, r=rate, t=t/(2*96), flag="c")
                        except:
                            pass
                    iv_out = torch.cat([iv_out,torch.from_numpy(iv).float().to(device).T],0)
                    
            iv_mean_error= loss_fn(iv_out,iv_target_out).cpu().numpy() 
            iv_infinity_error = np.max(np.absolute(iv_out.cpu().numpy()-iv_target_out.cpu().numpy()))
            
            print('iv_out_test', iv_out)
            print('iv_target', iv_target_out)
            print('iv_mean_error_test', iv_mean_error)
            print('iv_infinity_error_test', iv_infinity_error)
            with open("IV_Mean_Infinity_Errors.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(iv_mean_error.item(),iv_infinity_error.item())) 
            
            checkpoint = {"state_dict":model.state_dict(),
                         "T":T,
                         "pred":pred,
                         "MSE": loss_test,
                         "exotic_hedge_error": pe_var_h_test,
                         "iv_MSE_error": iv_mean_error,
                         "iv_infinity_error": iv_infinity_error,
                         "MSE_rel": loss_test_rel,
                         "Exotic_prc": pe_h_test,
                         "target_mat_T": target_mat_T}
            torch.save(checkpoint, filename)
                  
    return model_best   

if __name__ == '__main__':

    MC_samples_price=1000000 # this is generated once and used to validate trained model after each epoch
    MC_samples_var=400000
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=7)
    parser.add_argument('--LAMBDA', type=int, default=10000)
    parser.add_argument('--c', type=int, default=20000)
    parser.add_argument('--MC_samples_price',type=int,default=MC_samples_price)
    parser.add_argument('--MC_samples_var',type=int,default=MC_samples_var)
    parser.add_argument('--lower_bound', action='store_true', default=False)
    parser.add_argument('--upper_bound', action='store_true', default=True)
    parser.add_argument('--no_bound', action='store_true', default=False)
    
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        device='cuda:{}'.format(args.device)
        torch.cuda.set_device(args.device)
    else:
        device="cpu"
        
    # Set up training   
    data = np.load("call_prices_mat_6_21_uniform.npy")/1000
    print('data',data) 
    strikes_call = np.load("strikes_mat_6_21_uniform.npy")/1000
    print('strikes_call', strikes_call)
    n_steps=96
    timegrid = torch.linspace(0,0.5,n_steps+1).to(device)
    maturities =  range(16, 97, 16)
    period_length = 16
    n_maturities = len(maturities)
    S0 = 3.221 # initial asset price 
    rate = 0.0 # risk-free rate
    n_steps = period_length*n_maturities
    timegrid = torch.linspace(0,0.5,n_steps+1).to(device)

    seed=901

    torch.manual_seed(seed) # fixed for reproducibility
    model = Net_LSV(timegrid=timegrid, strikes_call=strikes_call, device=device, n_maturities=n_maturities)
    model.to(device)
    model.apply(init_weights)
    
    np.random.seed(seed) # fix seed for reproducibility
    z_val = np.random.normal(size=(MC_samples_price, n_steps))
    z_val_var = np.random.normal(size=(MC_samples_var, n_steps))
    np.random.seed(1234567) # fixed seed for test set model is evaluated after calibration completes 
    z_test = np.random.normal(size=(MC_samples_price, n_steps))
    z_test_var = np.random.normal(size=(MC_samples_var, n_steps))
    
    #  Random samples used to generate paths used for testing
    z_val = torch.tensor(z_val).to(device=device).float()
    z_val_var = torch.tensor(z_val_var).to(device=device).float()
    z_test = torch.tensor(z_test).to(device=device).float()
    z_test_var = torch.tensor(z_test_var).to(device=device).float()


    CONFIG = {"batch_size":50000,
              "batch_size_hedge":10000,
              "n_epochs":200,
              "initial_price":S0,
              "maturities":maturities,
              "learning_rate": 0.0001,
              "interest_rate":rate,
              "n_maturities":n_maturities,
              "strikes_call":strikes_call,
              "timegrid":timegrid,
              "n_steps":n_steps,
              "target_data":data,
              "seed":seed}
    
    model = train_nsde(model, z_val,z_val_var,z_test,z_test_var, CONFIG)
    