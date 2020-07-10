# Robust Pricing and Hedging via Neural SDEs

Code of numerical experiments in this [paper](https://arxiv.org/abs/2007.04154).

      @misc{gierjatowicz2020robust,
      title={Robust pricing and hedging via neural SDEs},
      author={Patryk Gierjatowicz and Marc Sabate-Vidales and David Šiška and Lukasz Szpruch and Žan Žurič},
      year={2020},
      eprint={2007.04154},
      archivePrefix={arXiv},
      primaryClass={q-fin.MF}
  }

...


## Target data
The file `Call_prices_59.pt` contains the target Vanilla call option prices generated with Heston model for bi-monthly maturities up to 1 year, and 21 different strikes between K=0.8 and K=1.2.

![Heston](/images/Heston.png)

Heston model parameters:

![params](/images/params_target.png)

Resulting target IV surface:

![Target data](/images/target_iv_surface.png)


## Scripts

* `nsde_LV.py`: Calibration to target prices of Neural SDE using Local Volatility model.
      
      python nsde_LV.py --device 0 --vNetWidth 50 --n_layers 20

* `nsde_LSV.py`: Calibration to target prices of Neural SDE using Local Stochastic Volatility model, where \sigma^S, b^V and \sigma^V are feed-forward neural networks:
![LSV](/images/Neural_SDE.png)
      
      python nsde_LV.py --device 0 --vNetWidth 50 --n_layers 20



