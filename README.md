# Assessing the Profitability of LSTM Trading For Ethereum

# Introduction

Within most applied dynamical systems, there often exists noise, or "random dynamics internal" to the very system itself, that contributes to the volatility of these systems over time [1]. Because the noise within a system represents the very essence of randomness itself, it is often the most noisy of dynamical systems that are the most difficult to predictively model. In finance, these dynamical systems often take the form of financial investments, such as stocks, market indices, and (recently) even cryptocurrencies. Being able to model these financial assets is crucial for investors; building a profitable portfolio necessitates being able to forecast potential reversals and continuations in the market. 

Traditionally, investors "forecasted" the market using indicators, like the Exponential Moving Average (EMA), to produce signals as to when to buy, sell, or hold stakes in an asset, in the hopes that such signals were to always precede major changes in the price of that asset. However, in the past few decades, the use of linear and non-linear models, such as Deep Learning Neural Networks, has become increasingly popular, and the idea of "forecasting the market" has evolved; to be able to forecast a financial asset, today, often refers to the use of mathematical models to precisely predict the price of that asset at some point in the future, using knowledge about how that asset has behaved in the past. 

The types of models that exist for such forecasting can be divided into two groups: linear and non-linear. The key advantage that non-linear models hold over the linear counterparts is that they are not reliant upon any prior assumptions about the systems they seek to model [2]; while linear models assume a normal distribution of its variables to capture any inter-linear dependencies, non-linear models do not and, therefore, are able to capture not only linear but also non-linear dependencies between variables [3]. Which group of models performs best ultimately depends on the asset itself [2]. 

Previous research has found cryptocurrencies to be non-linear in nature. They are not only "highly volatile in comparison to traditional currencies" but also the victim of "explosive" and "speculative behaviour" across "multiple periods" in time [2]. Further research has also found that the most prominent cryptocurrencies do not obey the principles of the Random Walk Hypothesis [4, 5], implying that the prices of these cryptocurrencies are in fact auto-regressive and justifying the use of a non-linear, auto-regressive forecasting model like the Long Short-Term Neural Network (LSTM) used in this study. 

Because cryptocurrencies are so volatile, technical indicators, like the Moving Absolute Convergence Divergence (MACD) and Stochastic Oscillator, may create false signals and therefore, in a logical conclusion, may not be as profitable [6]. The goal of my work in this study was to test the profitability of an LSTM Buy and Hold Strategy proposed in 2021 [7] but in the context of strictly cryptocurrencies. The performance of this strategy was assessed on only one cryptocurrency: Ethereum. In this study, I hypothesized that, because the robust architecture of a Stacked LSTM can effectively predict the highly volatile and non-linear dynamics of Ethereum better than most technical indicators, a Buy and Hold strategy that trades off of the predictions of that LSTM would be more profitable than a simpler yet more common strategy like the 5 & 15 EMA Crossover, which may produce false signals. Where more traditional strategies based on technical indicators like the EMA fail, I believe more robust strategies based on forecasting models like the LSTM may succeed. 

# A Brief Overview of the Stacked LSTM Architecture


For a given vector $$X$$ of price observations $$x_{1}, x_{2}, ..., x_{t}$$, the input, forget, and output gates for any said LSTM node are defined as such [8]: 


$$i_{t} = \sigma (W_{i}x_{t} + W_{hi}h_{t-1} + b_{i})$$




$$f_{t} = \sigma (W_{f}x_{t} + W_{hf}h_{t-1} + b_{f})$$


$$o_{t} = \sigma (W_{o}x_{t} + W_{ho}h_{t-1} + b_{o})$$



where $$\sigma$$ is the sigmoid activation function. $$W$$ and $$W_{h}$$ are matrices of weights unique to 
$$x_{t}$$ and $$h_{t-1}$$, respectively, and $$b$$ is a vector of bias parameters. The cell state at each timestamp is updated such that 




  $$c_{t} = f_{t} \times c_{t-1} + i_{t} \times \tilde{c_{t}}\text{,}$$


  $$\text{where } \tilde{c_{t}} = tanh(W_{c}x_{t} + W_{hc}h_{t-1} + b_{c})$$


and $$tanh$$ is the hyperbolic tangent activation function. The final output for any said LSTM layer at $$t$$ is therefore 


$$h_{t} = o_{t} \times tanh(c_{t})$$


where $$h_{t}$$ is then passed as $$h_{t-1}$$ into the next layer within a stacked, multi-layer architecture. Furthermore, the output produced by the Dense (and final) layer within our model is what is passed as $$h_{t-1}$$ for the first LSTM layer within the next timestamp. In this study, a Stacked LSTM of three LSTM layers and one Dense layer, with a linear activation, was used. All weights and biases were optimized using the _Adam_ Optimizer [9] upon a Mean Squared Error Loss Function.


# The LSTM Trading Strategy 

Because an LSTM already gives a good forecast, a more basic strategy was used [7]. If the forecasted price for tommorow is greater than the closing price of today, buy. If not, hold. 

# Features and Additional Notes

For the Stacked LSTM, four features were used: Closing Price, Trading Volume, MACD, and RSI. Sometimes, the use of these technical indicators as features can be effective [10]. While both MACD and RSI are of the same class of momentum indicators, please note that they are not redundant; MACD tracks the moving averages of price over time, while RSI tracks the relative velocities of that price for a given lookback period. 

If one wanted to use some but not all of the four features used in this study, they would need to merely change which columns they select from `prices_and_exogenous_features`. If you do this, please make sure to also change the corresponding line objects that are then selected for `pd.DataFrame` within the body of `class Prediction(bt.Indicator)`. If one wanted to use a technical indicator other than MACD or RSI as an exogenous feature, they would need to create a function that calculates that indicator. 


# References 

[1] Forgoston, E., & Moore, R. O. (2018). A primer on noise-induced transitions in applied dynamical systems. SIAM Review, 60(4), 969-1009.

[2] Dudek, G., Fiszeder, P., Kobus, P., & Orzeszko, W. (2024). Forecasting cryptocurrencies volatility using statistical and machine learning methods: A comparative study. Applied Soft Computing, 151, 111132.

[3] Bouteska, A., Abedin, M. Z., Hajek, P., & Yuan, K. (2024). Cryptocurrency price forecasting–A comparative analysis of ensemble learning and deep learning methods. International Review of Financial Analysis, 92, 103055.

[4] Palamalai, S., Kumar, K. K., & Maity, B. (2021). Testing the random walk hypothesis for leading cryptocurrencies. Borsa Istanbul Review, 21(3), 256-268.

[5] Tong, Z., Chen, Z., & Zhu, C. (2022). Nonlinear dynamics analysis of cryptocurrency price fluctuations based on Bitcoin. Finance Research Letters, 47, 102803.

[6] Lyukevich, I., Gorbatenko, I., & Bessonova, E. (2021, October). Cryptocurrency Market: Choice of Technical Indicators in Trading Strategies of Individual Investors. In Proceedings of the 3rd International Scientific Conference on Innovations in Digital Economy (pp. 408-416).

[7] Wang, B., & Zhang, X. Deep Learning Applying on Stock Trading. Stanford University, Tech. Rep., 2021.[Online]. Available: http://cs230. stanford. edu/projects spring 2021/reports/74. pdf.

[8] Bhandari, H. N., Rimal, B., Pokhrel, N. R., Rimal, R., Dahal, K. R., & Khatri, R. K. (2022). Predicting stock market index using LSTM. Machine Learning with Applications, 9, 100320.

[9] Kingma, D. P. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

[10] Him, C. K., & Pang, G. C. (2023). Stock Trend Prediction Using LSTM with MA, EMA, MACD and RSI Indicators. INTI Journal, 2023.
