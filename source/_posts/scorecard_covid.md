---
title: Developing and Calibrating Credit Risk Scorecards in COVID scenario
date: 16th June, 2020
tags: [Credit Risk Scorecards, Underwriting, Coronavirus]
---

![](/images/scorecards/img7.png)

The COVID-19 pandemic has impacted most of the countries in the world bringing economic activity to a standstill. The IMF has already warned that this pandemic will bring the ‘worst recession’ since the Great Depression in the 1930s, expecting a 3% reduction in global GDP in 2020. The IMF is now projecting India’s growth rate for FY21 to be 1.9%, down from 5.8% estimated in January (source: https://www.imf.org/en/Countries/IND).

Now since COVID-19 is a one-off event, the Risk teams of a Bank or a Non-Banking Financial Company (NBFC) or any credit lending start-up are having a hard time. Not only there is a collective effort to reduce risk and take appropriate action, but there is also a need for developing strategies & solutions to implement once an economic activity is back on track and banks start lending again.

Coming from a risk background and having worked on several Risk scorecards over time, I would like to share some of my understanding of how Credit Risk scorecards will change and needs regular monitoring & re-calibration during COVID-19 and post COVID-19.

The article focuses on 5 important aspects of a Behavioural scorecard, which form the pillars underpinning calibration and development or even monitoring a scorecard model.

1. **Roll Rate analysis**: As many Central Banks of affected countries have given relief through a 3-month moratorium or what we call as ‘Payment Holidays’ to existing customers, the roll rates will likely see an increase and there will likely be reduced Roll Backwards rate. A 3- to 6-month Payment holiday, along with the economic stress that the bank will inevitably go through, means that Banks will need to be more conservative during and after COVID19. They would be using a lower delinquency bucket as a cut-off for default in their usual scorecards.

As can be seen from the below table, the Roll Forward rates might observe an increasing shift with more impact on the risky buckets while Roll Backward rates might have a decreasing shift.

![Roll Rates comparison of Pre and Post COVID scenarios](/images/scorecards/img1.png)
**The data take is dummy to show a real comparison analysis. It does not belong to any specific data source**

![Roll Forward and Roll Backward rates](/images/scorecards/img2.png)
**The charts show trends which some of the scorecards might see and these are my personal views**

Moreover, the risky customers in the bucket 30-60 & 60-90 could benefit from the Payment holidays/moratorium. Also, there will be an overestimation of risk for the high-risk bracket customers as some fresh and large inflow from other brackets might be observed which is a probability that customers will not pay after the Payment Holidays gets over. This could indeed create an overestimation for the actuals customers which belong to these brackets.

Since none of the models have faced a scenario such as COVID19, Markov Chain & Matrix Multiplication could be used to extrapolate the Roll Rates and manage default rate expectations. So, if you take the Roll rate numbers for the Post-COVID scenario, the transition matrix for the next vintage will be the following:

![Transition Matrix](/images/scorecards/img3.png)

The below accounts in respective buckets for future vintage are calculated with Matrix Multiplication of Rolling rates & Number of accounts in the previous vintage.
![Next vinatage calculations with Transition Matrix](/images/scorecards/img4.png)

2. **Vintage Analysis**: Vintage Analysis is useful to compare the risk among loans originated in different vintages (snapshots). It also helps in monitoring the risk of a portfolio and finding an optimal performance window for the development of a scorecard. We usually take vintages at a quarterly frequency and plot default rates per Month On Book (MOB).

![Vintage Analysis](/images/scorecards/img5.png)
During COVID-19 or Post COVID-19, it will be worthwhile to perform Vintage analysis for 30+ & 60+ days delinquent accounts too. The intent will be to find trends for early delinquencies due to the economic distress everywhere. With this change, it is worthwhile to also look and change the performance window of the scorecards.

**This can be used in monitoring and calibration of scorecards**

3. **Alternate Data**: The fact of the matter is, that when an unseen scenario occurs and your model hasn’t been trained on similar data, it can lead to more innovation on what other features could be developed to get that impact. This is where some alternate data related to the spending pattern/behavior of customers during and after the lockdown might be helpful. Some of the variables that could be used are

1.Information related to salary giving us the income loss indicator
1.Change in the purchasing behavior of customers pre- vs. post-lockdown (Average spend in Entertainment, online shopping, Food, etc.)
1.Balances running low
1.Cash transfers to other accounts
1.Debit from accounts
1.Liquidating Fixed or Recurring Deposits (FDs/RDs)
1.Change in location pre-post lockdown to help collection teams

Even an ML model could be run while using an integration of such alternate data to mitigate risk not only during COVID-19 but after COVID-19 too. Some of the use cases for the consumption of this alternate data could be:

1.If a customer’s spending behavior remains the same while their salary is not credited, it could be a potential risk to the bank. Such data could also help in the collection strategy.
1.If a customer’s balance is running low during and post-COVID-19 it is again a potential risk

If a customer has liquidated his RDs to pay for installments, there is a potential he/she will continue paying installments

4. **Credit Scores from probabilities/Odds Ratio**: In a scorecard, we get the Bank Risk scores from the probability of default (PD) using the key values as below.

Base Score: 600 Goods to Bads ratio: 50:1 Points to double odds (PDO): 20

Score = Offset – Factor ∗ ln (Odds) where

where Odds = PD/(1-PD), Factor = 20/ln(2) & Offset = Base Score - Factor * ln(Goods to Bad ratio)

Post COVID-19, behavioral scores will drop for the majority of customers (especially for high-risk customers) and the scores might flatten out too. This behavior is observed as the density in the middle could shift to left which means more customers ranging in median score will be distributed between the lowest score and median score. More range will make it more scattered on the left side.

This means that Banks will need to be more conservative in all their strategies and decisions. In the below graph, I have taken numbers to showcase my understanding of the distribution of Scores will change Post COVID-19. It clearly shows that the Scores will be shifting to the left side and will flatten as we enter into this economic distress.

![Behaviour Score comparison between Normal and Post COVID scenario](/images/scorecards/img6.png)

Also, the metrics like Base Score, Goods to Bads ratio, pdo shall be recalibrated so that scorecard does not predict a high score for a high-risk customer. Like the points to double the odds (pdo) is something which could be increased to re-calibrate scores as per COVID-19 changes. Also, as a trend scorecard will observe a decrease of Good to Bads ratio under different buckets especially in high score buckets.

5. **Monitoring**: Model monitoring will be an important exercise and will be in high demand. The usual monitoring cycle of 60-90 days, could be changed to 30 days to mitigate risk at an early stage. This will also help show how the model (new or recalibrated as per COVID-19) is performing compare to the actuals default rates.

1.The upper limits for Red Amber Green (RAG) metrics could also be changed as per the scorecard and could be made more conservative.
1.A risk manager can also see that default rates don’t rank-order across all the score ranges and could overlap.
1.There could be an addition of segments specific to COVID-19 behavior analysis to check for PSI and GINI variation

**Conclusion**: With such an unforeseen event and turbulent economic times ahead, calibration and regular and close monitoring of scorecards will be key to mitigating risks to credit exposures. The scorecards should be made to align with the post-COVID-19 reality, predicting some of the use cases. Also, risk managers need to consider refraining from aggressive strategic decisions on lending. These steps become more relevant because we still are unclear on when everything is going back to normal, when economic activity will normalize again and when people will start spending so that banks become aggressive in acquiring more customers to take loans.

*Disclaimer: These are my personal views and neither has anything to do with any other source nor am I offering any advice on specific decisions. The article is for information and must not be reproduced or redistributed.* 
