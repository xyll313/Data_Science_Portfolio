## Project Background 

An app, **KITTENGRAM** allows users to post pictures of their cats and the other users can like or comments the pictures. Only cats are allowed in the app.

We decide to use 2 KPIs to study if the new Ads structure does give an improvment in revenue:
- **Daily active users** (DAU): to make sure the users come back
- **Clight-through rate** (CTR): New type of sponsored posts vs. old type of sponsorded posts.

#### Define the Hypothesis
* **H0**: tailered ads have no effect on the user engagement - will not affect selected KPIs
* **H1**: tailered ads increase selected KPIs

## Results
### 1. Daily Active Users (DAU)

Figure 1 shows a significant increase in user activity in terms of DAU with good consistency (ensures our results are not affected by Novelty effect) over time since the test started. 

![image](https://user-images.githubusercontent.com/29717509/182466913-e6257d00-f0b3-435c-8647-2a9c4f848453.png)
Figure1

Also observed is a slight increase (ca.1000) increase in user activity for the conrol group between 2021-11-02 to2021-11-20. This pattern is not observed in the test group. Therefore we would like to analyse this further if data is avaialble.

![image](https://user-images.githubusercontent.com/29717509/182467403-02df0a40-4278-4731-987f-69db545aae36.png)
Figure2

In addition,the statistics shown in Figure 2  suggest that the test is successful on user activity in terms of **daily active user(DAU)** by a much improved mean (15782 -> 29302) and median (15990 -> 29300). Nonetheless, we would like to have a look of **user retention** if we have access to this data.

We also compared the before and after test P values. P-value measures *the probability of obtaining the observed results providing the null hypothesis is true*. A before-test P value at 0.16 suggests there is no or negligible **pre-test bias** between the groups. An after test P value very close to zero (0.00) suggests a very high statistical significance, enable us to reject H0.

### 2. Click Through Rate (CTR)
Similar to DAU, we see a significant increase with good consistency  in CTR since the test started (2021-11-01).

![image](https://user-images.githubusercontent.com/29717509/182467442-67f924f7-f999-424f-af64-cd800d42a8e2.png)
Figure 3

The after test mean CTRs  shows an increase of ** 5% ** . Again, the before test P values at 0.75 suggest no or negligible pre-test bias; after teststart P values at 0.00 indicates a very high statistical significance, enable us to reject H0.

## Conclusions
The statistical significance of the AB tests enabled us to reject H0. Therefore we can conclude the tailored ads do improve desired KPIs (daily active users and click through rate)

