===
FAQ
===

Does LightFM have a GPU-based implementation?
=============================================
No, there is no option to run training or inference on the GPU with LightFM. There are
currently no plans to change this.
See https://github.com/lyst/lightfm/issues/429

What are the "learning to rank" and "hybrid" aspects of LightFM and how do they relate?
=======================================================================================
*Learning to rank* and *hybrid* recommendation models are independent concepts. 
*Learning to rank* just means that you are optimizing a ranking loss such as `WARP` or 
`BPR`. *Hybrid* refers to the fact that you incorporate user or item meta-data as additional features.
See: https://github.com/lyst/lightfm/issues/442

Adding user/item features makes my model perform worse than without features, what can I do?
============================================================================================
That's not unusual and might have various reasons. For one, make sure you 
don't drop per-user/item features, see the notes in :doc:`LightFM<lightfm>`. If that
doesn't help, your features might be simply uninformative and worsen the 
signal to noise ratio. You can experiment with different features and try
discretization strategies for continuous features. More strategies and ideas
can be found here:

- https://github.com/lyst/lightfm/issues/551
- https://github.com/lyst/lightfm/issues/486
- https://github.com/lyst/lightfm/issues/176
- https://github.com/lyst/lightfm/issues/430

My model is recommending the same popular items to all users, what can I do?
============================================================================
You can try to set your item bias vectors to all zeros. Another strategy is 
to apply inverse propensity weights to your features.
See these issues for more information:

- https://github.com/lyst/lightfm/issues/395
- https://github.com/lyst/lightfm/issues/176

How can I re-train my model on partial data and/or new users (user cold-start)?
===============================================================================
This depends a lot on your specific use case. Here are some helpful discussions:

- https://github.com/lyst/lightfm/issues/194
- https://github.com/lyst/lightfm/issues/347
- https://github.com/lyst/lightfm/issues/210
- https://github.com/lyst/lightfm/issues/371
- https://stackoverflow.com/questions/46924119/lightfm-handling-user-and-item-cold-start
