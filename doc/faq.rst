===
FAQ
===

Does LightFM have a GPU based implementation?
=============================================
 No, there is no option to run training or inference on the GPU with LightFM. There are
 currently no plans to change this.
- https://github.com/lyst/lightfm/issues/429

What are the "learning to rank" and "hybrid" aspects of LightFM and how do they relate?
=======================================================================================
*Learning to rank* and *hybrid* recommendation models are independent concepts. 
*Learning to rank* just means that you are optimizing a ranking loss such as `WARP` or 
`BPR`. *Hybrid* refers to the fact that you incorporate user or item 
meta-data as features additionally to your interactions.
See: https://github.com/lyst/lightfm/issues/442
