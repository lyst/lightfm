===================
Helpful Discussions
===================

Adding user/item features makes my model perform worse than without features
============================================================================
That's not unusual and might have different reasons. For example, your features
might be simply uninformative and worsen signal to noise ratio.

- https://github.com/lyst/lightfm/issues/551
- https://github.com/lyst/lightfm/issues/176
- https://github.com/lyst/lightfm/issues/430

Inverse Frequency/Propensity Weighting
======================================
- https://github.com/lyst/lightfm/issues/176

Iterative Training / New Users / Fold-In
========================================
- https://github.com/lyst/lightfm/issues/194

Hyperparameter Optimization
===========================
-https://gist.github.com/maciejkula/29aaf2db2efee5775a7f14dc387f0c0f

Cold-Start
==========
- https://stackoverflow.com/questions/46924119/lightfm-handling-user-and-item-cold-start
- https://github.com/lyst/lightfm/issues/371

Time Context Information
========================
- https://github.com/lyst/lightfm/issues/237

Memory Limits / Memmapped arrays 
================================
- https://github.com/lyst/lightfm/issues/389

Predict for all users / Batch Prediction / BLAS Predict
=======================================================
- https://github.com/lyst/lightfm/issues/427
- https://github.com/inpefess/lightfm/blob/predict_comparison/examples/batch_predict/predicting_with_matrix_multiplication.ipynb

Large Matrices / Helper Tools for Data Prep
===========================================
- https://github.com/lyst/lightfm/issues/258
- https://github.com/DomainGroupOSS/ml-recsys-tools/blob/df866704b1ab3ad05dbe40ebd4b2d92d071f8eab/ml_recsys_tools/data_handlers/interactions_with_features.py#L154


Short-Lived Items?
==================
- https://github.com/lyst/lightfm/issues/449

