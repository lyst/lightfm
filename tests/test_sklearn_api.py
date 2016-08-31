from lightfm.sklearn_compat import SKLearnLightFM

def test_sklearn_api():
    model = SKLearnLightFM()
    params = model.get_params()
    model.set_params(**params)
