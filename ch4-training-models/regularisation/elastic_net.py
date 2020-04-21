from sklearn.linear_model import ElasticNet


"""
    @brief Do elastic net regression, which uses a mix
    of regularisation terms from ridge and lasso,
    according to a mix ratio r.

    NOTE: r = 0 indicates Ridge Regression,
    whereas r = 1 indicates Lasso Regression.

    @param X data array of feature values.
    @param y data array of label values.
    @param r floating point mix ratio.
    @return Fitted elastic net.
"""
def do_elastic_net(X, y, r):
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=r)
    elastic_net.fit(X, y)
    return elastic_net