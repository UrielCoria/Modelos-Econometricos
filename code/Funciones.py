def Cuadro_Dickey_Fuller(data):
    from statsmodels.tsa.stattools import adfuller
    from pandas import DataFrame
    types_ADF = ['n','c','ct','ctt']

    return DataFrame(
        [[adfuller(data[item], regression=t)[1] for t in types_ADF ] for item in data.columns ],
        index=data.columns, columns=types_ADF)


def select_order_akaike(data, AR=1, MA=0, Exog=None, Trend='n'):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        import numpy as np
        from pandas import DataFrame
        warnings.filterwarnings('ignore')       # Elimina las advertencias de código
        
        pmax = AR
        qmax = MA
        P = np.arange(pmax+1)
        Q = np.arange(qmax+1)
        plabels = [f'p={p}' for p in P]
        qlabels = [f'q={q}' for q in Q]

        AIC = DataFrame(
        [[SARIMAX(data, order=[p,0,q], exog=Exog, trend=Trend).fit().aic for q in Q ] for p in P ],
        index=plabels, columns=qlabels)

        return AIC.style.highlight_min(axis=None)


def select_order_bayes(data, AR=1, MA=0, Exog=None, Trend='n'):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        import numpy as np
        from pandas import DataFrame
        warnings.filterwarnings('ignore')       # Elimina las advertencias de código
        
        pmax = AR
        qmax = MA
        P = np.arange(pmax+1)
        Q = np.arange(qmax+1)
        plabels = [f'p={p}' for p in P]
        qlabels = [f'q={q}' for q in Q]

        BIC = DataFrame(
        [[SARIMAX(data, order=[p,0,q], exog=Exog, trend=Trend).fit().bic for q in Q ] for p in P ],
        index=plabels, columns=qlabels)

        return BIC.style.highlight_min(axis=None)


def OLS_residual_diagnostic(Model, Lags=None):
    import pandas as pd
    from statsmodels.stats.diagnostic import het_arch
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.diagnostic import het_goldfeldquandt
    from statsmodels.stats.diagnostic import het_white
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey

    tests = ['Acorr Breusch-Godfrey', 'Het Breusch-Pagan-Godfrey', 'Het Goldfeld-Quandt', 'Het White']
    values = ['Valor Lagrange', 'Valor-P Lagrange', 'Valor F', 'Valor-P F']

    data = [
                list(acorr_breusch_godfrey(Model, nlags=Lags)),
                list(het_breuschpagan(Model.resid, Model.model.exog)),
                ['-', '-', het_goldfeldquandt(Model.resid, Model.model.exog)[0], het_goldfeldquandt(Model.resid, Model.model.exog)[1]],
                list(het_white(Model.resid, Model.model.exog))
            ]

    return pd.DataFrame(index=tests, columns=values, data=data)