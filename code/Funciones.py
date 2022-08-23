def Cuadro_Dickey_Fuller(data):
    n = []
    c = []
    ct = []
    ctt = []

    types_ADF = ['n','c','ct','ctt']
    from statsmodels.tsa.stattools import adfuller

    for t in types_ADF:
        for item in data.columns:
            if t == 'n':
                n.append(adfuller(data[item], regression=t)[1])
            elif t == 'c':
                c.append(adfuller(data[item], regression=t)[1])
            elif t == 'ct':
                ct.append(adfuller(data[item], regression=t)[1])
            elif t == 'ctt':
                ctt.append(adfuller(data[item], regression=t)[1])

    from pandas import DataFrame

    return DataFrame(columns=types_ADF, index=data.columns, data=list(zip(n,c,ct,ctt)))


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