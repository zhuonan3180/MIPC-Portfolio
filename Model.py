from pandas_datareader import data
import pandas as pd
import numpy as np
from dateutil.parser import parse

class MVPModel:

    def __init__(self):
        self._init_data()

    def _init_data(self):
        """Initiate pre-cooked tables to ease the algo process """
        self.tickers = ['AMZN','AAL.L','ABF.L','AZN.L','BBY.L','BP','BT-A.L','CPG.L','DMGT.L','DGEAF','GFRD.L','INF.L','KIE.L','LGEN.L','NG.L','TSCO.L','ULVR.L','UPS','VOD.L','OCT.L',
'PGR.L',
'RKH.L',
'GHT.L']
        # 'FGP.L','SRP.L'  two railway companies
        self.rf = self._get_rf_data(annulize_factor=365)
        self.target_return = 0
        self.volatility = 0
        self.data = self._get_data(self.tickers)
        self.returndata = self._get_return_data(self.data)

    def _get_data(self, tickers:list):
        """Pull data from yahoo finance by ticker
        Param: List of tickers to be pulled
        Output: DataFrame of daily price of each ticker """
        Datas = {}
        Dates = []
        df = pd.DataFrame()
        for i in tickers:
            DATA = data.DataReader(i, 'yahoo', '1980-01-01')
            Datas[i] = (DATA['Adj Close'])
            Dates.append(DATA.index[0])

        st_date = max(Dates)

        for tick, d in Datas.items():
            temp = d.loc[st_date:]
            df[tick] = temp

        return df

    def _get_rf_data(self, annulize_factor=365):
        """Pull data from local excel to get risk free rate
                Output: DataFrame of daily return of 30 year UK gilt """
        df = (pd.read_excel('UK30yr.xlsx').set_index('Date'))[::-1]
        temp = [parse(i) for i in df.index]
        df.index = temp
        df = ((((1+(df / 100))**(1/annulize_factor))-1)['Price'])[1:]
        return df

    def _get_return_data(self, df):
        """Get daily return by ticker
            Param: DataFrame containing Daily price of each ticker
            Output: DataFrame of daily return of each ticker """
        # df.pct_change(freq='M')[1:]
        return df.pct_change()[1:]

    def _get_dftilda(self,df):
        """Get daily risk premium by ticker by subtracting the risk free rate
            Param: DataFrame containing Daily return of each ticker
            Output: DataFrame of daily risk premium of each ticker """
        rf = self.rf
        start_date = df.index[0]
        rf = rf.loc[start_date:]
        return df.subtract(rf, axis=0)

    def _get_corr_matrix(self, df):
        """Get correlation matrix
            Param: DataFrame containing Daily return of each ticker
            Output: DataFrame of correlations of each ticker """
        return df.corr()

    def _compute_tangency(self, df_tilde, diagonalize_Sigma=False):
        """Compute tangency portfolio given a set of excess returns.

        Also, for convenience, this returns the associated vector of average
        returns and the variance-covariance matrix.

        Parameters
        ----------
        diagonalize_Sigma: bool
            When `True`, set the off diagonal elements of the variance-covariance
            matrix to zero.
        """
        Sigma = df_tilde.cov()
        # N is the number of assets
        N = Sigma.shape[0]
        Sigma_adj = Sigma.copy()
        if diagonalize_Sigma:
            Sigma_adj.loc[:, :] = np.diag(np.diag(Sigma_adj))

        mu_tilde = df_tilde.mean()
        Sigma_inv = np.linalg.inv(Sigma_adj)
        weights = Sigma_inv @ mu_tilde / (np.ones(N) @ Sigma_inv @ mu_tilde)
        omega_tangency = pd.Series(weights, index=mu_tilde.index)
        return omega_tangency, mu_tilde, Sigma

    ## Compute the mean, volatility, and Sharpe ratio for the tangency portfolio

    def portfolio_stats(self, omega, mu_tilde, Sigma, annualize_factor=252):
        """Get daily return by ticker
            Param: weights for assets, mean of risk premium, covariance matrix of risk premium, annualize factor = 252 or 12
            Output: DataFrame of risky assets fraction, annual mean return, annual volatility, sharpe ratio """
        mean = annualize_factor * mu_tilde @ omega
        vol = np.sqrt(annualize_factor) * np.sqrt(omega @ Sigma @ omega)
        sharpe_ratio = mean / vol
        upper = mean + 1.96 * vol/np.sqrt(10)
        lower = mean - 1.96 * vol / np.sqrt(10)
        interval = (lower, upper)
        df_stats = pd.DataFrame([omega.sum(), mean, interval, vol, sharpe_ratio],
                                index=['Fraction in Risky Asset', 'Mean Return', 'Return Confidence Interval', 'Volatlity', 'Sharpe Ratio'],
                                columns=['Stat'])
        return df_stats

    def target_mv_portfolio(self, df_tilde, target_return=0.06/252, diagonalize_Sigma=False):
        """Compute MV optimal portfolio, given target return and set of excess returns.

        Parameters
        ----------
        diagonalize_Sigma: bool
            When `True`, set the off diagonal elements of the variance-covariance
            matrix to zero.
        """
        omega_tangency, mu_tilde, Sigma = self._compute_tangency(df_tilde, diagonalize_Sigma=False)
        Sigma_adj = Sigma.copy()
        if diagonalize_Sigma:
            Sigma_adj.loc[:, :] = np.diag(np.diag(Sigma_adj))
        Sigma_inv = np.linalg.inv(Sigma_adj)
        N = Sigma_adj.shape[0]
        delta_tilde = ((np.ones(N) @ Sigma_inv @ mu_tilde) / (mu_tilde @ Sigma_inv @ mu_tilde)) * target_return
        omega_star = delta_tilde * omega_tangency
        return omega_star




model = MVPModel()

Daily_return = model.returndata
Correlation = model._get_corr_matrix(Daily_return)
Annualized_returns = 252*Daily_return.mean()
Risk_premiums = model._get_dftilda(Daily_return)
omega_tangency, mu_tilde, Sigma = model._compute_tangency(Risk_premiums)
tangency_portfolio_stats = model.portfolio_stats(omega_tangency, mu_tilde, Sigma)
omega_star = model.target_mv_portfolio(Risk_premiums, target_return=0.25/252)
Result = model.portfolio_stats(omega_star, mu_tilde, Sigma)

Daily_return.to_excel('Monthly_Return.xlsx')
print(Daily_return)
print(Correlation)
Correlation.to_excel('Correlation_nostartups.xlsx')
print('Annualized Return For Risky Assets are: ')
print(Annualized_returns)
Annualized_returns.to_excel('Annualized_daily_return.xlsx')
print('###################################################')
print(tangency_portfolio_stats)
print('###################################################')
print('Optimal Weights Allocation In Risky Assets Is: ')
print(omega_star)
omega_star.to_excel('Optimal Allocation.xlsx')
print('###################################################')
print('MV Portfolio Statistics Results: ')
print(Result)
Result.to_excel('Portfolio Performance.xlsx')
print('###################################################')

