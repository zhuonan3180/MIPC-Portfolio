from statsmodels.stats.outliers_influence import variance_inflation_factor
#pvpfopt
import Model as model
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as smt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re
from datetime import date
from sklearn.model_selection import cross_val_predict
import math
from sklearn.metrics import mean_squared_error
m = model.MVPModel()

class ScenarioAnalysis(model.MVPModel):

    def __init__(self):
        model.MVPModel.__init__(self)
        self._init_data()

    def _init_data(self):
        self.hist_macro = self._get_hist_macro()
        self.price = m.data
        self.rf = m._get_rf_data(annulize_factor=12)
        self.scen = self._get_scenario_data()
        self.monthly_return = self._get_portfolio_data()
        self.monthly_premium = self._get_monthly_premium()
        self.monthly_fama = self._get_monthly_fama_factors()
        self.quarterly_fama = self._get_quarterly_fama_factors()
        self.macro_corr = self._get_macro_corr()
        self.fama_model = self._get_fama_model()
        self.fama_macromodel = self._get_macrofama_model()
        self.expfama = self._get_fama_forecast()
        self.expreturn = self._get_return_forecast()


    def _get_portfolio_data(self):
        daily_price = self.price
        df = daily_price.resample('1M').mean()
        df = df.pct_change()
        df.index = pd.to_datetime(df.index, format="%Y%m").to_period('M')
        return df

    def _get_monthly_premium(self):
        df = self.monthly_return
        rf_daily = self.rf
        rf_monthly = rf_daily.resample('1M').mean()
        rf_monthly.index = pd.to_datetime(rf_monthly.index, format="%Y%m").to_period('M')
        start_date = df.index[0]
        end_date = df.index[-1]
        rf = rf_monthly.loc[start_date:end_date]
        return df.subtract(rf, axis=0)

    def _get_monthly_fama_factors(self):
        df = pd.read_excel('Fama3.xlsx').set_index('Date')
        yr = [str(i)[:4] for i in df.index]
        m = [str(i)[4:] for i in df.index]
        l = [yr[i]+'-' +m[i] for i in range(len(yr))]
        df.index = pd.to_datetime(l, format="%Y-%m").to_period('M')
        return df

    def _get_quarterly_fama_factors(self):
        df = pd.read_excel('Fama3.xlsx').set_index('Date')
        yr = [str(i)[:4] for i in df.index]
        m = [str(i)[4:] for i in df.index]
        l = [yr[i] + '-' + m[i] for i in range(len(yr))]
        df.index = pd.to_datetime(l, format="%Y-%m").to_period('M')
        df = df.resample('Q').mean()
        return df

    def _get_scenario_data(self):
        df = pd.read_excel('Brexit_scen.xlsx').set_index('Scenario')
        return df


    def _get_hist_macro(self):
        table = ['house price index', 'import price index', 'labor productivity',  'production index', 'Real GDP', 'unemploy']
        type = '.xls'
        df = []
        dates = []
        dates_end = []
        for i in table:
            path = i+type
            temp = pd.read_excel(path).set_index('observation_date')
            temp.index = pd.to_datetime(temp.index)
            dates.append(temp.index[0])
            dates_end.append(temp.index[-1])
            temp = temp.rename(columns = {temp.columns[0]:i})
            df.append(temp)
        res = df[0]
        for i in df[1:]:
            res = i.join(res)
        start = max(dates)
        end = min(dates_end)
        res = res.loc[start:end]
        res = res.resample('Q').mean()
        return res

    def _get_macro_corr(self):
        macro = self.hist_macro
        macro.corr().to_excel('Macro_factor_correlation.xlsx')
        return macro.corr()

    def _get_fama_model(self):
        Fama = self.monthly_fama
        port_his = self.monthly_premium
        start = port_his.index[0]
        end = Fama.index[-1]
        FF_reg = pd.DataFrame(index=port_his.columns)
        p_v = pd.DataFrame(columns=port_his.columns)
        rhs = sm.add_constant(Fama.loc[start:end])
        for corp in FF_reg.index:
            lhs = port_his.loc[start:end][corp]
            res = sm.OLS(lhs, rhs, missing='drop').fit()
            FF_reg.loc[corp, 'alpha'] = res.params['const']
            FF_reg.loc[corp, 'beta_m'] = res.params['Mkt-RF']
            p_v[corp] = res.pvalues
            FF_reg.loc[corp,'beta_s'] = res.params['SMB']
            FF_reg.loc[corp,'beta_v'] = res.params['HML']
            # FF_reg.loc[corp,'R_sq'] = res.rsquared
            # FF_reg.loc[corp,'MSE'] = res.mse_model

        return FF_reg

    def _get_macrofama_model(self):
        Fama = self.quarterly_fama
        macro = self.hist_macro
        start = macro.index[0]
        end = macro.index[-1]
        FF_macros_reg_2 = pd.DataFrame(columns=Fama.columns[:3])
        rhs = sm.add_constant(macro)

        for factor in FF_macros_reg_2.columns:
            lhs = Fama.loc[start:end][factor]
            res = sm.OLS(lhs, rhs, missing='drop').fit()
            FF_macros_reg_2[factor] = res.params
            # FF_macros_reg_2.loc['r-squared',factor] = res.rsquared
        return FF_macros_reg_2

    def _get_fama_forecast(self):
        reg = self.fama_macromodel
        scen = pd.DataFrame(index = self.scen.index)
        scen['const'] = 1
        scen = scen.join(self.scen)
        res = scen @ reg
        return res

    def _get_return_forecast(self):
        exp_fama = pd.DataFrame(index = self.expfama.index)
        exp_fama['const'] = 1
        exp_fama = (exp_fama.join(self.expfama)).T
        reg = self.fama_model
        res = np.dot(reg, exp_fama)
        result = pd.DataFrame(data=res, columns=exp_fama.index[1:], index = reg.index)
        return result

    def _get_optimal_weight(self):
        df = pd.read_excel('Optimal Allocation.xlsx', index_col=0)
        amz = self.monthly_premium['AMZN'].mean()
        ups = self.monthly_premium['UPS'].mean()
        exp_return = self.expreturn.T
        exp_return['AMZN'] = amz
        exp_return['UPS'] = ups
        print(exp_return)
        print(df)
        res = np.dot(exp_return, df)
        result = 12*pd.DataFrame(data=res, columns=['return'], index=exp_return.index)
        return result


test = ScenarioAnalysis()
# print(test.monthly_premium.mean())
# model.hist_macro.to_excel('Macro.xlsx')
# print(test.monthly_fama)
# print(test.quarterly_fama)
# model.hist_macro.corr().to_excel('Corr_factors.xlsx')
# print(test.macro_corr)
# print(test.fama_model)
# print(test.hist_macro)
# print(test.quarterly_fama)
# print(test.fama_macromodel)
# print(test.expfama)
print(test._get_optimal_weight())