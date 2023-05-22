default_paras={'pKa_H2PO4':7.21, 'pKa_TrisH': 8.07,'pKa_NH4':9.25,
    'HPO4':0,'H2PO4':0,'Tris':0,'NH4':0,'pH':None,
    'pKa_CO2_1':6.351,'pKa_CO2_2':10.329}

def cal_HCl_added(return_paras=False, **kwargs): # correct_CO2=False
    """ Calculate how much HCl is added to the media given chemical concentrations and pH. 

    Valid parameters are: 
        sol_C=[True,False]
        P_mix (molar)
        H2PO4/HPO4 (molar): addtional phosphate besides P_mix.
        Tris/NH4 (molar)
        pH: can be an array. 
    
    Additional options:
        # (This doesn't work in reality. Ignore.) correct_CO2=[True,False]: consider amphospheric CO2's effect on pH.
        return_paras: also return the parameter dictionary.
    """
    paras=get_paras(**kwargs)
    H=10.0**(-paras['pH'])
    k_H2O=1E-14
    k_H2PO4=10**(-paras['pKa_H2PO4'])
    k_TrisH=10**(-paras['pKa_TrisH'])
    k_NH4=10**(-paras['pKa_NH4'])

    out=(H/(k_H2PO4+H)*paras['HPO4'])-(k_H2PO4/(H+k_H2PO4)*paras['H2PO4'])+(H/(k_TrisH+H)*paras['Tris'])-(k_NH4/(H+k_NH4)*paras['NH4'])+H-(k_H2O/H)

    # This doesn't work in reality. Ignore. 
    # if correct_CO2:
    #     if correct_CO2=='default':
    #         correct_CO2=0.04
    #     K1=10**(-paras['pKa_CO2_1'])
    #     K2=10**(-paras['pKa_CO2_2'])
    #     correct=(2*K2/H+1)*K1/H*(3.44E-2*1E-2*correct_CO2)
    #     out-=correct
    
    if return_paras:
        out=(out,paras)
    return out


def cal_pH(HCl, **kwargs):
    """ Solve pH given the HCl amount and media condition. This is the reverse problem. 
    
    Note that sometimes the solver can't find the solution. If you see a warming message if maximum iteration reached, this is why.

    """
    from scipy.optimize import fsolve
    try:
        return [cal_pH(hcl) for hcl in HCl]
    except:      
        f=lambda pH: cal_HCl_added(pH=pH,**{key:value for key, value in kwargs.items() if key!='pH'})-HCl
        return fsolve(f,7.0)

def get_paras(**kwargs):
    """ Return a dictionary of media parameters. 
    
    Valid keys are: 
        sol_C=[True,False]
        P_mix (molar)
        H2PO4/HPO4 (molar): addtional phosphate besides P_mix.
        Tris/NH4 (molar)
        pH: usually passed as a varible. 
    """
    paras=default_paras.copy()
    for key, value in kwargs.items():
        if key=='sol_C':
            if value: # binary
                paras['H2PO4']+=1E-4 # total 0.5mL solution C per 500mL 1/2x Taub; solution C has is 0.1M H2PO4
        elif key=='P_mix': # unit: molar
            paras['H2PO4']+=value/2
            paras['HPO4']+=value/2
        elif key in ['pH']:
            paras[key]=value
        elif key in ['HPO4','H2PO4','Tris','NH4']:
            paras[key]+=value
        elif key not in ['C']:
            raise NotImplementedError(f"{key}={value}")
    return paras

def parse_cond(s):
    ''' Parse a media condition string to a dictionary. 
    
    The unit in the string is mM and the unit in the dictionary is M. 
    
    Example: "sol_C=False;Tris=5;C=10;NH4=0.85;P_mix=4" => {'sol_C': False, 'Tris': 5E-3, 'C':1E-2, 'NH4': 8.5E-4, 'P_mix':4E-3}
    '''
    paras={}
    for cond in s.split(";"):
        k,v=cond.split('=')
        if k=='sol_C':
            if v=='False':
                v=False
            else:
                v=True
        else:
            if v=='False':
                v=0
            else:
                v=float(v)*1E-3
        paras[k]=v
    return paras


def predict_initial_pH(media_cond,linear_model_correction=True, Tris_stock_conc_M=None,Tris_stock_pH=None):
    """ Predict the initial media pH made from a combination of these stock solutions: 1/2XTaub-SolC, sol_C, Tris stock, P_mix, NH4, C. 

    Because in actually experiments Chandana only uses media made from these stock solutions, only these keywords are supported, altough expanding the allowed keywords is possible. 

    Paras:
        media_cond: media condition in a string or a dictionary. Allow these keys: sol_C (True/False), Tris, P_mix, NH4, C. See parse_cond for an example format. 
        linear_model_correction: whether to correct the prediction by the linear model. 
        Tris_stock_conc_M, Tris_stock_pH: Tris stock is made by titrating Tris solution with HCl to a desired pH. The amount of HCl added here afftects the prediction. If media_cond['Tris'] is not 0, these two parameters are required. 
    
    """
    
    # calculate pre-added HCl/NaOH
    if isinstance(media_cond,str):
        media_cond=parse_cond(media_cond)
    HCl_pre=[0,-8E-5][int(media_cond['sol_C'])] # pre added NaOH from solution 
    if media_cond['Tris']>0:
        if Tris_stock_conc_M is None or Tris_stock_pH is None:
            raise ValueError("When Tris is added, specify Tris_stock_conc_M and Tris_stock_pH.") 
        else: 
            HCl_conc_in_stock=cal_HCl_added(Tris=Tris_stock_conc_M,pH=Tris_stock_pH)
            HCl_pre+=HCl_conc_in_stock*media_cond['Tris']/Tris_stock_conc_M

    pH_pred=cal_pH(HCl_pre,**media_cond)[0]
    if linear_model_correction:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_validate,ShuffleSplit

        data=pd.read_csv("20220428_titration_data_for_initial_pH_prediction.csv",index_col=0)
        X=data['pred_initial_pH'].values.reshape(-1,1)
        y=data['initial_pH'].values
        linear_model=LinearRegression().fit(X,y)
        pH_pred=linear_model.predict([[pH_pred]])[0]
    return pH_pred
