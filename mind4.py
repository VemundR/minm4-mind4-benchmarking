# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---

# Necessary packages.
import numpy as np
from numpy.linalg import *
import pandas as pd
import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('always', UserWarning) 
np.set_printoptions(suppress=True)

def mind4(mnr, rea, liste_d4, basisaar, startaar):
    res_dict = {}
    
    if type(liste_d4) == str:
        liste_d4 = [liste_d4]

    # CHECKS.
    # Checking object type.
    assert type(liste_d4) == list or type(liste_d4) == str, 'You need to create a list of all the series you wish to benchmark, and it must be in the form of a list object or string.'
    
    # Checking the monthly DF.
    assert isinstance(mnr, pd.DataFrame), 'The monthly dataframe is not a DataFrame.' 
    assert isinstance(mnr.index, pd.PeriodIndex), f'The monthly dataframe does not have a pd.PeriodIndex.'

    mnr_of_concern =  mnr[mnr.columns[mnr.columns.isin(liste_d4)]] # Filters out series not sent to benchmarking.
    mnr_of_concern = mnr_of_concern[(mnr_of_concern.index.year <= basisaar) & (mnr_of_concern.index.year >= startaar)]
    
    assert pd.Series(liste_d4).isin(mnr.columns).all(), f'{np.setdiff1d(liste_d4, mnr.columns)} are missing in the monthly dataframe.'
    if mnr_of_concern.isna().any().any() is np.True_:
        warnings.warn(f'There are NaN-values in {mnr_of_concern.columns[mnr_of_concern.isna().any()].to_list()} in the monthly dataframe.', UserWarning, stacklevel=2)
    if (mnr_of_concern == 0).any().any() is np.True_:
        warnings.warn(f'There are only zeroes in {mnr_of_concern.columns[(mnr_of_concern == 0).any()].to_list()} in the monthly dataframe.', UserWarning, stacklevel=2)

    # Checking the yearly DF.
    assert isinstance(rea, pd.DataFrame), 'The yearly dataframe is not a DataFrame.'  
    assert isinstance(rea.index, pd.PeriodIndex), 'The yearly dataframe does not have a pd.PeriodIndex.'

    rea_of_concern =  rea[rea.columns[rea.columns.isin(liste_d4)]] # Filters out series not sent to benchmarking.
    rea_of_concern = rea_of_concern[(rea_of_concern.index.year <= basisaar) & (rea_of_concern.index.year >= startaar)]
    
    assert pd.Series(liste_d4).isin(rea.columns).all(), f'{np.setdiff1d(liste_d4, rea.columns)} are missing in the monthly dataframe.'
    if rea_of_concern.isna().any().any() is np.True_:
        warnings.warn(f'There are NaN-values in {rea_of_concern.columns[rea_of_concern.isna().any()].to_list()} in the yearly dataframe.', UserWarning, stacklevel=2)
    if (rea_of_concern == 0).any().any() is np.True_:
        warnings.warn(f'There are only zeroes in {rea_of_concern.columns[(rea_of_concern == 0).any()].to_list()} in the yearly dataframe.', UserWarning, stacklevel=2)


    assert set(mnr_of_concern.index.year.unique()).difference(set(rea_of_concern.index.year)) == set(), f'There isnt values in both series for {set(mnr_of_concern.index.year.unique()).difference(set(rea_of_concern.index.year))}.'    
    
    assert isinstance(startaar, int), 'The start year must be an integer.' 
    assert isinstance(basisaar, int), 'The final year must be an integer.'
    assert basisaar < 2050, 'The final year must be less than 2050. Are you sure you entered it correctly?.'
    assert basisaar >= startaar, 'The start year cannot be greater than the final year.'

    
    print("The inputdata passed the checks.\n")
    # CHECKS DONE.
    
    # Preperation of input data.
    
    # Scaling of the leading and following values.
    avs = rea_of_concern/mnr_of_concern.resample('Y').sum()
    
    for elem in liste_d4:
        print(f'Benchmarking {elem} with MinD4 from {startaar} to {basisaar}.')

        # Laster maneds- og aarstall
        datam_ = mnr_of_concern[elem].values
        datay_ = rea_of_concern[elem].values

   
        # Different scaling factors for the leading and following values to ensure the volume figure is correct in the start year.
        avstemming1 = avs[avs.index.year==startaar][elem].values
        avstemming2 = avs[avs.index.year==basisaar][elem].values
        
        # The leading value.
        datamf = mnr[(mnr.index.year == startaar) & (mnr.index.month == 1)][elem].values

        # Scaling of the leading value.
        datayf = avstemming1*datamf

        # The following value.
        datame = mnr[(mnr.index.year == basisaar) & (mnr.index.month == 12)][elem].values

        # Scaling of the following value.
        dataye = avstemming2*datame
        
        datam=np.hstack((datamf,datam_,datame)) 
        datay=np.hstack((datayf,datay_,dataye)) 

        # Counting months and years.
        nm=datam.shape[0]
        ny=datay.shape[0]

        # Setting up submatrices a for A and -vectors x for X consisting of zeros.
        a1=np.zeros((nm,nm),dtype=np.float64)
        a2=np.zeros((nm,ny),dtype=np.float64)
        a3=np.zeros((ny,nm),dtype=np.float64)
        a4=np.zeros((ny,ny),dtype=np.float64)
        x1=np.zeros((nm,1 ),dtype=np.float64)
        x2=np.zeros((ny,1 ),dtype=np.float64)

        try:
            # Fills in submatrices a into A and -vectors x into X according to Skjæveland 1985, pages 18 and 19 (modified to benchmark by month instead of quarter)
            for i in range(0,nm):
                a1[i,i]=(1+1*(i>0 and i<nm-1))/datam[i]**2
                if i>0:
                    a1[i,i-1]=-1/(datam[i]*datam[i-1])
                if i<nm-1:
                    a1[i,i+1]=-1/(datam[i]*datam[i+1])

            for i in range(0,nm):
                for j in range(0,ny):
                    if (i-1)/12>=j-1 and (i-1)/12<j:
                        a2[i,j]=1
                        a3[j,i]=1
                    x2[j]=datay[j]

            # Combines submatrices a into A and -vectors x into X according to Skjæveland 1985, pages 18 and 19 (modified to reconcile by month instead of quarter)
            A=np.vstack((np.hstack((a1,a2)),np.hstack((a3,a4))))
            X=np.vstack((x1,x2))

            # Solves the equation system AY=X for Y if possible. Otherwise, returns null.
            Y=np.linalg.solve(A,X)
        except:
            Y=np.zeros((nm,1),dtype=np.float64) 

        
        res_dict[elem] = Y[1:nm-1].flatten()

    res = pd.DataFrame(res_dict, index = mnr_of_concern.index)
    
    # Removing resulting NaN series from the res-DataFrame.
    res = res[res.columns[~res.isna().any()]]
    
    # Checks that the deviation on the totals after benchmarking is zero for the series in liste_d4.
    # Skips the series that already trigger error messages in the input.
    skippe = mnr_of_concern.columns[mnr_of_concern.isna().any()].to_list() + mnr_of_concern.columns[(mnr_of_concern == 0).any()].to_list() + rea_of_concern.columns[rea_of_concern.isna().any()].to_list() + rea_of_concern.columns[(rea_of_concern == 0).any()].to_list()
    for elem in set(liste_d4) - set(skippe):
        if not (((res.resample('Y').sum()-rea_of_concern) >= -1) & ((res.resample('Y').sum()-rea_of_concern) <= 1)).all()[elem] is np.True_:
            warnings.warn(f'There are deviations on the benchmarked totals in {elem} so something did not go well.', UserWarning, stacklevel=2)

    print(f'\nBenchmarking with MinD4 done!')
    
    return res
