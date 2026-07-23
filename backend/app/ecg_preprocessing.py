import numpy as np
import scipy
import heartpy
import pandas as pd

from scipy import stats

def TINN(x:np.array):
  """ Compute all the triangular interpolation to calculate the TINN scores. It also computes HRV index from an array x which contains 
      all the interbeats times for a given ECG signal.

      The axis is divided in 2 parts respectively on the right and left of the abscissa of the maximum value of the gaussian distribution
      The TINN score calculation is defined in the WESAD Dataset paper, to calculate it we needthe closest triangular interpolation 
      of the gaussian distribution of the interbeats times. The triangular interpolation is defined by 2 lines that meet at the maximum value
      of the gaussian distribution and cross the x-axis in N on the first half of the x-axis and M on the second half of the x-axis. 
      Thus inside ]N;M[ the interpolation function != 0
      Outside of ]N;M[ the interpolation function equals 0.
  """

  kernel = stats.gaussian_kde(x) #Create an approximated kernel for gaussian distribution from the x array (interbeats times)
  absi=np.linspace(np.min(x),np.max(x),len(x)) # Compute the x-axis of the interbeats distribution (from minimum interbeat time to maximum interbeat time)
  val=kernel.evaluate(absi) # Fit the gaussian distribution to the created x-axis
  ecart=absi[1]-absi[0] # Space between 2 values on the axis
  maxind=np.argmax(val) # Select the index for which the gaussian distribution (val array) is maximum 
  max_pos=absi[maxind]  # Interbeat time (abscissa) for which the gaussian distribution is maximum
  maxvalue=np.amax(val) # Max of the gaussian distribution
  N_abs=absi[0:maxind+1] # First half of the x-axis
  M_abs=absi[maxind:] # Second half of the x-axis
  HRVindex=len(x)/maxvalue
  err_N=[]
  err_M=[]

  for i in range(0,len(N_abs)-1):
    N=N_abs[i]
    slope=(maxvalue)/(max_pos-N)
    D=val[0:maxind+1]
    q=np.clip(slope*ecart*np.arange(-i,-i+maxind+1),0,None) #Triangular interpolation on the First half of the x-axis
    diff=D-q 
    err=np.multiply(diff,diff)
    err1=np.delete(err,-1)
    err2=np.delete(err, 0)
    errint=(err1+err2)/2
    errtot=np.linalg.norm(errint) # Error area between the triangular interpolation and the gaussian distribution on the first half of the x-axis
    err_N.append((errtot,N,N_abs,q))
  
  for i in range(1,len(M_abs)):
    M=M_abs[i]
    slope=(maxvalue)/(max_pos-M)
    D=val[maxind:]
    q=np.clip(slope*ecart*np.arange(-i,len(D)-i),0,None) #Triangular interpolation on the second half of the x-axis
    diff=D-q
    err=np.multiply(diff,diff)
    err1=np.delete(err,-1)
    err2=np.delete(err, 0)
    errint=(err1+err2)/2
    errtot=np.linalg.norm(errint) # Error area between the triangular interpolation and the gaussian distribution on the second half of the x-axis
    err_M.append((errtot,M,M_abs,q))

  return (err_N,err_M,absi,val,HRVindex)

def best_TINN(x:np.array):
  """Select the best N and M that give the best triangular interpolation function approximation of the gaussian distrbution and return
    N; M; the TINN score = M-N ; and the HRV index
  
  """
  err_N,err_M,_,_,HRVindex=TINN(x)
  N=np.argmin(np.array(err_N,dtype=object)[:,0])
  M=np.argmin(np.array(err_M,dtype=object)[:,0])
  absN=err_N[N][1]
  absM=err_M[M][1]
  return float(absN),float(absM),float(absM-absN),HRVindex

def num_compare_NN50(x,i):
  """Count the number of HRV intervals differing more than 50 ms for a given HRV interval x[i]
  
  """
  ref=x[i]
  k=0
  diff=np.absolute(x-ref)
  k+=np.sum(np.where(diff>0.05,1,0))
  return k 

def compare_NN50(x):
  """ Returns the number and percentage of HRV intervals differing more than 50ms for all intervals
  
  """
  k=0
  for i in range(0,len(x)):
    k+=num_compare_NN50(x,i)
  if k==0:
    k=1
  return k,(k/(len(x)*len(x)))

def get_freq_features_ecg(x):
  """ Returns frequential features of the Heart Rate Variability signal (interbeats times) by computing FFT, to compute the Fouriers 
  Frequencies the mean of the Heart Rate variability is used as sampling period  
  """
  mean=np.mean(x)
  yf=np.array(scipy.fft.fft(x-mean))
  xf=scipy.fft.fftfreq(len(x),mean)[0:len(x)//2]
  psd=(2/len(yf))*np.abs(yf)[0:len(x)//2]
  fmean=np.mean(xf)
  fstd=np.std(xf)
  sumpsd=np.sum(psd)
  return fmean,fstd,sumpsd

def get_data_ecg(x):
  """ Collect the features of a given ECG signal x, using HeartPy package to compute the peak list (not the previous developed peak 
  detection function).  
  """
  working,mes=heartpy.process(x,700)
  peak=working["peaklist"]
  periods=np.array([(peak[i+1]-peak[i])/700 for i in range(0,len(peak)-1)])
  frequency=1/periods
  meanfreq = np.mean(frequency)
  stdfreq = np.std(frequency)
  HRV=np.array([(peak[i]-peak[i-1])/700 for i in range(1,len(peak))])
  _,_,T,HRVindex=best_TINN(HRV)
  num50,p50=compare_NN50(HRV)
  meanHRV=np.mean(HRV)
  stdHRV=np.std(HRV)
  rmsHRV=np.sqrt(np.mean(HRV**2))
  fmean,fstd,sumpsd=get_freq_features_ecg(HRV)
  return np.array([meanfreq,stdfreq,T,HRVindex,num50,p50,meanHRV,stdHRV,rmsHRV,fmean,fstd,sumpsd])

def extract_baseline_features(ecg_signal):
    return get_data_ecg(ecg_signal)

def normalize_features(features, baseline_features):

    result = np.divide(features, baseline_features)

    if np.isinf(result).any() or np.isnan(result).any():
        raise ValueError("Invalid normalized feature vector")

    return result

def load_ecg_csv(csv_path, column_name=None):

    df = pd.read_csv(csv_path)

    if column_name is None:
        column_name = df.columns[0]

    ecg_signal = df[column_name].values

    return np.array(ecg_signal)

def prepare_features_from_csv(
    target_csv_path,
    baseline_csv_path,
    column_name=None
):

    target_ecg = load_ecg_csv(
        target_csv_path,
        column_name
    )

    baseline_ecg = load_ecg_csv(
        baseline_csv_path,
        column_name
    )

    target_features = get_data_ecg(target_ecg)

    baseline_features = extract_baseline_features(
        baseline_ecg
    )

    normalized_features = normalize_features(
        target_features,
        baseline_features
    )

    return normalized_features